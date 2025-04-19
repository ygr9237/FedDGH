import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from libs import *
from hx_engines import server_test_fn

def metrics_aggregation_fn(results):
    evaluate_losses = [result["evaluate_loss"] * num_examples for num_examples, result in results]
    evaluate_accuracies = [result["evaluate_accuracy"] * num_examples for num_examples, result in results]
    total_examples = sum(num_examples for num_examples, result in results)

    aggregated_metrics = {
        "evaluate_loss": sum(evaluate_losses) / total_examples,
        "evaluate_accuracy": sum(evaluate_accuracies) / total_examples,
    }
    return aggregated_metrics

def aggregate(results):
    num_examples_total = sum(num_examples for (_, num_examples) in results)
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

class FedAvg(flwr.server.strategy.FedAvg):
    def __init__(self, sargs, server_model, save_ckp_dir,
                 query_dataloader, source_retrieval_dataloader, target_retrieval_dataloader,
                 *args, **kwargs):
        self.sargs = sargs
        self.server_model = server_model
        self.save_ckp_dir = save_ckp_dir
        self.query_dataloader = query_dataloader
        self.source_retrieval_dataloader = source_retrieval_dataloader
        self.target_retrieval_dataloader = target_retrieval_dataloader
        super().__init__(*args, **kwargs)
        self.previous_weights = None
        self.aggregated_accuracy = 0.0
        self.best_avg_mAP = 0.0
        self.best_target_mAP = 0.0

    def apply_pruning(self, model, g_model, percent: float, eps: float = 1e-10):
        model_layers = list(model.modules())
        g_layers = list(g_model.modules())
        layer_scores = {}
        weights = []

        for layer, g_layer in zip(model_layers, g_layers):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if hasattr(layer, 'requires_pruning') and not layer.requires_pruning:
                    continue

                theta_k = layer.weight.data
                theta_g = g_layer.weight.data
                combined_mag = (theta_k + theta_g).abs()
                sign_match = (theta_k * theta_g) > 0
                score = combined_mag * sign_match.float()
                layer_scores[layer] = score
                weights.append(score.flatten())

        if not weights:
            return

        all_scores = torch.cat(weights)
        norm_factor = torch.sum(all_scores) + eps
        all_scores.div_(norm_factor)

        num_keep = int(len(all_scores) * percent)
        if num_keep <= 0:
            return
        threshold = torch.topk(all_scores, num_keep, sorted=True).values[-1]

        for layer, g_layer in zip(model_layers, g_layers):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if hasattr(layer, 'requires_pruning') and not layer.requires_pruning:
                    continue
                if layer not in layer_scores:
                    continue

                score = layer_scores[layer]
                normalized_score = score / norm_factor

                mask = (normalized_score >= threshold).float()

                layer.weight.data = layer.weight.data * mask + g_layer.weight.data * (1 - mask)
    def aggregate_fit_ABPI(self, server_round, results, failures):
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        global_state_dict = self.server_model.state_dict()
        global_params = [v.cpu().numpy() for v in global_state_dict.values()]
        client_updates = []
        distances = []

        for _, fit_res in results:
            parameters = fit_res.parameters
            local_params = flwr.common.parameters_to_ndarrays(parameters)

            client_model = copy.deepcopy(self.server_model)
            client_state_dict = {k: torch.tensor(v) for k, v in zip(global_state_dict.keys(), local_params)}
            client_model.load_state_dict(client_state_dict)

            if not self.sargs.eta == 1:
                self.apply_pruning(client_model, self.server_model, self.sargs.eta)

            pruned_local_params = [v.cpu().numpy() for v in client_model.state_dict().values()]
            client_updates.append(pruned_local_params)

            delta = np.concatenate([np.square(np.ravel(lp - gp))
                                    for lp, gp in zip(pruned_local_params, global_params)])
            distances.append(np.linalg.norm(delta))

            if self.sargs.restore == 1:
                client_model.load_state_dict(client_state_dict)

        epsilon = 1e-6
        distances = [max(d, epsilon) for d in distances]
        total_distance = sum(distances)

        if self.previous_weights is None:
            self.previous_weights = [1 / len(results)] * len(results)

        updated_weights = [
            (1 - self.sargs.beta) * prev + self.sargs.beta * (d / total_distance)
            for prev, d in zip(self.previous_weights, distances)
        ]

        weight_sum = sum(updated_weights)
        normalized_weights = [w / weight_sum for w in updated_weights]

        weights_results = [
            (params, weight)
            for params, weight in zip(client_updates, normalized_weights)
        ]

        aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = flwr.common.ndarrays_to_parameters(aggregated_ndarrays)

        self.previous_weights = updated_weights
        metrics_aggregated = {}

        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

    def aggregate_fit(self, server_round, results, failures):
        aggregated_metrics = metrics_aggregation_fn([(result.num_examples, result.metrics) for _, result in results])

        if not self.sargs.beta == 0:
            aggregated_parameters = self.aggregate_fit_ABPI(server_round, results, failures)[0]
        else:
            aggregated_parameters = super().aggregate_fit(server_round, results, failures)[0]

        aggregated_parameters = flwr.common.parameters_to_ndarrays(aggregated_parameters)

        aggregated_keys = list(self.server_model.state_dict().keys())
        self.server_model.load_state_dict(
            collections.OrderedDict(
                {key: torch.tensor(value) for key, value in zip(aggregated_keys, aggregated_parameters)}),
            strict=False,
        )

        if aggregated_metrics["evaluate_accuracy"] > self.aggregated_accuracy:
            self.aggregated_accuracy = aggregated_metrics["evaluate_accuracy"]

        test_results = server_test_fn(
            self.query_dataloader,
            self.source_retrieval_dataloader,
            self.target_retrieval_dataloader,
            self.server_model,
            self.sargs.code_length,
            self.sargs.topk,
            device=torch.device("cuda"),
        )

        source_corss_map = test_results["source_cross_map"]
        target_single_map = test_results["target_single_map"]
        save_dir = "{}/{}/{}".format(self.save_ckp_dir, self.sargs.method, self.sargs.target)
        if self.sargs.save_model == 1:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if source_corss_map > self.best_avg_mAP:
            self.best_avg_mAP = source_corss_map
            if self.sargs.save_model == 1:
                torch.save(
                    self.server_model,
                    "{}/{}-server-best-source_corss_map.ptl".format(save_dir, self.sargs.target),
                )
        if target_single_map > self.best_target_mAP:
            self.best_target_mAP = target_single_map
            if self.sargs.save_model == 1:
                torch.save(
                    self.server_model,
                    "{}/{}-server-best-target_single_map.ptl".format(save_dir, self.sargs.target),
                )
        if server_round == self.sargs.num_rounds:
            logger.info("\n" + " " * 24 + " = " * 29 + "\n")
            logger.info(" " * 25 + "All round finish, best result: - source_cross_map: {:.4f},"
                                   " target_single_map: {:.4f}\n".format(self.best_avg_mAP, self.best_target_mAP))
            logger.info(" " * 24 + " = " * 29 + "\n")

        aggregated_parameters = [v.cpu().numpy() for v in self.server_model.state_dict().values()]
        return flwr.common.ndarrays_to_parameters(aggregated_parameters), {}