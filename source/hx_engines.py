import os, sys
from libs import *
from torch.utils.data.dataset import ConcatDataset
import math

def client_fit_fn(
    args,
    train_s_dataloader,
    query_dataloader,
    retrieval_dataloader,
    code_length,
    topk,
    num_epochs,
    client_model,
    client_optim,
    device = torch.device("cpu"),
):
    logger.info("\n" + " " * 25 + "Start Client {} Fitting ...\n".format(args.client) + " " * 25 + " = " * 20)

    args.current_round += 1
    client_model = client_model.to(device)

    server_model = copy.deepcopy(client_model)
    for parameter in server_model.parameters():
        parameter.requires_grad = False

    max_iter_t = num_epochs * len(train_s_dataloader)
    interval_iter = max_iter_t // 3
    iter_num = 0
    loss = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        client_model.train()
        for data_s, target_s, index in train_s_dataloader:
            data_s = data_s.to(device)
            label_s = target_s.to(device)
            features = client_model(data_s)
            logit1 = client_model.ce_fc(client_model.hash_layer(features))

            l2_loss = 0
            for client_param, server_param in zip(client_model.parameters(), server_model.parameters()):
                if client_param.requires_grad:
                    diff = client_param - server_param
                    l2_loss += torch.sum(diff ** 2)

            loss = criterion(logit1, label_s.argmax(1)) + args.alfa * l2_loss

            loss.backward()

            if not args.eta == 1:
                apply_pruning(client_model, server_model, args.eta)

            iter_num += 1
            if iter_num % interval_iter == 0 or iter_num == max_iter_t:
                logger.info(" " * 25 + 'client {}: database:{}; Iter:{}/{}; loss:{:.4f}'.format
                            (args.client, args.subdataset, iter_num, max_iter_t, loss.item()))

            client_optim.step(), client_optim.zero_grad()

    mAP = evaluate(client_model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   save=False,
                   )

    logger.info(" " * 25 + "Round {}: Training finish, {:<8} client {}: database:{}, map:{:.4f}".format
                (args.current_round, "evaluate", args.client, args.subdataset, mAP))

    logger.info("\n" + " " * 25 + "Finish Client {} Fitting ...\n".format(args.client) + " " * 25 + " = " * 20)

    return {
        "evaluate_loss":loss.item(), "evaluate_accuracy":mAP.item()
    }

def server_test_fn(
    query_dataloader,
    source_retrieval_dataloader,
    target_retrieval_dataloader,
    server_model,
    code_length,
    topk,
    device = torch.device("cpu"),
):
    server_model = server_model.to(device)
    source_corss_map = evaluate(server_model,
                       query_dataloader,
                       source_retrieval_dataloader,
                       code_length,
                       device,
                       topk,
                       save=False,
                       )
    target_single_map = evaluate(server_model,
                   query_dataloader,
                   target_retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   save=False,
                   )
    logger.info("\n" + " " * 25 + "Testing finish, {:<8}: - source_cross_map:{:.4f} - target_single_map:{:.4f}\n"
                .format("evaluate", source_corss_map, target_single_map))

    return {
        "source_cross_map": source_corss_map.item(), "target_single_map": target_single_map.item()
    }

def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, save):
    model.eval()

    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)

    onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
    onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)

    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )

    model.train()

    return mAP

def generate_code(model, dataloader, code_length, device):

    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            features = model(data)
            outputs = model.hash_layer(features)
            code[index, :] = outputs.sign().cpu()

    return code


def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=-1,
                           ):
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]
        retrieval_cnt = retrieval.sum().int().item()

        if retrieval_cnt == 0:
            continue

        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()
        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP

def apply_pruning(model, g_model, percent, eps=1e-10):
    weight = dict()
    modules = list(model.modules())

    for idx, (layer, g_layer) in enumerate(zip(model.modules(), g_model.modules())):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if hasattr(layer, 'requires_pruning') and not layer.requires_pruning:
                continue
            if layer.weight.grad is None:
                continue
            else:
                theta_k = layer.weight.data
                theta_g = g_layer.weight.data
                combined_magnitude = (theta_k + theta_g).abs()  # |θ^k + θ^g|
                sign_consistency = (theta_k * theta_g) > 0  # sign(θ^k·θ^g) = +1

                weight_score = combined_magnitude * sign_consistency.float()
                weight[modules[idx]] = weight_score

    all_scores = torch.cat([torch.flatten(x) for x in weight.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    all_scores.div_(norm_factor)

    num_params = int(len(all_scores) * percent)
    if num_params <= 0:
        return
    threshold, _ = torch.topk(all_scores, num_params, sorted=True)
    acceptable_score = threshold[-1]
    keep_masks = dict()
    for m, g in weight.items():
        keep_masks[m] = ((g / norm_factor) >= acceptable_score).float()

    for m in keep_masks.keys():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            mask = keep_masks[m]
            m.weight.grad.mul_(mask)

    return keep_masks