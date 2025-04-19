import os, sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from hx_ABPI import *
from hx_data import ImageDataset, load_test_data
from hx_engines import server_test_fn
from hx_loss import CosSim

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="127.0.0.1"), parser.add_argument("--server_port",
                                                                                                    type=int,
                                                                                                    default=9990)
    (parser.add_argument("--dataset", type=str, default="PACS"),
     parser.add_argument("--subdataset", type=str, default="cartoon"))
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=50)

    parser.add_argument("--code_length", type=int, default=64)
    parser.add_argument("--last_node", type=int, default=768)
    parser.add_argument("--target", type=str, default="art_painting")
    parser.add_argument('--setting', type=str, default='cross_test')
    parser.add_argument('-w', '--num-workers', default=1, type=int,
                        help='Number of loading data threads.(default: 1)')
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument('-k', '--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: -1)')
    parser.add_argument('--root', type=str, default='../../datasets/', help="root")
    parser.add_argument('-a', '--arch', default='convnext_tiny', type=str,
                        choices=['resnet18', 'resnet50', 'convnext_tiny', 'convnext_base', 'vgg16'],
                        help='Model architecture to use')

    parser.add_argument("--method", type=str, default="FedDGH")
    parser.add_argument("--save_model", type=int, default=0, help='1 to save, 0 to no save')

    parser.add_argument("--strategy", type=str, default="ABPI", help="Hyperparameter strategy")
    parser.add_argument("--eta", type=float, default=0.7)
    parser.add_argument("--alfa", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--restore", type=int, default=0)

    args = parser.parse_args()
    args.root = os.path.join(args.root, args.dataset)
    print("=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =")
    logger.info(
        "strategy:{} - eta:{} - alfa:{} - beta:{} - restore:{} - save_model:{}".format
        (args.strategy, args.eta, args.alfa, args.beta, args.restore, args.save_model))

    if args.arch == 'resnet18':
        server_model = create_model('resnet18', pretrained=True, num_classes=0)
        args.last_node = 512
    elif args.arch == 'resnet50':
        server_model = create_model('resnet50', pretrained=True, num_classes=0)
        args.last_node = 2048
    elif args.arch == 'convnext_tiny':
        server_model = create_model('convnext_tiny', pretrained=True, num_classes=0)
        args.last_node = 768
    elif args.arch == 'convnext_base':
        server_model = create_model('convnext_base', pretrained=True, num_classes=0)
        args.last_node = 1024
    elif args.arch == 'vgg16':
        server_model = torchvision.models.vgg16(pretrained=True)
        server_model.classifier = server_model.classifier[:-3]
        args.last_node = 4096
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    server_model.hash_layer = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Linear(args.last_node, args.code_length),
        nn.Tanh()
    )

    server_model.ce_fc = CosSim(args.code_length, args.num_classes, learn_cent=False)

    initial_parameters = [value.cpu().numpy() for key, value in server_model.state_dict().items()]
    initial_parameters = flwr.common.ndarrays_to_parameters(initial_parameters)
    save_ckp_dir = "../../ckps/{}".format(args.dataset)
    if not os.path.exists(save_ckp_dir):
        os.makedirs(save_ckp_dir)

    logger.info("Dataset: {}; Test Target Domain: {}".format(args.dataset, args.target))
    print("=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =")

    cross_test_database = os.path.join(args.root, "{}_cross_test".format(args.target))
    target_data = os.path.join(args.root, args.target)

    query_dataloader, source_retrieval_dataloader \
        = load_test_data(cross_test_database, target_data, args.batch_size, args.num_workers, args.num_classes,
                    task="cross_test")
    _, target_retrieval_dataloader \
        = load_test_data(target_data, target_data, args.batch_size, args.num_workers, args.num_classes,
                    task="single_test")

    flwr.server.start_server(
        server_address="{}:{}".format(args.server_address, args.server_port),
        config=flwr.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=FedAvg(
            min_available_clients=args.num_clients, min_fit_clients=args.num_clients,
            server_model=server_model,
            initial_parameters=initial_parameters,
            save_ckp_dir=save_ckp_dir,
            query_dataloader=query_dataloader,
            source_retrieval_dataloader=source_retrieval_dataloader,
            target_retrieval_dataloader=target_retrieval_dataloader,
            sargs=args
        ),
    )
