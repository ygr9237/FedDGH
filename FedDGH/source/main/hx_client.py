import os, sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from hx_ABPI import *
from hx_data import ImageDataset, load_source_data
from hx_engines import client_fit_fn
from hx_loss import CosSim

class Client(flwr.client.NumPyClient):
    def __init__(self,
                 train,
                 query,
                 retrieval,
                 code_length,
                 topk,
                 num_epochs,
                 client_model,
                 client_optim,
                 save_ckp_dir,
                 args,
                 ):

        self.train = train
        self.query = query
        self.retrieval = retrieval
        self.code_length = code_length
        self.topk = topk
        self.num_epochs = num_epochs
        self.client_model = client_model
        self.client_optim = client_optim
        self.save_ckp_dir = save_ckp_dir
        self.args = args

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            client_optim,
            T_max=client_optim.num_rounds,
        )

    def get_parameters(self,
                       config,
                       ):
        parameters = [value.cpu().numpy() for key, value in self.client_model.state_dict().items()]
        return parameters

    def fit(self,
            parameters, config,
            ):
        logger.info("\n" + " " * 25 + '-----the client {}-----'.format(self.args.client))
        logger.info(" " * 25 + 'retrieval setting: {}'.format(self.args.setting))
        logger.info(" " * 25 + 'dataset: {}'.format(self.args.dataset))
        logger.info(" " * 25 + 'batch_size: {}'.format(self.args.batch_size))
        logger.info(" " * 25 + 'num_classes: {}'.format(self.args.num_classes))
        logger.info(" " * 25 + 'code_length: {}'.format(self.args.code_length))
        logger.info(" " * 25 + 'querry(source/10%): {}'.format(self.args.subdataset.split('/')[-1]))
        logger.info(" " * 25 + 'train/retrieval database(source/90%): {}'.format(self.args.subdataset.split('/')[-1]))
        logger.info(" " * 25 + 'net: {}'.format(self.args.arch))
        logger.info(" " * 25 + 'train epoch: {}'.format(self.args.num_epochs))
        logger.info(" " * 25 + 'leaning rate: {}, weight_decay: {}'.format
        (self.args.leaning_rate, self.args.weight_decay))
        logger.info(" " * 25 + 'eta: {}, alfa: {}'.format(self.args.eta, self.args.alfa))

        keys = [key for key in self.client_model.state_dict().keys()]
        self.client_model.load_state_dict(
            collections.OrderedDict({key: torch.tensor(value) for key, value in zip(keys, parameters)}),
            strict=False,
        )

        self.lr_scheduler.step()

        results = client_fit_fn(
            self.args,
            self.train,
            self.query,
            self.retrieval,
            self.code_length,
            self.topk,
            self.num_epochs,
            self.client_model,
            self.client_optim,
            device=torch.device("cuda"),
        )

        return self.get_parameters({}), len(self.train.dataset), results


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    pl.seed_everything(25)
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="127.0.0.1"), parser.add_argument("--server_port",
                                                                                                    type=int,
                                                                                                    default=9990)
    (parser.add_argument("--dataset", type=str, default="PACS"),
     parser.add_argument("--subdataset", type=str, default="cartoon"))
    parser.add_argument("--client", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--current_round", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=5)

    parser.add_argument("--code_length", type=int, default=64)
    parser.add_argument("--last_node", type=int, default=768)
    parser.add_argument('--setting', type=str, default='train_val')
    parser.add_argument('-w', '--num-workers', default=1, type=int)
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument('-k', '--topk', default=-1, type=int, help='Calculate map of top k.(default: -1)')
    parser.add_argument('--root', type=str, default='../../datasets/', help="root")
    parser.add_argument('-a', '--arch', default='convnext_tiny', type=str,
                        choices=['resnet18', 'resnet50', 'convnext_tiny', 'convnext_base', 'vgg16'],
                        help='Model architecture to use')
    parser.add_argument("--leaning_rate", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--eta", type=float, default=0.7)
    parser.add_argument("--alfa", type=float, default=0.01)

    args = parser.parse_args()
    args.root = os.path.join(args.root, args.dataset)
    data_subdataset = os.path.join(args.root, args.subdataset)

    if args.setting == 'train_val':
        train_dataloader, query_dataloader, retrieval_dataloader \
            = load_source_data(data_subdataset, args.batch_size, args.num_workers, args.num_classes, task=args.setting)
    else:
        raise ValueError(f"Unsupported data-setting: {args.setting}")

    if args.arch == 'resnet18':
        client_model = create_model('resnet18', pretrained=True, num_classes=0)
        args.last_node = 512
    elif args.arch == 'resnet50':
        client_model = create_model('resnet50', pretrained=True, num_classes=0)
        args.last_node = 2048
    elif args.arch == 'convnext_tiny':
        client_model = create_model('convnext_tiny', pretrained=True, num_classes=0)
        args.last_node = 768
    elif args.arch == 'convnext_base':
        client_model = create_model('convnext_base', pretrained=True, num_classes=0)
        args.last_node = 1024
    elif args.arch == 'vgg16':
        client_model = torchvision.models.vgg16(pretrained=True)
        client_model.classifier = client_model.classifier[:-3]
        args.last_node = 4096
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    client_model.hash_layer = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Linear(args.last_node, args.code_length),
        nn.Tanh()
    )

    for layer in client_model.hash_layer.modules():
        layer.requires_pruning = False

    client_model.ce_fc = CosSim(args.code_length, args.num_classes, learn_cent=False)

    client_optim = optim.SGD(
        client_model.parameters(), weight_decay=args.weight_decay,
        lr=args.leaning_rate, momentum=0.9,
    )

    client_optim.num_rounds = args.num_rounds

    save_ckp_dir = "../../ckps/{}".format(args.dataset)
    if not os.path.exists(save_ckp_dir):
        os.makedirs(save_ckp_dir)
    client = Client(
        train_dataloader,
        query_dataloader,
        retrieval_dataloader,
        args.code_length,
        args.topk,
        args.num_epochs,
        client_model,
        client_optim,
        save_ckp_dir,
        args,
    )
    flwr.client.start_client(
        server_address="{}:{}".format(args.server_address, args.server_port),
        client=client.to_client(),
    )
