import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from libs import *
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

class Onehot(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        target_onehot = torch.zeros(self.num_classes)
        target_onehot[sample] = 1

        return target_onehot

def train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def query_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

def load_source_data(source_list, batch_size, num_workers, num_classes, task='train_val'):
    ImageDataset.init(source_list, source_list, task)
    query_dataset = ImageDataset('query', num_classes, query_transform(), target_transform=Onehot(num_classes))
    train_dataset = ImageDataset('train', num_classes, train_transform(), target_transform=Onehot(num_classes))
    retrieval_dataset = ImageDataset('retrieval', num_classes, query_transform(), target_transform=Onehot(num_classes))
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )

    return train_dataloader, query_dataloader, retrieval_dataloader

def load_test_data(source_list, target_list, batch_size, num_workers, num_classes, task='cross_test'):
    ImageDataset.init(source_list, target_list, task)
    query_dataset = ImageDataset('query', num_classes, query_transform(), target_transform=Onehot(num_classes))
    retrieval_dataset = ImageDataset('retrieval', num_classes, query_transform(), target_transform=Onehot(num_classes))
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )

    return query_dataloader, retrieval_dataloader

def load_visualization_data(source_list, batch_size, num_workers, num_classes, task='Visualization'):
    ImageDataset.init(source_list, source_list, task)
    visualization_data = ImageDataset('train', num_classes, train_transform(), target_transform=Onehot(num_classes))

    visualization_dataloader = DataLoader(
        visualization_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
    )

    return visualization_dataloader


class ImageDataset(Dataset):
    def __init__(self, mode, num_classes, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.num_classes = num_classes

        if mode == 'train':
            self.data = ImageDataset.TRAIN_S_DATA
            self.targets = ImageDataset.TRAIN_S_TARGETS
        elif mode == 'query':
            self.data = ImageDataset.QUERY_DATA
            self.targets = ImageDataset.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = ImageDataset.RETRIEVAL_DATA
            self.targets = ImageDataset.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.target_transform(self.targets[index]), index

    def __len__(self):
        return self.data.shape[0]

    def get_targets(self):
        one_hot = torch.zeros((self.targets.shape[0], self.num_classes))
        for i in range(self.targets.shape[0]):
            one_hot[i, :] = self.target_transform(self.targets[i])
        return one_hot

    @staticmethod
    def init(source_list, target_list, task):
        source_data = []
        source_label = []
        target_data = []
        target_label = []
        source_list += '.txt'
        with open(source_list, 'r') as f:
            for line in f:
                source_data.append(line.split()[0].replace('/datasets', '/datasets'))
                source_label.append(int(line.split()[1]))

        target_list += '.txt'
        with open(target_list, 'r') as f:
            for line in f:
                target_data.append(line.split()[0].replace('/datasets', '/datasets'))
                target_label.append(int(line.split()[1]))

        source_data = np.array(source_data)
        source_label = np.array(source_label)
        target_data = np.array(target_data)
        target_label = np.array(target_label)

        if task == 'train_val':
            perm_index = np.random.permutation(source_data.shape[0])
            query_index = perm_index[:int(0.1 * source_data.shape[0])]
            train_database_index = perm_index[int(0.1 * source_data.shape[0]):]

            ImageDataset.QUERY_DATA = source_data[query_index]
            ImageDataset.QUERY_TARGETS = source_label[query_index]

            ImageDataset.TRAIN_S_DATA = source_data[train_database_index]
            ImageDataset.TRAIN_S_TARGETS = source_label[train_database_index]

            ImageDataset.RETRIEVAL_DATA = source_data[train_database_index]
            ImageDataset.RETRIEVAL_TARGETS = source_label[train_database_index]

            logger.info('Query Num: {}'.format(ImageDataset.QUERY_DATA.shape[0]))
            logger.info('Train / Retrieval Database Num: {}\n'.format(ImageDataset.RETRIEVAL_DATA.shape[0]))

        elif task == 'cross_test':
            target_perm_index = np.random.permutation(target_data.shape[0])
            query_index = target_perm_index[:int(0.1 * target_data.shape[0])]

            ImageDataset.QUERY_DATA = target_data[query_index]
            ImageDataset.QUERY_TARGETS = target_label[query_index]

            ImageDataset.RETRIEVAL_DATA = source_data
            ImageDataset.RETRIEVAL_TARGETS = source_label

            logger.info('Query Num: {}'.format(ImageDataset.QUERY_DATA.shape[0]))
            logger.info('Retrieval Source Database Num: {}\n'.format(ImageDataset.RETRIEVAL_DATA.shape[0]))

        elif task == 'single_test':
            perm_index = np.random.permutation(target_data.shape[0])
            query_index = perm_index[:int(0.1 * target_data.shape[0])]
            database_index = perm_index[int(0.1 * target_data.shape[0]):]

            ImageDataset.QUERY_DATA = target_data[query_index]
            ImageDataset.QUERY_TARGETS = target_label[query_index]

            ImageDataset.RETRIEVAL_DATA = target_data[database_index]
            ImageDataset.RETRIEVAL_TARGETS = target_label[database_index]

            logger.info('Query Num: {}'.format(ImageDataset.QUERY_DATA.shape[0]))
            logger.info('Retrieval Target Database Num: {}\n'.format(ImageDataset.RETRIEVAL_DATA.shape[0]))

        elif task == 'Visualization':

            ImageDataset.TRAIN_S_DATA = source_data
            ImageDataset.TRAIN_S_TARGETS = source_label

            logger.info('Database Num: {}\n'.format(ImageDataset.TRAIN_S_DATA.shape[0]))
