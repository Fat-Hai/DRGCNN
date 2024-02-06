from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import os
import pickle
from torchvision import datasets
from utils.func import mean_and_std, print_dataset_info

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def data_transforms(cfg):
    data_aug = cfg.config_data.config_data_augmentation
    aug_args = cfg.config_data_augmentation_args

    operations = {
        'config_random_crop': random_apply(
            transforms.RandomResizedCrop(
                size=(cfg.config_data.config_input_size, cfg.config_data.config_input_size),
                scale=aug_args.config_random_crop.config_scale,
                ratio=aug_args.config_random_crop.config_ratio
            ),
            p=aug_args.config_random_crop.config_prob
        ),
        'config_horizontal_flip': transforms.RandomHorizontalFlip(
            p=aug_args.config_horizontal_flip.config_prob
        ),
        'config_vertical_flip': transforms.RandomVerticalFlip(
            p=aug_args.config_vertical_flip.config_prob
        ),
        'config_color_distortion': random_apply(
            transforms.ColorJitter(
                brightness=aug_args.config_color_distortion.config_brightness,
                contrast=aug_args.config_color_distortion.config_contrast,
                saturation=aug_args.config_color_distortion.config_saturation,
                hue=aug_args.config_color_distortion.config_hue
            ),
            p=aug_args.config_color_distortion.config_prob
        ),
        'config_rotation': random_apply(
            transforms.RandomRotation(
                degrees=aug_args.config_rotation.config_degrees,
                fill=0
            ),
            p=aug_args.config_rotation.config_prob
        ),
        'config_translation': random_apply(
            transforms.RandomAffine(
                degrees=0,
                translate=aug_args.config_translation.config_range,
                fill=0
            ),
            p=aug_args.config_translation.config_prob
        ),

    }


    augmentations = []
    for op in data_aug:
        if op not in operations:
            raise NotImplementedError('Not implemented data augmentation operations: {}'.format(op))
        augmentations.append(operations[op])

    normalization = [
        transforms.Resize((cfg.config_data.config_input_size, cfg.config_data.config_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.config_data.config_mean, cfg.config_data.config_std)
    ]

    train_preprocess = transforms.Compose([
        *augmentations,
        *normalization
    ])

    test_preprocess = transforms.Compose(normalization)

    return train_preprocess, test_preprocess


def random_apply(op, p):
    return transforms.RandomApply([op], p=p)


def simple_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
def generate_dataset(cfg):
    if cfg.config_data.config_mean == 'auto' or cfg.config_data.config_std == 'auto':
        mean, std = auto_statistics(
            cfg.config_base.config_data_path,
            cfg.config_data.config_input_size,
            cfg.config_train.config_batch_size,
            cfg.config_train.config_num_workers
        )
        cfg.config_data.config_mean = mean
        cfg.config_data.config_std = std

    train_transform, test_transform = data_transforms(cfg)

    datasets = generate_dataset_from_folder(
            cfg.config_base.config_data_path,
            train_transform,
            test_transform
     )

    print_dataset_info(datasets)
    return datasets


def auto_statistics(data_path,  input_size, batch_size, num_workers):
    print('Calculating mean and std of training set for data normalization.')
    transform = simple_transform(input_size)

    train_path = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(train_path, transform=transform)

    return mean_and_std(train_dataset, batch_size, num_workers)


def generate_dataset_from_folder(data_path, train_transform, test_transform):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    train_dataset = CustomizedImageFolder(train_path, train_transform, loader=pil_loader)
    test_dataset = CustomizedImageFolder(test_path, test_transform, loader=pil_loader)
    val_dataset = CustomizedImageFolder(val_path, test_transform, loader=pil_loader)

    return train_dataset, test_dataset, val_dataset


def generate_dataset_from_pickle(pkl, train_transform, test_transform):
    data = pickle.load(open(pkl, 'rb'))
    train_set, test_set, val_set = data['train'], data['test'], data['val']

    train_dataset = DatasetFromDict(train_set, train_transform, loader=pil_loader)
    test_dataset = DatasetFromDict(test_set, test_transform, loader=pil_loader)
    val_dataset = DatasetFromDict(val_set, test_transform, loader=pil_loader)

    return train_dataset, test_dataset, val_dataset

class CustomizedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader):
        super(CustomizedImageFolder, self).__init__(root, transform, target_transform, loader=loader)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class DatasetFromDict(Dataset):
    def __init__(self, imgs, transform=None, loader=pil_loader):
        super(DatasetFromDict, self).__init__()
        self.imgs = imgs
        self.loader = loader
        self.transform = transform
        self.targets = [img[1] for img in imgs]
        self.classes = sorted(list(set(self.targets)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

