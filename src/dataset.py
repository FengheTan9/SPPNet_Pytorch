from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def creat_imagenet_train_dataset(train_dir, batch):
    # create dataset and data loader
    dataset_224 = datasets.ImageFolder(train_dir, transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation= 0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Dataset 224 size created')
    dataloader_224 = data.DataLoader(
        dataset_224,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
        batch_size=batch,
    )
    print('Dataloader 224 size created')
    dataset_180 = datasets.ImageFolder(train_dir, transforms.Compose([

        transforms.RandomResizedCrop(size=224),
        transforms.Resize(180),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Dataset 180 size created')
    dataloader_180 = data.DataLoader(
        dataset_180,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
        batch_size=batch,
    )
    print('Dataloader 180 size created')
    return dataloader_224,dataloader_180


def creat_imagenet_val_dataset(val_dir, batch):
    # create dataset and data loader
    dataset = datasets.ImageFolder(val_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    print('Dataset 224 size created')
    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
        batch_size=batch)
    print('Dataloader 224 size created')
    return dataloader
