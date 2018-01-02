import torch
from torchvision import datasets, transforms

def get_loader(image_size, batch_size, num_workers):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5])

    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        normalize])

    svhn_train = datasets.SVHN(root='./svnh', download=True, transform=transform, split='train')
    svhn_test = datasets.SVHN(root='./svnh', download=True, transform=transform, split='test')

    svhn_loader_train = torch.utils.data.DataLoader(
        dataset=svhn_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    svhn_loader_test = torch.utils.data.DataLoader(
        dataset=svhn_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return svhn_loader_train, svhn_loader_test
