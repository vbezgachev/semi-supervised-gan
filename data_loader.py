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

    svnh = datasets.SVHN(root='./svnh', download=True, transform=transform)

    svnh_loader = torch.utils.data.DataLoader(
        dataset=svnh,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return svnh_loader