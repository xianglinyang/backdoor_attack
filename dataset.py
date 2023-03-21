from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms
def load_dataset(dataset_name):
    if dataset_name.lower() == "mnist":
        train_dataset = torchvision.datasets.MNIST("data/mnist", train=True, download=True)
        test_dataset = torchvision.datasets.MNIST("data/mnist", train=False, download=True)
        train_data = train_dataset.data.numpy()
        train_labels = train_dataset.targets
        test_data = test_dataset.data.numpy()
        test_labels = test_dataset.targets
    elif dataset_name.lower() == "fmnist":
        pass
    elif dataset_name.lower() == "cifar10":
        pass
    else:
        raise NotImplementedError

    return train_data, train_labels, test_data, test_labels

def build_transform(dataset_name):
    if dataset_name == "MNIST":
        # train_transform = transforms.ToTensor()
        # test_transform = transforms.ToTensor()
        mean, std = (0.5,), (0.5,)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    elif dataset_name == "CIFAR10":
        pass
    else:
        raise NotImplementedError

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    #     ])
    # mean = torch.as_tensor(mean)
    # std = torch.as_tensor(std)
    # detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # you can use detransform to recover the image
    
    return train_transform, test_transform


class DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        x = Image.fromarray(x, mode="L")
        
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)