from torch.utils.data import Dataset
from PIL import Image

def load_dataset(dataset_name):
    if dataset_name.lower() == "mnist":
        train_dataset = MNIST("data/mnist", train=True, download=True)
        test_dataset = MNIST("data/mnist", train=False, download=True)
    elif dataset_name.lower() == "fmnist":
        pass
    elif dataset_name.lower() == "cifar10":
        pass
    else:
        raise NotImplementedError

    return train_dataset, test_dataset

def build_transform(dataset_name):
    if dataset_name == "MNIST":
        train_transform = transforms.ToTensor()
        test_transform = transforms.ToTensor()
        # mean, std = (0.5,), (0.5,)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        #     ])

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
        x = Image.fromarray(x.numpy(), mode="L")
        
        if self.transform:
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)