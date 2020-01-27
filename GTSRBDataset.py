import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class GTSRBDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        data = np.load(npz_file)
        self.images = data['images']
        self.labels= data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label
