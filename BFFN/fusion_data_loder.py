from PIL import Image
from torch.utils.data import Dataset

class PairedEyeDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super(PairedEyeDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.classes = [0, 1, 2, 3, 4]

    def __getitem__(self, index):
        x, x_aux, y = self.dataset[index]
        x = self.pil_loader(x)
        x_aux = self.pil_loader(x_aux)

        if self.transform:
            x = self.transform(x)
            x_aux = self.transform(x_aux)
        return x, x_aux, y

    def __len__(self):
        return len(self.dataset)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')