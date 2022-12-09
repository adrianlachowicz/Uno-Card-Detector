import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from utils import get_img_data


class UnoDataset(Dataset):
    """
    The Uno dataset.
    Arguments:
        train (bool) - Use dataset for training or validation.
    """

    def __init__(self, train: bool = True):
        if train is True:
            self.data_path = "./data/train/"
        else:
            self.data_path = "./data/valid/"

        self.xml_paths = glob.glob(self.data_path + "*.xml")
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.xml_paths)

    def __getitem__(self, item):
        xml_path = self.xml_paths[item]

        img, labels, boxes = get_img_data(xml_path, self.data_path)

        img = self.transform(img)
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        target = {"boxes": boxes, "labels": labels}

        return img, target

    def collate_fn(self, batch):
        return list(zip(*batch))
