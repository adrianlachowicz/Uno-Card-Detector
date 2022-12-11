import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from utils import get_img_data
from config import NUM_WORKERS, BATCH_SIZE


class UnoDataset(Dataset):
    """
    The Uno dataset.
    Arguments:
        train (bool) - Use dataset for training or validation.
        img_size (int) - Image size.
    """

    def __init__(self, train: bool = True, img_size: int = 256):
        if train is True:
            self.data_path = "./data/train/"
        else:
            self.data_path = "./data/valid/"

        self.img_size = img_size

        self.xml_paths = glob.glob(self.data_path + "*.xml")
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    def __len__(self):
        return len(self.xml_paths)

    def __getitem__(self, item):
        xml_path = self.xml_paths[item]

        img, labels, boxes = get_img_data(xml_path, self.data_path, self.img_size)

        img = self.transform(img)
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        target = {"boxes": boxes, "labels": labels}

        return img, target

    def collate_fn(self, batch):
        return list(zip(*batch))


def get_train_dataloader():
    """
    The function prepares and returns train dataloader.

    Returns:
        train_dataloader (DataLoader) - The train dataloader.
    """

    dataset = UnoDataset(train=True)
    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )

    return train_dataloader


def get_val_dataloader():
    """
    The function prepares and returns validation dataloader.

    Returns:
        val_dataloader (DataLoader) - The validation dataloader.
    """

    dataset = UnoDataset(train=False)
    val_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=dataset.collate_fn,
        shuffle=False
    )

    return val_dataloader
