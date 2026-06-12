import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import sys

from _datasets import register_dataset
from _datasets._utils import BaseDataset
from utils.global_consts import DATASET_PATH

TRANSFORMS = {
    "default_train": lambda x: x,
    "default_test": lambda x: x,
}


class MyTinyImageNet(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: transforms = None,
        download: bool = True,
    ) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

        if not os.path.exists(os.path.join(self.root, "TinyImageNet", "processed", "x_train_01.npy")) and self.download:
            import subprocess
            import zipfile
            from PIL import Image as PILImage

            raw_zip = os.path.join(self.root, "tiny-imagenet-200.zip")
            raw_dir = os.path.join(self.root, "tiny-imagenet-200")
            out_dir = os.path.join(self.root, "TinyImageNet", "processed")
            os.makedirs(out_dir, exist_ok=True)

            if not os.path.exists(raw_zip):
                print("Downloading TinyImageNet...", file=sys.stderr)
                subprocess.run(
                    ["wget", "-q", "--show-progress", "http://cs231n.stanford.edu/tiny-imagenet-200.zip", "-O", raw_zip],
                    check=True,
                )

            if not os.path.exists(raw_dir):
                print("Unzipping TinyImageNet...", file=sys.stderr)
                with zipfile.ZipFile(raw_zip, "r") as z:
                    z.extractall(self.root)

            print("Preprocessing TinyImageNet into .npy chunks...", file=sys.stderr)

            classes = sorted(os.listdir(os.path.join(raw_dir, "train")))
            class_to_idx = {c: i for i, c in enumerate(classes)}

            train_imgs, train_labels = [], []
            for cls in classes:
                img_dir = os.path.join(raw_dir, "train", cls, "images")
                for fname in sorted(os.listdir(img_dir)):
                    img = PILImage.open(os.path.join(img_dir, fname)).convert("RGB")
                    train_imgs.append(np.array(img, dtype=np.float32) / 255.0)
                    train_labels.append(class_to_idx[cls])
            train_imgs = np.array(train_imgs)
            train_labels = np.array(train_labels)
            for i, (x, y) in enumerate(zip(np.array_split(train_imgs, 20), np.array_split(train_labels, 20))):
                np.save(os.path.join(out_dir, f"x_train_{i+1:02d}.npy"), x)
                np.save(os.path.join(out_dir, f"y_train_{i+1:02d}.npy"), y)

            val_imgs, val_labels = [], []
            with open(os.path.join(raw_dir, "val", "val_annotations.txt")) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    fname, cls = parts[0], parts[1]
                    img = PILImage.open(os.path.join(raw_dir, "val", "images", fname)).convert("RGB")
                    val_imgs.append(np.array(img, dtype=np.float32) / 255.0)
                    val_labels.append(class_to_idx[cls])
            val_imgs = np.array(val_imgs)
            val_labels = np.array(val_labels)
            for i, (x, y) in enumerate(zip(np.array_split(val_imgs, 20), np.array_split(val_labels, 20))):
                np.save(os.path.join(out_dir, f"x_val_{i+1:02d}.npy"), x)
                np.save(os.path.join(out_dir, f"y_val_{i+1:02d}.npy"), y)

            print("Preprocessing done.", file=sys.stderr)

        self.data = []
        for num in range(20):
            self.data.append(
                np.load(
                    os.path.join(
                        self.root + "/TinyImageNet",
                        "processed/x_%s_%02d.npy" % ("train" if self.train else "val", num + 1),
                    )
                )
            )
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(
                np.load(
                    os.path.join(
                        self.root + "/TinyImageNet",
                        "processed/y_%s_%02d.npy" % ("train" if self.train else "val", num + 1),
                    )
                )
            )
        self.targets = np.concatenate(np.array(self.targets)).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


@register_dataset("seq-tinyimagenet")
class SequentialTinyImageNet(BaseDataset):
    N_CLASSES_PER_TASK = 40
    N_TASKS = 5

    TRAIN_TRANSFORM = transforms.ToTensor()
    TEST_TRANSFORM = transforms.ToTensor()
    BASE_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    INPUT_SHAPE = (64, 64, 3)

    def __init__(
        self,
        num_clients: int,
        batch_size: int,
        train_transform: str = "default_train",
        test_transform: str = "default_test",
        partition_mode: str = "distribution",
        distribution_alpha: float = 0.05,
        class_quantity: int = 4,
    ):
        super().__init__(
            num_clients,
            batch_size,
            partition_mode,
            distribution_alpha,
            class_quantity,
        )
        self.train_transf = train_transform
        self.test_transf = test_transform

        for split in ["train", "test"]:
            dataset = MyTinyImageNet(
                DATASET_PATH,
                train=True if split == "train" else False,
                download=True,
                transform=self.BASE_TRANSFORM,
            )
            dataset.targets = np.array(dataset.targets).astype(np.int64)
            setattr(self, f"{split}_dataset", dataset)

        self._split_fcil(num_clients, partition_mode, distribution_alpha, class_quantity)

        for split in ["train", "test"]:
            getattr(self, f"{split}_dataset").data = None
            getattr(self, f"{split}_dataset").targets = None

    def train_transform(self, x):
        return TRANSFORMS[self.train_transf](x)

    def test_transform(self, x):
        return TRANSFORMS[self.test_transf](x)


@register_dataset("joint-tinyimagenet")
class JointTinyImageNet(SequentialTinyImageNet):
    N_CLASSES_PER_TASK = 200
    N_TASKS = 1
