# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.utils.data import Dataset
import os

from PIL import Image
import json
import torch
import pathlib
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

ROOT_PATH = str(pathlib.Path(__file__).parent.parent.absolute())
COCO_ROOT_DIR = f"{ROOT_PATH}/genecis_evaluation/val2017"
FOCUS_OBJECT = f"{ROOT_PATH}/genecis_evaluation/annotations/focus_object.json"


def download_coco(root_dir: str):
    coco_zip_url = "http://images.cocodataset.org/zips/val2017.zip"
    with urlopen(coco_zip_url) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(str(pathlib.Path(root_dir).parent))


class COCODataset(Dataset):
    def __init__(self, transform=None, root_dir=COCO_ROOT_DIR) -> None:
        super().__init__()
        if not pathlib.Path(root_dir).exists():
            download_coco(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def load_sample(self, sample):
        val_image_id = sample["val_image_id"]
        fpath = os.path.join(self.root_dir, f"{int(val_image_id):012d}.jpg")
        img = Image.open(fpath)

        if self.transform is not None:
            img = self.transform(img)

        return img


class COCOValSubset(COCODataset):
    def __init__(self, val_split_path, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            val_samples = json.load(f)

        self.val_samples = val_samples
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        """
        Follow same return signature as CIRRSubset
        """

        sample = self.val_samples[index]
        reference = sample["reference"]

        target = sample["target"]
        gallery = sample["gallery"]
        caption = sample["condition"]

        reference, target = [self.load_sample(i) for i in (reference, target)]
        gallery = [self.load_sample(i) for i in gallery]

        if self.transform is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        # By construction, target_rank = 0
        return reference, caption, gallery_and_target, 0

    def __len__(self):
        return len(self.val_samples)


if __name__ == "__main__":
    dataset = COCOValSubset(FOCUS_OBJECT)
    y = dataset[22]
    refe, capt, gal_and_ta, _ = y

    print(capt)
    refe.show()
    gal_and_ta[0].show()
