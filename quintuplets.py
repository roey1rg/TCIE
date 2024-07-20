from dataclasses import dataclass
import json
import os
from diffusers import DiffusionPipeline
import torch
from quintuplet_prompts_claude import QUINUPLETS_RAW
from PIL import Image
from tqdm import tqdm
import numpy as np
from nlp_utils import add_text_to_image
from IPython.display import display
from nlp_utils import get_project_root

QUINTUPLETS_DATASET_PATH = '/home/dcor/roeyron/datasets/quintuplets_v1'


@dataclass
class Quadruplet:
    query: Image.Image
    positive: Image.Image
    negative: Image.Image
    prompt: str


@dataclass
class QuadrupletId:
    quintuplet_id: str
    which: str


def load_quadruplet(qn_dataset_path: str, qd_id: QuadrupletId) -> Quadruplet:
    qn = Quintuplet.load(qn_dataset_path, qd_id.quintuplet_id)
    qd = qn.get_quadruplet(qd_id.which)
    return qd


@dataclass
class Quintuplet:
    """
    QARD - Quintuplet Attribute Relationships Dataset
    """
    anchor_image: Image.Image
    gamma_image: Image.Image
    delta_image: Image.Image

    anchor_gamma_shared_text: str
    anchor_delta_shared_text: str
    raw_data: dict

    @staticmethod
    def get_dp_path(dataset_path: str, dp_id: str):
        return os.path.join(dataset_path, dp_id)

    def save(self, dataset_path: str, dp_id: str):
        dp_path = self.get_dp_path(dataset_path, dp_id)
        os.mkdir(dp_path)
        self.anchor_image.save(os.path.join(dp_path, "anchor.jpg"), format="JPEG")
        self.gamma_image.save(os.path.join(dp_path, "gamma.jpg"), format="JPEG")
        self.delta_image.save(os.path.join(dp_path, "delta.jpg"), format="JPEG")

        data = {
            "anchor_gamma_shared_text": self.anchor_gamma_shared_text,
            "anchor_delta_shared_text": self.anchor_delta_shared_text,
            "raw_data": self.raw_data,
        }
        json_path = os.path.join(dp_path, "data.json")
        with open(json_path, "w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load(cls, dataset_path: str, dp_id: str):
        dp_path = cls.get_dp_path(dataset_path, dp_id)
        anchor_image = Image.open(os.path.join(dp_path, "anchor.jpg"))
        gamma_image = Image.open(os.path.join(dp_path, "gamma.jpg"))
        delta_image = Image.open(os.path.join(dp_path, "delta.jpg"))

        json_path = os.path.join(dp_path, "data.json")
        with open(json_path, "r") as file:
            data = json.load(file)
        return cls(
            anchor_image=anchor_image,
            gamma_image=gamma_image,
            delta_image=delta_image,
            anchor_gamma_shared_text=data["anchor_gamma_shared_text"],
            anchor_delta_shared_text=data["anchor_delta_shared_text"],
            raw_data=data,
        )

    def get_quadruplet(self, which: str) -> Quadruplet:
        if which == "gamma_positive":
            return Quadruplet(
                self.anchor_image,
                self.gamma_image,
                self.delta_image,
                self.anchor_gamma_shared_text,
            )
        elif which == "delta_positive":
            return Quadruplet(
                self.anchor_image,
                self.delta_image,
                self.gamma_image,
                self.anchor_delta_shared_text,
            )
        else:
            raise ValueError()


def get_splits_ids():
    json_path = os.path.join(get_project_root(), 'train_test_split_ids.json')
    with open(json_path, 'r') as file:
        splits_ids = json.load(file)
    return {k: [int(e) for e in v] for k, v in splits_ids.items()}
    # all_ids = sorted(os.listdir(QUINTUPLETS_DATASET_PATH))
    # return {'train': all_ids[50:], 'test': all_ids[:50]}


def visualize_quintuplet(qn: Quintuplet, show=True, save_path=None):
    images = [qn.gamma_image, qn.anchor_image, qn.delta_image]
    images = [img.resize((512, 512)) for img in images]
    rd = qn.raw_data['raw_data']
    texts = [rd['gamma'], rd['anchor'], rd['delta']]
    images = [add_text_to_image(img, txt, font_size=22) for img, txt in zip(images, texts)]
    img = Image.fromarray(np.concatenate(images, axis=1))
    img = add_text_to_image(img, qn.anchor_gamma_shared_text, vertical_position='bottom', horizontal_position=1/3, alignment='center', font_size=22)
    img = add_text_to_image(img, qn.anchor_delta_shared_text, vertical_position='bottom', horizontal_position=2/3, alignment='center', font_size=22)
    if save_path is not None:
        img.convert('RGB').save(save_path)
    if show:
        display(img)


def visualize_quadruplet(dataset_path, qd_id: QuadrupletId):
    qn = Quintuplet.load(dataset_path, qd_id.quintuplet_id)
    qd = qn.get_quadruplet(qd_id.which)
    images = [qd.query, qd.positive, qd.negative]
    images = [img.resize((512, 512)) for img in images]
    rd = qn.raw_data['raw_data']
    rd = qn.raw_data['raw_data']
    q_prompt = rd['anchor']
    if qd_id.which == 'gamma_positive':
        p_prompt = rd['gamma']
        n_prompt = rd['delta']
    else:
        p_prompt = rd['delta']
        n_prompt = rd['gamma']
    texts = [f'query\n"{q_prompt}"', f'positive\n"{p_prompt}"', f'negative\n"{n_prompt}"']
    images = [add_text_to_image(img, txt, font_size=30, wrap_text_width=None) for img, txt in zip(images, texts)]
    img = Image.fromarray(np.concatenate(images, axis=1))
    question = qn.anchor_gamma_shared_text if qd_id.which == 'gamma_positive' else qn.anchor_delta_shared_text
    img = add_text_to_image(img, question, vertical_position='bottom', horizontal_position=1/6, alignment='center', font_size=24)
    display(img)


def generate_quintuplets():
    output_dir_path = "/home/dcor/roeyron/datasets/quintuplets_v1"
    os.mkdir(output_dir_path)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to("cuda")

    for q_ind, q in tqdm(enumerate(QUINUPLETS_RAW), desc="Generating Quintuplets"):
        image_anchor = pipe(q["anchor"], target_size=(512, 512)).images[0]
        image_gamma = pipe(q["gamma"], target_size=(512, 512)).images[0]
        image_delta = pipe(q["delta"], target_size=(512, 512)).images[0]

        dp_id = f"{q_ind:03d}"
        Quintuplet(
            image_anchor,
            image_gamma,
            image_delta,
            q["anchor-gamma"],
            q["anchor-delta"],
            q,
        ).save(output_dir_path, dp_id)


if __name__ == "__main__":
    generate_quintuplets()
