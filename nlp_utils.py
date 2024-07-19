from collections import defaultdict
import os
import pandas as pd
import getpass
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
import matplotlib as mpl
import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_animals_data_df(data_path=None, limit_per_class=10):
    data_path = (
        data_path or "/home/dcor/roeyron/interpretability_multi_hop/patchscopes/code/nlp_project/animals10/raw-img/"
    )
    translate = {
        "cane": "dog",
        "cavallo": "horse",
        "elefante": "elephant",
        "farfalla": "butterfly",
        "gallina": "hen",
        "gatto": "cat",
        "mucca": "cow",
        "pecora": "sheep",
        "ragno": "spider",
        "scoiattolo": "squirrel",
    }
    ddl = defaultdict(list)
    for class_dir_name in sorted(os.listdir(data_path)):
        class_dir_path = os.path.join(data_path, class_dir_name)
        img_names = sorted(os.listdir(class_dir_path))[:limit_per_class]
        for img_name in img_names:
            img_path = os.path.join(class_dir_path, img_name)
            ddl["class_name"].append(translate[class_dir_name])
            ddl["image_path"].append(img_path)
    return pd.DataFrame(ddl)


def get_celeba_data_df(data_path=None, limit=200):
    dataset_path = data_path or "/home/dcor/roeyron/interpretability_multi_hop/patchscopes/code/celeba-dataset"
    eval_path = os.path.join(dataset_path, "list_attr_celeba.csv")
    df = pd.read_csv(eval_path)
    if limit:
        df = df.sample(limit, random_state=42)
    df["image_path"] = df.image_id.apply(
        lambda image_id: os.path.join(dataset_path, "img_align_celeba/img_align_celeba/", image_id)
    )
    df = df[list(df.columns[-1:]) + list(df.columns[:-1])]
    df = df.applymap(lambda x: {-1: False, 1: True}.get(x, x))
    return df


def get_project_root():
    return os.path.abspath(os.path.dirname(__file__))


def get_cache_dir() -> str:
    user = getpass.getuser()
    if user == "roeyron":
        return "/home/dcor/roeyron/.cache/"
    return f"/home/joberant/NLP_2324/{user}/.cache/"


def wrap(s, max_width):
    return "\n".join(textwrap.wrap(s, max_width))


def add_text_to_image(
    image,
    text,
    font_size=30,
    box_alpha=165,
    vertical_position="top",
    horizontal_position=0,
    alignment="left",
    wrap_text_width=40,
):
    if wrap_text_width is not None:
        text = wrap(text, wrap_text_width)
    # Ensure the image has an alpha channel
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    draw = ImageDraw.Draw(image)
    # Load a font
    font = ImageFont.truetype("arial.ttf", font_size)
    # Calculate text size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Set margins
    margin_x = 10
    margin_y = 10

    # Calculate vertical position
    if vertical_position.lower() == "bottom":
        text_y = image.height - text_height - margin_y
    else:  # Default to top
        text_y = margin_y

    # Calculate horizontal position
    image_width = image.width
    text_x = int(horizontal_position * image_width)

    if alignment.lower() == "center":
        text_x -= text_width // 2
    elif alignment.lower() == "right":
        text_x -= text_width
    # 'left' alignment doesn't need adjustment

    # Ensure text stays within image bounds
    text_x = max(margin_x, min(text_x, image_width - text_width - margin_x))

    # Create a new image for the semi-transparent background
    background = Image.new("RGBA", image.size, (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(background)

    # Draw semi-transparent white background rectangle for the text
    bg_draw.rectangle(
        [text_x, text_y, text_x + text_width, text_y + text_height * 1.4],
        fill=(255, 255, 255, box_alpha),
    )

    # Composite the background onto the original image
    image = Image.alpha_composite(image, background)

    # Add text to image
    draw = ImageDraw.Draw(image)
    draw.text((text_x, text_y), text, fill="black", font=font)

    return image


def get_cos_sim(a, b):
    a = a.astype("float32")
    b = b.astype("float32")
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert np.isnan(cos_sim).sum() == 0
    return cos_sim


def set_default_figure_params():
    params = {
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": False,
        "figure.figsize": [5.5, 5.5],
        "figure.dpi": 200,
    }
    mpl.rcParams.update(params)


def load_model(device=DEVICE):
    cache_dir = get_cache_dir()
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, cache_dir=cache_dir
    )
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model = model.to(device)
    return processor, model
