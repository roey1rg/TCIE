from collections import defaultdict
import os
import pandas as pd
import getpass
from PIL import Image, ImageDraw, ImageFont


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


def get_cache_dir() -> str:
    user = getpass.getuser()
    if user == "roeyron":
        return "/home/dcor/roeyron/.cache/"
    return f"/home/joberant/NLP_2324/{user}/.cache/"


def add_text_to_image(image, text, font_size=30):
    draw = ImageDraw.Draw(image)
    # Load a font
    font = ImageFont.truetype("arial.ttf", font_size)
    # Calculate text size and position for upper-left corner
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = 10  # Margin from the left edge
    text_y = 10  # Margin from the top edge

    # Draw white background rectangle for the text
    draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height * 1.4], fill=(255, 255, 255, 10))

    # Add text to image
    draw.text((text_x, text_y), text, fill='black', font=font)

    return image
