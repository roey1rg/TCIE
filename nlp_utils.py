from collections import defaultdict
import os
import pandas as pd
import getpass


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
