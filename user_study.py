import json
import pickle
import random
import re
import os

import numpy as np
import pandas as pd

from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import nlp_utils
from hidden_state_utils import ROOT_PATH

results_dir = "/home/dcor/roeyron/TCIE/results/celeba_conditioned_embeddings"
IMAGES_RESULTS_PATH = f"{ROOT_PATH}/celeb_a_results"

results_fnames = sorted(os.listdir(results_dir), key=lambda name: int(name.split("_")[1].split(".")[0]))
results_fpaths = [os.path.join(results_dir, fname) for fname in results_fnames]
LAYER_INDEX = 15
TOKEN_INDEX = -1
K = 5
SAMPLES_PER_QUESTION = 10


def load_hidden_question_df(index: int):
    print(f"Loading DF for question {index}")
    q_i_paths = []
    for result_fpath in results_fpaths:
        if f"qind_0{index}_chunk" in result_fpath:
            q_i_paths.append(result_fpath)
    data_frames = []
    for q_i_path in q_i_paths:
        try:
            with open(q_i_path, "rb") as f:
                data_frames.append(pickle.load(f))
        except:
            continue
    return pd.concat(data_frames)


def get_hidden_nearest_neighbors(df: pd.DataFrame, sampled_indices: list[int]) -> list[list[int]]:
    print("Calculating Nearest Neighbors")
    results = []
    hidden_tokens = np.array([hs[LAYER_INDEX][TOKEN_INDEX] for hs in df["hidden_states"]])
    neighbors = NearestNeighbors(n_neighbors=K + 1, metric="cosine")
    neighbors.fit(hidden_tokens)
    for token_index in sampled_indices:
        token = hidden_tokens[token_index]
        distances, indices = neighbors.kneighbors([token])
        distances, indices = distances[0, :], indices[0, :]
        results.append(indices)
    return results


def get_question_results(question_index: int):
    df = load_hidden_question_df(question_index)
    question = re.sub("[^0-9a-zA-Z]+", "_", df.iloc[0]["question"])
    question_dir = f"{IMAGES_RESULTS_PATH}/{question}"

    print(f"Loaded DF for {question}")
    with open(f"{question_dir}/descriptions.json", "e") as f:
        descriptions = json.load(f)
    sampled_indices = df[df["image_id"].isin(descriptions.keys())].index
    sampled_nearest_indices = get_hidden_nearest_neighbors(df, sampled_indices)
    for i, nearest_indices in enumerate(sampled_nearest_indices):
        images = [Image.open(df["image_path"].iloc[index]) for index in nearest_indices]
        hidden_result = np.concatenate([img.resize((180, 180)) for img in images], axis=1)
        clip_result = np.concatenate([img.resize((180, 180)) for img in images], axis=1)
        result_image = Image.fromarray(np.concatenate([hidden_result, clip_result], axis=0))
        if not os.path.isdir(question_dir):
            os.makedirs(question_dir)
        result_image.save(f"{question_dir}/result_{i}.jpg")
    # clip_model, clip_preprocess = clip.load("ViT-B/32", device=nlp_utils.DEVICE)


if __name__ == "__main__":
    for i in tqdm([0, 1, 2, 3, 4, 6]):
        get_question_results(i)
        break
