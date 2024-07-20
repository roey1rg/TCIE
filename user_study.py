import json
import pickle
import clip
import re
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
        print(f"Hidden states Appended {results[-1]}")
    return results


def get_clip_image_text_similarity(image, text):
    image_processed = clip_preprocess(image).unsqueeze(0).to(nlp_utils.DEVICE)
    text_tokenized = clip.tokenize([text]).to(nlp_utils.DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_processed)
        text_features = clip_model.encode_text(text_tokenized)
    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features[0], text_features[0]).item()
    sim = (sim + 1) / 2
    return sim


def get_clip_indices(df: pd.DataFrame, descriptions: dict, sampled_indices: list[int]) -> list[list[int]]:
    print("Calculating CLIP results")

    results = []
    for sampled_index in tqdm(sampled_indices):
        desc = descriptions[df.loc[sampled_index]["image_id"]]
        scores = []
        for dp_ind, dp_row in df.iterrows():
            image = Image.open(dp_row["image_path"])
            scores.append(get_clip_image_text_similarity(image, desc))
        results.append([sampled_index] + list(np.argsort(scores)[-K:]))
        print(f"original scores : {list(np.argsort(scores)[-K:])}")
        print(f"CLIP Appended {results[-1]}")
    return results


def get_question_results(question_index: int):
    df = load_hidden_question_df(question_index)
    df.reset_index(drop=True, inplace=True)
    print("Df indices are: ", df.index)
    print("Df len is: ", len(df))
    question = re.sub("[^0-9a-zA-Z]+", "_", df.iloc[0]["question"])
    question_dir = f"{IMAGES_RESULTS_PATH}/{question}"

    print(f"Loaded DF for {question}")
    with open(f"{question_dir}/descriptions.json", "r") as f:
        descriptions = json.load(f)
    sampled_indices = df[df["image_id"].isin(descriptions.keys())].index
    print("sampled_indices", sampled_indices)
    sampled_nearest_indices = get_hidden_nearest_neighbors(df, sampled_indices)
    sampled_clip_indices = get_clip_indices(df, descriptions, sampled_indices)
    for i, (nearest_indices, clip_indices) in enumerate(zip(sampled_nearest_indices, sampled_clip_indices)):
        hidden_images = [Image.open(df["image_path"].loc[index]) for index in nearest_indices]
        hidden_result = np.concatenate([img.resize((180, 180)) for img in hidden_images], axis=1)
        clip_images = [Image.open(df["image_path"].loc[index]) for index in clip_indices]
        clip_result = np.concatenate([img.resize((180, 180)) for img in clip_images], axis=1)
        result_image = Image.fromarray(np.concatenate([hidden_result, clip_result], axis=0))
        if not os.path.isdir(question_dir):
            os.makedirs(question_dir)
        result_image.save(f"{question_dir}/result_{i}.jpg")


if __name__ == "__main__":
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=nlp_utils.DEVICE)

    for i in [0, 1, 2, 3, 4, 6]:
        get_question_results(i)
