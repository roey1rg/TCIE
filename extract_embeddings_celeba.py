import torch
from PIL import Image
import numpy as np
from collections import defaultdict
import os
import pandas as pd
from tqdm.auto import tqdm
import pickle
import argparse
import nlp_utils  # our utils
from hidden_state_utils import get_reduced_hidden_states_to_store, get_hidden_states
from nlp_utils import load_model

DATAFRAME_CHUNK_SIZE = 400


QUESTIONS = [
    "describe the hair of the person in the image",
    "describe the expression of the person in the image",
    "describe the background color of the image",
    "what is the style of the image?",
    "look on the lips of the person in the image",
    "describe the shirt of the person in the image",
    "describe the nose of the person in the image",
    "describe the eyes of the person in the image",
    "What is the mouth of the person",
    "describe the ears of the person"
]


def get_list_subset(lst, n_parts, part_ind):
    part_size = len(lst) // n_parts
    remainder = len(lst) % n_parts
    start_ind = part_ind * part_size + min(part_ind, remainder)
    end_ind = start_ind + part_size + (1 if part_ind < remainder else 0)
    return lst[start_ind:end_ind]


def main(n_parts, part, output_dir_path):
    processor, model = load_model()
    processor, model = load_model()

    df_data = nlp_utils.get_celeba_data_df(limit=6000)

    question_indices = get_list_subset(list(range(len(QUESTIONS))), n_parts, part)
    pbar = tqdm(total=len(question_indices) * len(df_data), desc="Extract hidden states")
    for q_ind in question_indices:
        question = QUESTIONS[q_ind]

        ddl_inference = defaultdict(list)

        chunk_ind = 0
        for dp_ind, (_, dp_row) in enumerate(df_data.iterrows()):
            image = Image.open(dp_row.image_path)
            hidden_states, input_ids, output_ids = get_hidden_states(image, question, model, processor)

            ddl_inference["question"].append(question)
            ddl_inference["image_id"].append(dp_row.image_id)
            ddl_inference["image_path"].append(dp_row.image_path)
            ddl_inference["input_ids"].append(input_ids)
            ddl_inference["output_ids"].append(output_ids)
            hidden_states = get_reduced_hidden_states_to_store(hidden_states, len(input_ids))
            ddl_inference["hidden_states"].append(hidden_states)
            pbar.update()

            if (dp_ind > 0) and ((dp_ind % DATAFRAME_CHUNK_SIZE == 0) or (dp_ind == len(df_data)-1)):
                df_inference = pd.DataFrame(ddl_inference)
                out_file_path = os.path.join(output_dir_path, f"qind_{q_ind:02d}_chunk_{chunk_ind:02d}.pickle")
                with open(out_file_path, "wb") as f:
                    pickle.dump(df_inference, f)
                print(f"saved: {out_file_path}")
                ddl_inference = defaultdict(list)
                chunk_ind += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_path", type=str, default=None, help="Output directory path")
    parser.add_argument("--n_parts", type=int, default=1, help="Number of parts")
    parser.add_argument("--part", type=int, default=0, help="Part index")
    args = parser.parse_args()

    output_dir_path = args.output_dir_path
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    n_parts = args.n_parts
    part = args.part

    print("n_parts", n_parts)
    print("part", part)
    main(n_parts, part, output_dir_path)


"""
cd TCIE
export PYTHONPATH=$PYTHONPATH:$(pwd)
python extract_embeddings_celeba.py --output_dir /home/dcor/roeyron/TCIE/results/celeba_conditioned_embeddings --n_parts 16 --part 0

"""
