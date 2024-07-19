from PIL import Image
import numpy as np
from collections import defaultdict
import os
import pandas as pd
from tqdm.auto import tqdm
import pickle
import argparse

from hidden_state_utils import get_reduced_hidden_states_to_store, get_hidden_states
from nlp_utils import load_model
from quintuplets import get_splits_ids, QuadrupletId, QUINTUPLETS_DATASET_PATH, load_quadruplet


CHUNKS_SIZE = 5


def get_list_subset(lst, n_parts, part_ind):
    part_size = len(lst) // n_parts
    remainder = len(lst) % n_parts
    start_ind = part_ind * part_size + min(part_ind, remainder)
    end_ind = start_ind + part_size + (1 if part_ind < remainder else 0)
    return lst[start_ind:end_ind]


def extract_embeddings(n_parts, part, output_dir_path):
    processor, model = load_model()
    qn_ids = get_splits_ids()["train"]
    qd_ids = [QuadrupletId(qn_id, which) for qn_id in qn_ids for which in ["gamma_positive", "delta_positive"]]
    qd_ids = get_list_subset(qd_ids, n_parts, part)

    ddl_inference = defaultdict(list)
    chunk_id = 0
    for i, qd_id in tqdm(enumerate(qd_ids), desc="Extract hidden states", total=len(qd_ids)):
        qd = load_quadruplet(QUINTUPLETS_DATASET_PATH, qd_id)
        question = qd.prompt
        for image_type in ["query", "positive", "negative"]:
            image = getattr(qd, image_type)
            hidden_states, input_ids, output_ids = get_hidden_states(image, question, model, processor)

            # hidden states
            hidden_states = get_reduced_hidden_states_to_store(hidden_states, len(input_ids))
            ddl_inference["hidden_states"].append(hidden_states)
            ddl_inference["input_ids"].append(input_ids)
            ddl_inference["output_ids"].append(output_ids)

            # QuadrupletId
            ddl_inference["quintuplet_id"].append(qd_id.quintuplet_id)
            ddl_inference["which"].append(qd_id.which)

            # is it the query, positive or negative image
            ddl_inference["image_type"].append(image_type)

        if i > 0 and i % CHUNKS_SIZE == 0 or i == len(qd_ids) - 1:
            df_inference = pd.DataFrame(ddl_inference)
            out_file_path = os.path.join(output_dir_path, f"part_{n_parts:02d}_chunk_{chunk_id:02d}.pickle")
            with open(out_file_path, "wb") as f:
                pickle.dump(df_inference, f)
            print(f"saved: {out_file_path}")
            ddl_inference = defaultdict(list)
            chunk_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_path", type=str, help="Output directory path")
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
    extract_embeddings(n_parts, part, output_dir_path)


if __name__ == "__main__":
    main()


"""
cd patchscopes/code/nlp_project
export PYTHONPATH=$PYTHONPATH:$(pwd)
python nlp_project/text_conditioned_image_embedding.py
"""
