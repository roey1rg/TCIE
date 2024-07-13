
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import numpy as np
from collections import defaultdict
import os
import pandas as pd
from tqdm.auto import tqdm
import pickle
import nlp_utils  # our utils
import sys


DATAFRAME_CHUNK_SIZE = 400


QUESTIONS = [
    'describe the hair of the person in the image',
    'describe the expression of the person in the image',
    'describe the background color of the image',
    'what is the style of the image?',
    'look on the lips of the person in the image',
    'describe the shirt of the person in the image',
    'describe the nose of the person in the image',
]


def get_reduced_hidden_states_to_store(hidden_states, n_input_tokens):
    result = []
    for hidden_states_l in hidden_states[0]:
        hidden_states_l = hidden_states_l[0][-n_input_tokens:].cpu().detach().numpy()  # (n_input_tokens, 4096)
        result.append(hidden_states_l)
    return np.stack(result, axis=0)


def get_list_subset(lst, n_parts, part_ind):
    part_size = len(lst) // n_parts
    remainder = len(lst) % n_parts
    start_ind = part_ind * part_size + min(part_ind, remainder)
    end_ind = start_ind + part_size + (1 if part_ind < remainder else 0)
    return lst[start_ind:end_ind]


def main(n_parts, part, output_dir_path):
    cache_dir = '/home/dcor/roeyron/.cache/'
    device = 'cuda'
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir)
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model = model.to(device)
    df_data = nlp_utils.get_celeba_data_df(limit=1200)

    ddl_inference = defaultdict(list)

    question_indices = get_list_subset(list(range(len(QUESTIONS))), n_parts, part)
    pbar = tqdm(total=len(question_indices) * len(df_data), desc='Extract hidden states')
    for q_ind in question_indices:
        question = QUESTIONS[q_ind]

        chunk_ind = 0
        for dp_ind, (_, dp_row) in enumerate(df_data.iterrows()):
            image = Image.open(dp_row.image_path)
            prompt = f"[INST] <image>\n{question} [/INST]"
            inputs = processor(prompt, image, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=100, output_hidden_states=True, return_dict_in_generate=True)
            hidden_states = output['hidden_states']

            input_ids = list(inputs['input_ids'][0].detach().cpu().numpy())      # [1, 733, 16289
            output_ids = list(output['sequences'][0].detach().cpu().numpy())     # [1, 733, 16289, ...
            ddl_inference['question'].append(question)
            ddl_inference['image_id'].append(dp_row.image_id)
            ddl_inference['image_path'].append(dp_row.image_path)
            ddl_inference['input_ids'].append(input_ids)
            ddl_inference['output_ids'].append(output_ids)
            hidden_states = get_reduced_hidden_states_to_store(hidden_states, len(input_ids))
            ddl_inference['hidden_states'].append(hidden_states)
            pbar.update()

            if dp_ind % DATAFRAME_CHUNK_SIZE == 0:
                df_inference = pd.DataFrame(ddl_inference)
                out_file_path = os.path.join(output_dir_path, f'qind_{q_ind:02d}_chunk_{chunk_ind:02d}.pickle')
                with open(out_file_path, 'wb') as f:
                    pickle.dump(df_inference, f)
                ddl_inference = defaultdict(list)
                chunk_ind += 1


def run():
    output_dir_path = '/home/dcor/roeyron/interpretability_multi_hop/patchscopes/code/nlp_project/inference_cond_embd_1'
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    args = sys.argv
    print(args)
    try:
        n_parts = int(args[1])
        part = int(args[2])
    except:
        n_parts = 1
        part = 0
    print('n_parts', n_parts)
    print('part', part)
    main(n_parts, part, output_dir_path)


if __name__ == "__main__":
    run()


"""
cd patchscopes/code/nlp_project
export PYTHONPATH=$PYTHONPATH:$(pwd)
python nlp_project/text_conditioned_image_embedding.py
"""
