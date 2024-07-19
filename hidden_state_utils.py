import pathlib

import numpy as np
from PIL import Image

from nlp_utils import DEVICE

ROOT_PATH = str(pathlib.Path(__file__).parent.absolute())


def get_reduced_hidden_states_to_store(hidden_states, n_input_tokens):
    result = []
    for hidden_states_l in hidden_states[0]:
        hidden_states_l = hidden_states_l[0][-n_input_tokens:].cpu().detach().numpy()  # (n_input_tokens, 4096)
        result.append(hidden_states_l)
    return np.stack(result, axis=0)


def get_hidden_states(image: Image.Image, caption: str, model, processor):
    prompt = f"[INST] <image>\n{caption} [/INST]"
    inputs = processor(prompt, image, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=100, output_hidden_states=True, return_dict_in_generate=True)
    hidden_states = output["hidden_states"]
    input_ids = list(inputs["input_ids"][0].detach().cpu().numpy())  # [1, 733, 16289
    output_ids = list(output["sequences"][0].detach().cpu().numpy())  # [1, 733, 16289, ...
    hidden_states = get_reduced_hidden_states_to_store(hidden_states, len(input_ids))
    return hidden_states, input_ids, output_ids
