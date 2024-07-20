import json

import nlp_utils
import re
from tqdm import tqdm
from PIL import Image

from hidden_state_utils import ROOT_PATH

SAMPLES_PER_QUESTION = 10

LIMIT = 1200
results_path = "/home/dcor/roeyron/TCIE/results/baseline_celeba_conditioned_embeddings/"
QUESTIONS = [
    "describe the hair of the person in the image",
    "describe the expression of the person in the image",
    "describe the background color of the image",
    "what is the style of the image?",
    "look on the lips of the person in the image",
    "describe the nose of the person in the image",
]


def get_llava_desc(image, text):
    prompt = f"[INST] <image>\n{text} \n Limit your response to no more than 2 short sentences. [/INST]"
    inputs = processor(prompt, image, return_tensors="pt").to(nlp_utils.DEVICE)
    output = llava_model.generate(**inputs, max_new_tokens=100, output_hidden_states=True, return_dict_in_generate=True)
    result_text = processor.decode(output["sequences"][0], skip_special_tokens=True)
    result_text = result_text.split(prompt.replace("<image>", " "))[1].strip()
    print("The output results are:", result_text)
    return result_text


def get_question_description(question: str):
    question = re.sub("[^0-9a-zA-Z]+", "_", question)
    descriptions = {}
    celeb_a_df.sample()
    for dp_ind, (_, dp_row) in tqdm(enumerate(celeb_a_df.sample(SAMPLES_PER_QUESTION).iterrows())):
        image = Image.open(dp_row["image_path"])
        descriptions[dp_row["image_id"]] = get_llava_desc(image, question)
    with open(f"{ROOT_PATH}/celeb_a_results/{question}/descriptions.json", "w") as f:
        json.dump(descriptions, f)


def main():
    for question in QUESTIONS:
        print(f"Getting descriptions of {question}")
        get_question_description(question)


if __name__ == "__main__":
    print("Loading Llava model")
    processor, llava_model = nlp_utils.load_model(nlp_utils.DEVICE, "/data/.cache")

    print("Loading celeb_a dataset")
    celeb_a_df = nlp_utils.get_celeba_data_df(data_path=f"{ROOT_PATH}/celeb_a_dataset", limit=1200)
    main()
