import torch
from PIL import Image
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import pickle
from nlp_utils import load_model, get_animals_data_df
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUESTIONS = ["describe the animal in the image", "describe the background of the image"]


def get_n_layers(model):
    all_module_names = [name for name, _ in model.named_modules()]
    lang_model_related_layers = [module_name for module_name in all_module_names if "language_model.model.layers." in module_name]
    layers = list(set([e.split("language_model.model.layers.")[1].split(".")[0] for e in lang_model_related_layers]))
    n_layers = max([int(e) for e in layers])
    return n_layers


def extract_hidden_states(processor, model, df_data, question):
    ddl_inference = defaultdict(list)
    for _, row in tqdm(df_data.iterrows(), total=len(df_data)):
        image = Image.open(row.image_path)
        prompt = f"[INST] <image>\n{question} [/INST]"
        inputs = processor(prompt, image, return_tensors="pt").to(DEVICE)
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        hidden_states = output["hidden_states"]
        hidden_states = tuple(
            tuple(t.detach().cpu().numpy() for t in inner_tuple)
            for inner_tuple in hidden_states
        )
        input_ids = list(inputs["input_ids"][0].detach().cpu().numpy())
        output_ids = list(output["sequences"][0].detach().cpu().numpy())
        ddl_inference["hidden_states"].append(hidden_states)
        ddl_inference["input_ids"].append(input_ids)
        ddl_inference["output_ids"].append(output_ids)
    return ddl_inference, input_ids


def train_classifiers_and_get_accuracies(ddl_inference, y, token_locs, layers):
    ddl = defaultdict(list)
    for token_loc, layer in tqdm(list(itertools.product(token_locs, layers))):
        X = np.array([hs[0][layer][0][token_loc] for hs in ddl_inference["hidden_states"]])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classifier = MLPClassifier(max_iter=200)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        ddl["token_loc"].append(token_loc)
        ddl["layer"].append(layer)
        ddl["accuracy"].append(accuracy)

    df_acc = pd.DataFrame(ddl)
    return df_acc


def main(limit_per_class=50):
    processor, model = load_model()
    df_data = get_animals_data_df(limit_per_class=limit_per_class)
    results_dir_path = "/home/dcor/roeyron/TCIE/results/classification_experiment_fix_layers"
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)

    for question in QUESTIONS:
        ddl_inference, input_ids = extract_hidden_states(processor, model, df_data, question)
        class_names = sorted(list(df_data.class_name.unique()))
        labels = df_data.class_name.apply(lambda name: class_names.index(name)).to_list()
        token_locs = list(range(-len(input_ids), 0))
        layers = list(range(1, 33))
        df_acc = train_classifiers_and_get_accuracies(ddl_inference, labels, token_locs, layers)

        token_locs = df_acc.token_loc.unique()
        token_ids = np.array(input_ids)[token_locs]
        tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)

        data = {"df_acc": df_acc, "tokens": tokens, "class_names": class_names}

        q_str = question.replace(" ", "_")
        pickle_path = os.path.join(results_dir_path, f"{q_str}_training_results.pickle")
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)


def create_graphs(results_dir_path, question):
    q_str = question.replace(" ", "_")
    pickle_path = os.path.join(results_dir_path, f"{q_str}_training_results.pickle")
    # Reloading the data from the pickle file
    with open(pickle_path, "rb") as f:
        loaded_data = pickle.load(f)

    df_acc = loaded_data["df_acc"]
    tokens = loaded_data["tokens"]
    class_names = loaded_data["class_names"]

    heatmap_path = os.path.join(results_dir_path, f"{q_str}_heatmap.png")
    graph_path = os.path.join(results_dir_path, f"{q_str}_graph.png")
    plot_heatmap(df_acc, tokens, heatmap_path)
    plot_graph(class_names, df_acc, graph_path)


def plot_graph(class_names, df_acc, token_loc=-1, output_path=None):
    x = df_acc[df_acc.token_loc == token_loc].layer
    y = df_acc[df_acc.token_loc == token_loc].accuracy
    plt.figure(figsize=(8, 5))
    plt.title("Classifier Accuracy v.s. Hidden Layer Depth")
    plt.grid(True)
    plt.ylabel("Accuracy")
    plt.xlabel("layer")
    plt.plot(x, y, label="trained classifier accuracy")
    plt.axhline(1 / len(class_names), label="random guess", c="red")
    plt.legend(loc="upper right")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_heatmap(df_acc, tokens, output_path=None):
    plt.figure(figsize=(25, 8))
    heatmap_data = df_acc.pivot(columns=["token_loc"], index="layer", values="accuracy")
    ax = sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
    ax.invert_yaxis()
    plt.title("Accuracy Heatmap")
    plt.xticks(np.array(range(len(tokens))) + 0.5, tokens)
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
