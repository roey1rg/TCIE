from genecis_evaluation.coco_dataset import COCOValSubset, FOCUS_OBJECT
from PIL import Image

from hidden_state_utils import get_hidden_states
from nlp_utils import load_model


def evaluate_index(
    reference: Image.Image, caption: str, gallery_and_target: list[Image.Image], target_index: int
) -> bool:
    hidden_states, input_ids, output_ids = get_hidden_states(reference, caption, model, processor)
    print(hidden_states)


def main():
    for i in range(min(len(dataset), 2)):
        reference, caption, gallery_and_target, target_index = dataset[i]
        evaluate_index(reference, caption, gallery_and_target, target_index)


if __name__ == "__main__":
    dataset = COCOValSubset(FOCUS_OBJECT)
    print("Loading model")
    processor, model = load_model()
    print("Successfully loaded the model")
    main()
