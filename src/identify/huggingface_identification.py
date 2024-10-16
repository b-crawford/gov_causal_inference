from tqdm import tqdm
from transformers import pipeline

tqdm.pandas()

DEFAULT_HUGGINGFACE_MODEL = "facebook/bart-large-mnli"
ZERO_SHOT_CATEGORIES = {
    "possibly": "A piece of research potentially involving causal inference. i.e. a piece of research attempting to quantitatively identify the factors contributing to some phenomenon.",
    "unlikely": "Non-causal research.",
    "unclear": "Not enough information to describe the research.",
}
reverse_category_map = {value: key for key, value in ZERO_SHOT_CATEGORIES.items()}


def huggingface_inference(descriptions_df, model, causal_cutoff):

    print(f"Running HuggingFace identification with model: {model}")
    classifier = pipeline("zero-shot-classification", model=model, device="mps")

    descriptions_df["huggingface_output"] = descriptions_df[
        "scraped_text"
    ].progress_apply(lambda x: classifier(x, list(ZERO_SHOT_CATEGORIES.values())))

    if causal_cutoff is not None:

        descriptions_df["huggingface_prob_causal"] = descriptions_df[
            "huggingface_output"
        ].apply(extract_possibly_probs)

        descriptions_df["predicted_label"] = descriptions_df[
            "huggingface_prob_causal"
        ].apply(lambda x: "possibly" if x > causal_cutoff else "unlikely")

    else:

        descriptions_df["predicted_label"] = descriptions_df[
            "huggingface_output"
        ].apply(result_to_max_index)

    return descriptions_df


def extract_possibly_probs(result):
    return result["scores"][0]


def result_to_max_index(result):
    # Find the label with the highest probability (score)
    max_score_index = result["scores"].index(max(result["scores"]))
    largest_label = result["labels"][max_score_index]
    return reverse_category_map[largest_label]
