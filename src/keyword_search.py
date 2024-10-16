import spacy
from spacy.cli import download
from tqdm import tqdm

tqdm.pandas()

# set up parameters
POSITIVE_KEYWORDS = [
    "impact",
    "drivers",
    "effect",
    "affect",
    "causal",
    "influence",
    "relation",
    "relationship",
    "contribute",
    "factor",
]
NEGATIVE_KEYWORDS = ["political influence"]


# Function to ensure spaCy model is downloaded
def ensure_model_downloaded(model_name):
    try:
        # Try to load the spaCy model
        nlp = spacy.load(model_name)
        print(f"Loaded model: {model_name}")
        return nlp
    except OSError:
        # If the model is not found, download it
        print(f"Model {model_name} not found. Downloading...")
        download(model_name)
        # Load the model after downloading
        nlp = spacy.load(model_name)
        print(f"Downloaded and loaded model: {model_name}")
        return nlp


def find_keywords(nlp, text, keywords):
    """Search for causal keywords in the text."""
    doc = nlp(text)
    found_keywords = [token.text for token in doc if token.text.lower() in keywords]
    return found_keywords


def extract_keywords(scraped_df, positive_keywords=None, negative_keywords=None):

    # without keywords just return the dataframe
    if positive_keywords is None and negative_keywords is None:
        print("No keywords entered, returning input dataframe")
        return scraped_df

    nlp = ensure_model_downloaded("en_core_web_sm")

    if positive_keywords is not None:
        print("Running positive keyword search.")
        scraped_df["positive_keywords"] = scraped_df["scraped_text"].progress_apply(
            lambda x: find_keywords(nlp, x, positive_keywords)
        )

        scraped_df["contains_positive_keyword"] = scraped_df["positive_keywords"].apply(
            lambda x: len(x) > 0
        )

    if negative_keywords is not None:
        print("Running negative keyword search.")
        scraped_df["negative_keywords"] = scraped_df["scraped_text"].progress_apply(
            lambda x: find_keywords(nlp, x, negative_keywords)
        )

        scraped_df["contains_negative_keyword"] = scraped_df["negative_keywords"].apply(
            lambda x: len(x) > 0
        )

    return scraped_df