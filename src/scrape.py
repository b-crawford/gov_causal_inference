import requests
from bs4 import BeautifulSoup
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import argparse
import utils
import os
import hashlib

BASE_URL = "https://www.gov.uk"


def build_start_url(start_date):

    # additional query params
    query_url = (
        "/search/all?content_purpose_supergroup%5B%5D=research_and_statistics"
        + f"&public_timestamp%5Bfrom%5D={start_date}"
        + "&order=updated-newest"
    )

    return BASE_URL + query_url


def get_links_from_page(url):
    # Send GET request to the page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all <li> elements with the specific class
    links = []
    for li in soup.find_all("li", class_="gem-c-document-list__item"):
        a_tag = li.find("a", class_="govuk-link")
        if a_tag and a_tag.has_attr("href"):
            # Append the full link (base URL + href)
            full_link = BASE_URL + a_tag["href"]
            links.append(full_link)
            print(f"Found link: {full_link}")

    return links, soup


def get_next_page_url(soup):
    # Find the next page link in pagination
    next_page = soup.find("a", class_="govuk-link govuk-pagination__link", rel="next")
    if next_page:
        next_url = BASE_URL + next_page["href"]
        return next_url
    return None


def get_all_urls(start_url):
    url = start_url
    all_links = []

    while url:
        print(f"Scraping page: {url}")

        # Get links from the current page
        links, soup = get_links_from_page(url)
        all_links.extend(links)

        # Find the next page URL
        url = get_next_page_url(soup)

        # Sleep to avoid being too aggressive with requests
        time.sleep(np.random.uniform(low=0.5, high=1.5))

    return all_links


def scrape_details_for_links(links):
    """Loops through each link and gather important information"""
    details_list = []

    for link in tqdm(links):
        response = requests.get(link)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract relevant info
        title = extract_title(soup)
        lead_paragraph = extract_lead_paragraph(soup)
        context = extract_context(soup)
        details = extract_details_section(soup)
        publishing_institution = extract_publishing_institution(soup)

        details_list += [
            {
                "url": link,
                "title": title,
                "description": lead_paragraph,
                "context": context,
                "publishing_institution": publishing_institution,
                "details": details,
                "info_scraped_at": datetime.now(),
            }
        ]
        # Sleep between requests to avoid overwhelming the server
        time.sleep(np.random.uniform(low=0.5, high=1.5))

    return details_list


def extract_details_section(soup):
    """Extracts the 'details' section content from a given soup object."""
    details_section = soup.find("section", id="details")
    if details_section:
        details_text = details_section.get_text(separator="\n", strip=True)
        return details_text
    return pd.NA


def extract_context(soup):
    """Extracts the text content of a specific element based on the class name."""
    # Find the <span> element with the specific class
    element = soup.find("span", class_="govuk-caption-xl gem-c-title__context")

    # If the element exists, extract and return its text
    if element:
        element_text = element.get_text(strip=True)
        return element_text
    return pd.NA


def extract_title(soup):
    """Extracts the text content of the <h1> element with the specified class."""
    # Find the <h1> element with the specific class
    h1_element = soup.find("h1", class_="gem-c-title__text govuk-heading-l")

    # If the element exists, extract and return its text
    if h1_element:
        h1_text = h1_element.get_text(strip=True)
        return h1_text
    return pd.NA


def extract_lead_paragraph(soup):
    """Extracts the text content of the <p> element with the specified class."""
    # Find the <p> element with the specific class
    p_element = soup.find("p", class_="gem-c-lead-paragraph")

    # If the element exists, extract and return its text
    if p_element:
        p_text = p_element.get_text(strip=True)
        return p_text
    return pd.NA


def extract_publishing_institution(soup):
    """Extracts the 'From:' institution from a given soup object."""
    metadata_list = soup.find("dl", class_="gem-c-metadata__list")

    if metadata_list:
        # Find all 'dt' elements (terms) and iterate through them
        terms = metadata_list.find_all("dt", class_="gem-c-metadata__term")

        for term in terms:
            if term.get_text(strip=True) == "From:":
                # The corresponding 'dd' element should contain the institution
                institution_tag = term.find_next_sibling("dd")

                if institution_tag:
                    # Extract the institution name, potentially inside an <a> tag
                    institution = institution_tag.get_text(strip=True)
                    return institution

    return pd.NA


def validate_date(date_str):
    """Validate the date format and value."""
    try:
        # Try to create a datetime object from the input string
        valid_date = datetime.strptime(date_str, "%d/%m/%Y")
        return valid_date
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{date_str}'. Use DD/MM/YYYY format."
        )


def save_to_csv(details_df, start_date, folder_path):

    # path to data file
    path = os.path.join(folder_path, "basic_descriptions.csv")
    print(f"Saving basic descriptions to: {path}")

    # save
    details_df.to_csv(path, index=False)

    # save meta data
    metadata = pd.DataFrame(
        {
            "start_date": [start_date],
            "run_datetime": [datetime.now()],
            "n_urls": [len(details_df)],
        }
    )
    metadata_path = os.path.join(folder_path, "scrape_metadata.csv")
    metadata.to_csv(metadata_path, index=False)


def empty_details_df():
    return pd.DataFrame(
        {
            "url": [],
        }
    )


def get_previously_scraped(folder_path):
    try:
        path = os.path.join(folder_path, "basic_descriptions.csv")
        return pd.read_csv(path)
    except:
        return empty_details_df()


def combine_text(title, description, details, context):
    return f"{title} ({context}) \n{description}\n{details}"


def combine_text_df(descriptions_df):
    descriptions_df["scraped_text"] = descriptions_df.apply(
        lambda row: combine_text(
            row["title"], row["description"], row["details"], row["context"]
        ),
        axis=1,
    )

    return descriptions_df


def hash_string(string_):
    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()

    # Encode the string and update the hash object
    hash_object.update(string_.encode("utf-8"))

    # Get the hexadecimal representation of the hash
    return hash_object.hexdigest()


def main(start_date, run_id, rescrape):

    # get the folder for this run
    folder_path = utils.get_run_folder_path(run_id)

    # get the initial url to start scraping from
    start_url = build_start_url(start_date)

    # Get all the urls
    links = get_all_urls(start_url)

    # Output the result
    print(f"Number of links found in time window: {len(links)}")

    # If we are rescraping then set previously scraped to None
    # otherwise find the previously scraped urls from file
    if rescrape:
        print("Rescraping all links.")
        previously_scraped = empty_details_df()
    else:
        previously_scraped = get_previously_scraped(folder_path)
        links = [i for i in links if i not in list(previously_scraped["url"])]
        print(f"Of these, {len(links)} have not previously been scraped.")

    # Scrape remaining
    new_details = scrape_details_for_links(links)

    # collate all text and hash it
    new_details = combine_text_df(pd.DataFrame(new_details))
    new_details["project_hash"] = new_details["scraped_text"].apply(hash_string)

    # Concat
    to_save = pd.concat([previously_scraped, new_details], axis=0)

    # Save to file
    save_to_csv(to_save, start_date, folder_path)


if __name__ == "__main__":

    # get the date from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--run_id",
        type=int,
        help="The run ID for the data over which we should process.",
        default=None,
    )
    parser.add_argument(
        "-d",
        "--date",
        type=validate_date,  # Use the custom validation function
        help="The date to start gathering data from (format: DD/MM/YYYY)",
        default=datetime.today().strftime("%d/%m/%Y"),
    )
    parser.add_argument("--rescrape", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # run main function
    main(args.date.strftime("%d/%m/%Y"), args.run_id, args.rescrape)
