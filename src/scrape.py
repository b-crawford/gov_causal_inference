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
    """Loops through each link and gathers the 'details' section content."""
    details_list = []

    for link in tqdm(links):
        response = requests.get(link)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the "details" section
        details = extract_details_section(soup)
        details_list += [
            {"url": link, "details": details, "details_scraped_at": datetime.now()}
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
    return "No details found"


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
            "details": [],
            "details_scraped_at": datetime.now(),
        }
    )


def get_previously_scraped(folder_path):
    try:
        path = os.path.join(folder_path, "basic_descriptions.csv")
        return pd.read_csv(path)
    except:
        return empty_details_df()


def main(start_date, run_id, rescrape):

    # get the folder for this run
    folder_path = utils.get_run_folder_path(run_id)

    # get the initial url to start scraping from
    start_url = build_start_url(start_date)

    # Get all the urls
    links = get_all_urls(start_url)

    # Output the result
    print(f"Numebr of links found in time window: {len(links)}")

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

    # Concat
    to_save = pd.concat([previously_scraped, pd.DataFrame(new_details)], axis=0)

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
