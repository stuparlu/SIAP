import requests
from bs4 import BeautifulSoup

# create a list of project names
project_names = ["Hard Berries", "I Met with an Accident - Inspired by Benedict Das",
                 "Weirdwood Manor — Unlock Your Imagination",
                 "METHOD - comedy series", "GOBAG - A Vacuum Compressible Carry-On Bag For Any Adventure",
                 "SONS OF SOIL Promo Trailer Shoot!", "CatZapper - Get cat litter off your social media",
                 "Awaken: Volume 1 by Koti Saavedra", "Salvi & Maya children's books: small books, big message.",
                 "BARAGONTA - Luxury Hotel & Resort Villa"]

# loop through each project name and search for its corresponding link on Kickstarter
for project in project_names:
    # make a request to Kickstarter's search results page
    url = f"https://www.kickstarter.com/search?term={project}"
    response = requests.get(url)

    # parse the HTML content of the page using Beautiful Soup
    soup = BeautifulSoup(response.content, "html.parser")
    print(response.content)
    # extract the link of the first project in the search results
    project_link = soup.find("div", class_="search-result").find("a")["href"]

    # print the project name and link
    print(f"{project}: {project_link}")
