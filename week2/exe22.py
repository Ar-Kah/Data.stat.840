import re
import bs4
import requests

def getpagetext(parsedpage):
    # Remove HTML elements that are scripts
    scriptelements=parsedpage.find_all('script')
    # Concatenate the text content from all table cells
    for scriptelement in scriptelements:
        # Extract this script element from the page.
        # This changes the page given to this function!
        scriptelement.extract()
    pagetext=parsedpage.get_text()
    return (pagetext)


def getbookcontents(url):

    webpage_html = requests.get(url)
    webpage_parsed = bs4.BeautifulSoup(webpage_html.content, 'html.parser')

    text = getpagetext(webpage_parsed)
    print(text)

def getpageurls(parsedpage, N=10):
    """
    @parsedpage: Parsed html documents using bs4
    @N: amount of book to bee crawled
    """
    urls = []
    # Get all of the list items
    all_bookelements = parsedpage.find_all('ol')

    # The index of the Top 100 ebooks in the last 30 days is 4
    top_last_30_days = all_bookelements[4]
    days30_bookelements = top_last_30_days.find_all('a')
    for index, element in enumerate(days30_bookelements):

        if index >= N:
            break

        posfix = "https://www.gutenberg.org"

        url = str(element.get('href'))
        urls.append(posfix + url)

    return urls

def main():
    """
    This is the exercise 2.2 from the course Data.stat.840.
    The goal of this exercise is to make a webcrawler that
    is capable of crawling through ebooks on the Project
    Gutenberg web page

    @author: Aaro Karhu
    @omakone3@archlinux
    """

    # save the url of Project Gutenbergs top ebooks
    webpage_url = 'https://www.gutenberg.org/browse/scores/top'
    webpage_html = requests.get(webpage_url)

    # parsed document with bs4
    webpage_parsed = bs4.BeautifulSoup(webpage_html.content, 'html.parser')
    urls = getpageurls(webpage_parsed)

    getbookcontents(urls[0])


if __name__ == '__main__':
    main()
