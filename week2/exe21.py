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



def getpageurls(webpage_parsed):
    # Find elements that are hyperlinks
    pagelinkelements=webpage_parsed.find_all('a')
    # print(type( pagelinkelements[0].get('href') ))
    pageurls=[]
    for pagelink in pagelinkelements:

        crawled_url = str(pagelink.get('href'))

        if re.search("^https:", crawled_url):
            pageurls.append(crawled_url)


    return(pageurls)



def basicwebcrawler(seedpage_url, maxpages, max_links_from_page=10):
    # Store URLs crawled and their text content
    num_pages_crawled=0
    crawled_urls=[]
    crawled_texts=[]

    # Remaining pages to crawl: start from a seed page URL
    pagestocrawl=[seedpage_url]

    # Process remaining pages until a desired number
    # of pages have been found
    while (num_pages_crawled < maxpages) & (len(pagestocrawl) > 0):
        # Retrieve the topmost remaining page and parse it
        pagetocrawl_url=pagestocrawl[0]

        # check if we have allready crawled the page
        if pagetocrawl_url in crawled_urls:
            pagestocrawl.pop(0)
            continue

        print('Getting page:')
        print(pagetocrawl_url)
        pagetocrawl_html=requests.get(pagetocrawl_url)
        pagetocrawl_parsed=bs4.BeautifulSoup(pagetocrawl_html.content,'html.parser')
        # Get the text and URLs of the page
        pagetocrawl_text=getpagetext(pagetocrawl_parsed)
        pagetocrawl_urls=getpageurls(pagetocrawl_parsed)
        # Store the URL and content of the processed page
        num_pages_crawled=num_pages_crawled+1
        crawled_urls.append(pagetocrawl_url)
        crawled_texts.append(pagetocrawl_text)
        # Remove the processed page from remaining pages,
        # but add the new URLs
        pagestocrawl=pagestocrawl[1:len(pagestocrawl)]
        if len(pagetocrawl_urls) > max_links_from_page:
            pagetocrawl_urls = pagetocrawl_urls[0:max_links_from_page]

        pagestocrawl.extend(pagetocrawl_urls)

    print(f"number of crawled pages: { num_pages_crawled }")
    return(crawled_urls,crawled_texts)

def main():
    #mywebpage_url='https://www.tuni.fi/en/'
    mywebpage_url='https://www.w3schools.com/html/html_links.asp'

    mycrawled_urls_and_texts=basicwebcrawler(mywebpage_url, 30)

if __name__ == '__main__':
    main()
