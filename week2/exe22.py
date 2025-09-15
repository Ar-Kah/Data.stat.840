import numpy as np
import nltk
from pathlib import Path
import re
import bs4
import requests



def tagtowordnet(postag):
    wordnettag=-1
    if postag[0]=='N':
        wordnettag='n'
    elif postag[0]=='V':
        wordnettag='v'
    elif postag[0]=='J':
        wordnettag='a'
    elif postag[0]=='R':
        wordnettag='r'
    return(wordnettag)


def getbooktext(parsedpage):
    book_split = str(parsedpage).split("\n")

    book_title = ""
    start_index = 0
    end_index = len(book_split)

    for number, line in enumerate(book_split):
        if re.search("START OF THE PROJECT GUTENBERG EBOOK", line):
            start_index = number + 1  # start after this line
        elif re.search("END OF THE PROJECT GUTENBERG EBOOK", line):
            end_index = number       # end before this line
            break
        elif re.search("Title:", line):
            split_line = line.split()
            split_line = split_line[1:len(split_line)]
            book_title = ''.join(split_line)

    book_contents = book_split[start_index:end_index]

    # Write the book contents into a .txt file in book
    with open(f"books/{book_title}.txt", "w") as file:
        print(f"Writing book contents into file: book/{book_title}.txt")
        print("")
        file.write(''.join(book_contents))
        file.close()


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
    elements = webpage_parsed.find_all('a')

    for element in elements:
        url = str(element.get('href'))

        if re.search(".txt.utf-8", url):
            txt_file_url = url

    # ebook href is in the second index of the 'td' elements
    prefix = "https://www.gutenberg.org"
    return prefix + txt_file_url

def getbookurls(parsedpage, N=10):

    urls = []
    # Get all of the list items
    all_bookelements = parsedpage.find_all('ol')

    # The index of the Top 100 ebooks in the last 30 days is 4
    top_last_30_days = all_bookelements[4]
    days30_bookelements = top_last_30_days.find_all('a')
    for index, element in enumerate(days30_bookelements):

        if index >= N:
            break

        prefix = "https://www.gutenberg.org"

        url = str(element.get('href'))
        urls.append(prefix + url)


    return urls


def cleanuptext(text):
    split_text = text.split()
    clean_text_array = []
    for word in split_text:
        stripped_text = word.strip("[]_ *")
        stripped_text = stripped_text.lower()
        clean_text_array.append(stripped_text)

    clean_text = " ".join(clean_text_array)
    return clean_text

def stemtext(nltktexttostem):
    stemmer = nltk.stem.porter.PorterStemmer()
    stemmedtext=[]
    for text in nltktexttostem:
        # Stem the word
        stemmedword=stemmer.stem(text)
        # Store the stemmed word
        stemmedtext.append(stemmedword)
    return(stemmedtext)

def lematizetext(nltktexttolemmatize):
    lemmatizer=nltk.stem.WordNetLemmatizer()
    # Tag the text with POS tags
    taggedtext=nltk.pos_tag(nltktexttolemmatize)
    # Lemmatize each word text
    lemmatizedtext=[]
    for l in range(len(taggedtext)):
        # Lemmatize a word using the WordNet converted POS tag
        wordtolemmatize=taggedtext[l][0]
        wordnettag=tagtowordnet(taggedtext[l][1])
        if wordnettag!=-1:
            lemmatizedword=lemmatizer.lemmatize(wordtolemmatize,wordnettag)
        else:
            lemmatizedword=wordtolemmatize
        # Store the lemmatized word
        lemmatizedtext.append(lemmatizedword)
    return(lemmatizedtext)

def wordprssessing():

    directory = Path("/home/omakone3/Programming/Data.stat.840/week2/books/")


    crawled_nltktexts = []
    for number, txt_file in enumerate(directory.glob("*.txt")):
        if number == 1:
            break
        # Open .txt file to start word prossessing
        print(f"Opening file: {txt_file.name}")
        with open(txt_file, 'r') as file:
            text = file.read()
            clean_text = cleanuptext(text)
            tokenized_words = nltk.word_tokenize(clean_text)
            nltk_text = nltk.Text(tokenized_words)
            crawled_nltktexts.append(nltk_text)

            file.close()

    return crawled_nltktexts

def main():
    """
    This is the exercise 2.2 from the course Data.stat.840.
    The goal of this exercise is to make a webcrawler that
    is capable of crawling through ebooks on the Project
    Gutenberg web page

    @author: Aaro Karhu
    @omakone3@archlinux
    """

    nltk.download('wordnet')
    nltk.download('punkt')
    # save the url of Project Gutenbergs top ebooks
    # webpage_url = 'https://www.gutenberg.org/browse/scores/top'
    # webpage_html = requests.get(webpage_url)

    # # parsed document with bs4
    # webpage_parsed = bs4.BeautifulSoup(webpage_html.content, 'html.parser')
    # urls = getbookurls(webpage_parsed, 20)

    # for url in urls:
    #     booktext_url = getbookcontents(url)
    #     print(f"Getting page: {booktext_url}")
    #     bookpage_html = requests.get(booktext_url)
    #     bookpage_parsed = bs4.BeautifulSoup(bookpage_html.content, 'html.parser')
    #     # Save the book contents into txt files
    #     getbooktext(bookpage_parsed)

    # Here whe already have saved the top N books into local memory to save some RAM.
    # By calling the word prossessing we go through the book contents and prosses the
    # text so that we can do stemming and lemitization later. The returned array will
    # be in nltk Text format.
    nltk.download('averaged_perceptron_tagger_eng')
    nltktexts = wordprssessing()

    crawled_lematizedtexts = []
    for nltktext in nltktexts:
        lematizedtext = lematizetext(nltktext)
        lematizedtext = nltk.Text(lematizedtext)
        crawled_lematizedtexts.append(lematizedtext)

    vocabularies = []
    indicies_in_vocabularies = []

    for lematized in crawled_lematizedtexts:
        uniqueresults = np.unique(lematized, return_inverse=True)
        uniquewords = uniqueresults[0]
        wordindices=uniqueresults[1]
        vocabularies.append(uniquewords)
        indicies_in_vocabularies.append(wordindices)

    # Unify vocabulary into big vocab
    tempvocabulary=[]
    for k in range(len(crawled_lematizedtexts)):
        tempvocabulary.extend(vocabularies[k])
        # Find the unique elements among all vocabularies
    uniqueresults=np.unique(tempvocabulary, return_inverse=True)
    del(tempvocabulary)
    unifiedvocabulary=uniqueresults[0]
    wordindices=uniqueresults[1]

    # Translate previous indices to the unified vocabulary.
    # Must keep track where each vocabulary started in
    # the concatenated one.
    vocabularystart=0
    indices_in_unifiedvocabulary=[]
    for k in range(len(crawled_lematizedtexts)):
        # In order to shift word indices, we must temporarily
        # change their data type to a Numpy array
        tempindices=np.array(indicies_in_vocabularies[k])
        tempindices=tempindices+vocabularystart
        tempindices=wordindices[tempindices]
        indices_in_unifiedvocabulary.append(tempindices)
        vocabularystart=vocabularystart+len(vocabularies[k])

    print(unifiedvocabulary[1000:1020])

if __name__ == '__main__':
    main()
