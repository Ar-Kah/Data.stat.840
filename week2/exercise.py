from tqdm import tqdm
import matplotlib.pyplot
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


def zipf_curve(a, N):
    ranks = np.arange(1, N+1)
    harmonic = np.sum(1.0 / (ranks ** a))
    probs = (1.0 / (ranks ** a)) / harmonic
    return ranks, probs


def wordprssessing():

    directory = Path("/home/omakone3/Programming/Data.stat.840/week2/books/")


    crawled_nltktexts = []
    for number, txt_file in enumerate(directory.glob("*.txt")):
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
    webpage_url = 'https://www.gutenberg.org/browse/scores/top'
    webpage_html = requests.get(webpage_url)

    # parsed document with bs4
    webpage_parsed = bs4.BeautifulSoup(webpage_html.content, 'html.parser')
    urls = getbookurls(webpage_parsed, 20)


    print()
    print("All crawled books will be saved in directory books.")
    while True:
        print("Do you want to download the books? (y / n)")
        answer = input()
        if answer == 'y':
            break
        elif answer == 'n':
            return

    for url in urls:
        booktext_url = getbookcontents(url)
        print(f"Getting page: {booktext_url}")
        bookpage_html = requests.get(booktext_url)
        bookpage_parsed = bs4.BeautifulSoup(bookpage_html.content, 'html.parser')
        # Save the book contents into txt files
        getbooktext(bookpage_parsed)

    # Here whe already have saved the top N books into local memory to save some RAM.
    # By calling the word prossessing we go through the book contents and prosses the
    # text so that we can do stemming and lemitization later. The returned array will
    # be in nltk Text format.
    nltk.download('averaged_perceptron_tagger_eng')
    nltktexts = wordprssessing()
    print()

    crawled_lematizedtexts = []
    print("Lematizing all texts")
    for index, nltktext in enumerate(tqdm(nltktexts)):
        lematizedtext = lematizetext(nltktext)
        lematizedtext = nltk.Text(lematizedtext)
        crawled_lematizedtexts.append(lematizedtext)
    print()
    vocabularies = []
    indicies_in_vocabularies = []

    print("Find Unique words and their indicies")
    for index, lematized in enumerate(tqdm(crawled_lematizedtexts)):
        uniqueresults = np.unique(lematized, return_inverse=True)
        uniquewords = uniqueresults[0]
        wordindices=uniqueresults[1]
        vocabularies.append(uniquewords)
        indicies_in_vocabularies.append(wordindices)

    print()
    # Unify vocabulary into big vocab
    tempvocabulary=[]
    print("Appending vocabulary into big vocab")
    for k in tqdm(range(len(crawled_lematizedtexts))):
        tempvocabulary.extend(vocabularies[k])
        # Find the unique elements among all vocabularies
    print()
    uniqueresults=np.unique(tempvocabulary, return_inverse=True)
    del(tempvocabulary)

    unifiedvocabulary=uniqueresults[0]
    wordindices=uniqueresults[1]

    # Translate previous indices to the unified vocabulary.
    # Must keep track where each vocabulary started in
    # the concatenated one.
    vocabularystart=0
    indices_in_unifiedvocabulary=[]
    print("")
    for k in tqdm(range(len(crawled_lematizedtexts))):
        # In order to shift word indices, we must temporarily
        # change their data type to a Numpy array
        tempindices=np.array(indicies_in_vocabularies[k])
        tempindices=tempindices+vocabularystart
        tempindices=wordindices[tempindices]
        indices_in_unifiedvocabulary.append(tempindices)
        vocabularystart=vocabularystart+len(vocabularies[k])


    unifiedvocabulary_totaloccurrencecounts=np.zeros((len(unifiedvocabulary),1))
    unifiedvocabulary_documentcounts=np.zeros((len(unifiedvocabulary),1))
    unifiedvocabulary_meancounts=np.zeros((len(unifiedvocabulary),1))
    unifiedvocabulary_countvariances=np.zeros((len(unifiedvocabulary),1))


    # First pass: count occurrences
    for k in range(len(crawled_lematizedtexts)):
        occurrencecounts=np.zeros((len(unifiedvocabulary),1))
        occurrencecounts = np.bincount(indices_in_unifiedvocabulary[k],
                                    minlength=len(unifiedvocabulary)).reshape(-1, 1)
        unifiedvocabulary_totaloccurrencecounts= \
            unifiedvocabulary_totaloccurrencecounts+occurrencecounts
        unifiedvocabulary_documentcounts= \
            unifiedvocabulary_documentcounts+(occurrencecounts>0)
    # Mean occurrence counts over documents
    unifiedvocabulary_meancounts= \
    unifiedvocabulary_totaloccurrencecounts/len(crawled_lematizedtexts)

    # Second pass to count variances
    for k in range(len(crawled_lematizedtexts)):
        occurrencecounts=np.zeros((len(unifiedvocabulary),1))
        occurrencecounts = np.bincount(indices_in_unifiedvocabulary[k],
                                    minlength=len(unifiedvocabulary)).reshape(-1, 1)
        unifiedvocabulary_countvariances=unifiedvocabulary_countvariances+ \
            (occurrencecounts-unifiedvocabulary_meancounts)**2
    unifiedvocabulary_countvariances= \
        unifiedvocabulary_countvariances/(len(crawled_lematizedtexts)-1)

    highest_totaloccurrences_indices=np.argsort(\
        -1*unifiedvocabulary_totaloccurrencecounts,axis=0)
    print(np.squeeze(unifiedvocabulary[\
        highest_totaloccurrences_indices[0:100]]))
    print(np.squeeze(\
        unifiedvocabulary_totaloccurrencecounts[\
        highest_totaloccurrences_indices[0:100]]))


    # Tell the library we want each plot in its own window
    # Create a figure and an axis
    myfigure, myaxes = matplotlib.pyplot.subplots()
    # Plot the sorted occurrence counts of the words against their ranks
    horizontalpositions=range(len(unifiedvocabulary))
    verticalpositions=np.squeeze(unifiedvocabulary_totaloccurrencecounts[\
    highest_totaloccurrences_indices])
    myaxes.plot(horizontalpositions,verticalpositions)
    # Plot the top-500 occurrence counts of the words against their ranks
    myfigure, myaxes = matplotlib.pyplot.subplots()
    horizontalpositions=range(500)
    verticalpositions=np.squeeze(unifiedvocabulary_totaloccurrencecounts[\
    highest_totaloccurrences_indices[0:500]])
    myaxes.plot(horizontalpositions,verticalpositions, label="Data")

    # Plot zifp's law
    a = 1.1
    r, probs = zipf_curve(a, 500)
    probs_scaled = probs * np.sum(verticalpositions)
    matplotlib.pyplot.plot(r, probs_scaled, 'r-', label=f"Zipf a={a}")
    matplotlib.pyplot.legend()

    matplotlib.pyplot.show()

if __name__ == '__main__':
    main()
