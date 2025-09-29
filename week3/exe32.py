import tqdm
import scipy
import numpy as np
import nltk
import bs4
import requests
import re

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
    return ''.join(book_contents)

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

def cleanuptext(text):
    split_text = text.split()
    clean_text_array = []
    for word in split_text:
        stripped_text = word.strip("[]_ *()")
        stripped_text = stripped_text.lower()
        clean_text_array.append(stripped_text)

    clean_text = " ".join(clean_text_array)
    return clean_text


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

def findwordindex(wordstring, wordarray):
    for index, word in enumerate(wordarray):
        if word == wordstring:
            return index

    return -1


def main():

    website_url = "https://www.gutenberg.org/ebooks/84"
    book_url = getbookcontents(website_url)
    bookpage_html = requests.get(book_url)
    bookpage_parsed = bs4.BeautifulSoup(bookpage_html.content, 'html.parser')
    text = getbooktext(bookpage_parsed)
    clean_text = cleanuptext(text)

    tokenized_text = nltk.word_tokenize(clean_text)
    nltk_text = nltk.Text(tokenized_text)

    lemmatized_nltktext = lematizetext(nltk_text)
    lemmatized_nltktext=nltk.Text(lemmatized_nltktext)


    nltk_text.concordance('monster')
    print()
    nltk_text.concordance('science')
    print()
    nltk_text.concordance('horror')
    print()
    nltk_text.concordance('fear')
    print()

    # End of main

if __name__ == '__main__':
    main()
