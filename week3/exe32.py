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

    # uniqueresults = np.unique(lemmatized_nltktext, return_inverse=True)
    # uniquewords = uniqueresults[0]
    # textinindicies = uniqueresults[1]

    # vocab_size = len(textinindicies)
    # distanceoccurrences = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    # sumdistances = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    # sumabsdistances = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    # sumdistancesquares = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)

    # latestoccurencepositions = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)

    # # Count occurrances of two words
    # print("Counting occurrances")
    # for index, currentword in enumerate(tqdm.tqdm(textinindicies, ncols=100)):

    #     windowsize = min(index, 10)

    #     for x in range(windowsize):

    #         previousword = textinindicies[index - x - 1]

    #         if latestoccurencepositions[currentword, previousword] < index:
    #             distanceoccurrences[currentword, previousword] += 1
    #             sumdistances[currentword, previousword] += ((index - x - 1) - index)
    #             sumabsdistances[currentword, previousword] += abs((index - x - 1) - index)
    #             sumdistancesquares[currentword, previousword] += ((index - x - 1) - index) ** 2

    #             # Store the same occurance but change the word indecies arround
    #             distanceoccurrences[previousword, currentword] += 1
    #             sumdistances[previousword, currentword] += ((index - x - 1) - index)
    #             sumabsdistances[previousword, currentword] += abs((index - x - 1) - index)
    #             sumdistancesquares[previousword, currentword] += ((index - x - 1) - index) ** 2

    #             # Mark that we found this pair while counting down frrom index,
    #             # so we do not count more distant occurrences of the pair
    #             latestoccurencepositions[currentword, previousword] = index
    #             latestoccurencepositions[previousword, currentword] = index


    # distancemeans = scipy.sparse.lil_matrix((vocab_size, vocab_size))
    # absdistancemeans = scipy.sparse.lil_matrix((vocab_size, vocab_size))
    # distancevariances = scipy.sparse.lil_matrix((vocab_size, vocab_size))
    # absdistancevariances = scipy.sparse.lil_matrix((vocab_size, vocab_size))

    # print("Count occurance statistics")
    # for m in tqdm.tqdm(range(vocab_size), ncols=100):
    #     # Find the column indices that have at least two occurrences
    #     tempindices = np.nonzero(distanceoccurrences[m, :].toarray() > 1)[1]

    #     for j in tempindices:
    #         count = distanceoccurrences[m, j]
    #         if count <= 1:
    #             continue

    #         # Mean distances
    #         distancemeans[m, j] = sumdistances[m, j] / count
    #         absdistancemeans[m, j] = sumabsdistances[m, j] / count

    #         # Variance of signed distance
    #         meanterm = distancemeans[m, j] ** 2
    #         meanterm *= count / (count - 1)
    #         distancevariances[m, j] = sumdistancesquares[m, j] / (count - 1) - meanterm

    #         # Variance of absolute distance
    #         meanterm = absdistancemeans[m, j] ** 2
    #         meanterm *= count / (count - 1)
    #         absdistancevariances[m, j] = sumdistancesquares[m, j] / (count - 1) - meanterm

    # overalldistancecount=np.sum(distanceoccurrences)
    # overalldistancesum=np.sum(sumdistances)
    # overallabsdistancesum=np.sum(sumabsdistances)
    # overalldistancesquaresum=np.sum(sumdistancesquares)
    # overalldistancemean=overalldistancesum / overalldistancecount
    # overallabsdistancemean=overallabsdistancesum / overalldistancecount
    # overalldistancevariance=overalldistancesquaresum / (overalldistancecount-1) - overalldistancecount / (overalldistancecount - 1) * overalldistancemean
    # overallabsdistancevariance=overalldistancesquaresum / (overalldistancecount-1) - overalldistancecount/(overalldistancecount - 1) * overallabsdistancemean


    # mywordindex=findwordindex('science', uniquewords)
    # tempindices=np.nonzero(distanceoccurrences[mywordindex,:]>1)[1]
    # # Sort the pairs by lowest mean absolute distance
    # lowest_meandistances_indices=np.argsort(np.squeeze(np.array(\
    #     absdistancemeans[mywordindex,tempindices].todense())),axis=0)
    # # Print the top-50 lowest-distance pairs
    # for k in range(50):
    #     otherwordindex=tempindices[lowest_meandistances_indices[k]]
    #     # Print word pairs, absolute distances and distances (mean+std)
    #     print((textinindicies[mywordindex],\
    #         textinindicies[otherwordindex],\
    #         absdistancemeans[mywordindex,otherwordindex],\
    #         np.sqrt(absdistancevariances[\
    #         mywordindex,otherwordindex]),\
    #         distancemeans[mywordindex,otherwordindex],\
    #         np.sqrt(distancevariances[mywordindex,otherwordindex])))

    nltk_text.concordance('monster')
    print()
    nltk_text.concordance('science')
    print
    nltk_text.concordance('horror')
    print()
    nltk_text.concordance('fear')
    print()

    # End of main

if __name__ == '__main__':
    main()
