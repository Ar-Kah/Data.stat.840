import tqdm
import scipy
import numpy as np
import nltk
import bs4
import requests
import re

# import functions that are used a lot
from exe32 import getbookcontents, getbooktext, cleanuptext, lematizetext, findwordindex

def compute_p_values(n_vocab, distanceoccurrences, absdistancemeans, absdistancevariances, overallabsdistancemean, overallabsdistancevariance, overalldistancecount, absdistancepvalues ):
    for m in range(n_vocab):
        # Find pairs of word m
        tempindices=np.nonzero(distanceoccurrences[m,:]>1)[1]
        # For computation we need to transform these to non-sparse vectors
        meanterm=absdistancemeans[m,tempindices].todense()
        varianceterm=absdistancevariances[m,tempindices].todense()
        occurrenceterm=distanceoccurrences[m,tempindices].todense()
        # Compute the t-test statistic for each pair
        tempstatistic=(meanterm-overallabsdistancemean)/ \
        np.sqrt(varianceterm/occurrenceterm+ \
        overallabsdistancevariance/overalldistancecount)
        # Compute the t-test degrees of freedom for each pair
        tempdf=(np.power(varianceterm/occurrenceterm+\
        overallabsdistancevariance/overalldistancecount,2))/ \
        ( (np.power(varianceterm/occurrenceterm,2))/(occurrenceterm-1)+ \
        ((overallabsdistancevariance/overalldistancecount)**2)/ \
        (overalldistancecount-1) )
        # Compute the t-test p-value for each pair
        temppvalue=scipy.stats.t.cdf(tempstatistic,tempdf)
        # Store the t-test p-value for each pair
        absdistancepvalues[m,tempindices]=np.squeeze(np.array(temppvalue))


def print_top_p_collections(wordstring, distanceoccurrences, absdistancepvalues, remainingvocabulary,
                            absdistancemeans, absdistancevariances, distancemeans, distancevariances, wordarray):

    # Find the chosen word and words that occurred with it at least 2 times
    mywordindex=findwordindex(wordstring, wordarray)
    if mywordindex==-1:
        print('Word not found: '+wordstring)
        return

    # Require at least 10 pair occurrences
    minpairoccurrences=10
    tempindices=np.nonzero(distanceoccurrences[mywordindex,:]>minpairoccurrences)[1]
    # Sort the pairs by lowest pvalue
    lowest_meandistances_indices=np.argsort(np.squeeze(np.array(absdistancepvalues[mywordindex,tempindices].todense())),axis=0)
    # Print the top-50 lowest-distance pairs
    print('\nLowest p-values\n')
    for k in range(min(50,len(lowest_meandistances_indices))):
        otherwordindex=tempindices[lowest_meandistances_indices[k]]
        # Print the words, their absolute distances (mean+std) and distances (mean+std)
        print('{!s}--{!s}: {:d} occurrences, absdist: {:.1f} +- {:.1f}, offset: {:.1f} +- {:.1f}, pvalue: {:f}'.format(
            remainingvocabulary[mywordindex],
            remainingvocabulary[otherwordindex],
            int(distanceoccurrences[mywordindex,otherwordindex]),
            absdistancemeans[mywordindex,otherwordindex],
            np.sqrt(absdistancevariances[mywordindex,otherwordindex]),
            distancemeans[mywordindex,otherwordindex],
            np.sqrt(distancevariances[mywordindex,otherwordindex]),
            absdistancepvalues[mywordindex,otherwordindex]))

def count_statistics(vocab_size, distanceoccurrences, distancemeans, sumdistances, absdistancemeans, sumabsdistances,
                    distancevariances, sumdistancesquares, absdistancevariances):

    print("Count occurance statistics")
    for m in tqdm.tqdm(range(vocab_size), ncols=100):
        # Find the column indices that have at least two occurrences
        tempindices = np.nonzero(distanceoccurrences[m, :].toarray() > 1)[1]

        for j in tempindices:
            count = distanceoccurrences[m, j]
            if count <= 1:
                continue

            # Mean distances
            distancemeans[m, j] = sumdistances[m, j] / count
            absdistancemeans[m, j] = sumabsdistances[m, j] / count

            # Variance of signed distance
            meanterm = distancemeans[m, j] ** 2
            meanterm *= count / (count - 1)
            distancevariances[m, j] = sumdistancesquares[m, j] / (count - 1) - meanterm

            # Variance of absolute distance
            meanterm = absdistancemeans[m, j] ** 2
            meanterm *= count / (count - 1)
            absdistancevariances[m, j] = sumdistancesquares[m, j] / (count - 1) - meanterm

    print()
    # ------------------------------
    # End of count_statistics



def count_word_pair_occurances(textinindicies, latestoccurencepositions, distanceoccurrences, sumdistances,
                     sumabsdistances, sumdistancesquares):

    # Count occurrances of two words
    print("Counting occurrances")
    for index, currentword in enumerate(tqdm.tqdm(textinindicies, ncols=100)):

        windowsize = min(index, 10)

        for x in range(windowsize):

            previousword = textinindicies[index - x - 1]

            if latestoccurencepositions[currentword, previousword] < index:

                distanceoccurrences[currentword, previousword] += 1
                sumdistances[currentword, previousword] += ((index - x - 1) - index)
                sumabsdistances[currentword, previousword] += abs((index - x - 1) - index)
                sumdistancesquares[currentword, previousword] += ((index - x - 1) - index) ** 2

                # Store the same occurance but change the word indecies arround
                distanceoccurrences[previousword, currentword] += 1
                sumdistances[previousword, currentword] += ((index - x - 1) - index)
                sumabsdistances[previousword, currentword] += abs((index - x - 1) - index)
                sumdistancesquares[previousword, currentword] += ((index - x - 1) - index) ** 2

                # Mark that we found this pair while counting down frrom index,
                # so we do not count more distant occurrences of the pair
                latestoccurencepositions[currentword, previousword] = index
                latestoccurencepositions[previousword, currentword] = index

    print()
    # ---------------------------
    # End of count_occurrances


def print_N_lowestmeandistance(N, absdistancemeans, distanceoccurrences, words):

    all_pairs = []

    for wordindex, word in enumerate(words):
        tempindices = np.nonzero(distanceoccurrences[wordindex,:] > 1)[1]
        for j in tempindices:
            if j <= wordindex:   # skip duplicates
                continue
            mean_dist = absdistancemeans[wordindex, j]
            # convert sparse to scalar safely
            all_pairs.append((wordindex, j, mean_dist))

    # Sort by mean distance ascending
    all_pairs_sorted = sorted(all_pairs, key=lambda x: x[2])

    # Take top N
    top_N_pairs = all_pairs_sorted[:N]

    for word_i, word_j, mean_d in top_N_pairs:
        print(words[word_i], words[word_j], mean_d)

    # End of print_N_lowestmeandistance
    # -----------------------------------


def prune_text(vocabulary, highest_totaloccurrences_indeces):
    english_stopwords = nltk.corpus.stopwords.words('english')
    pruning_desition = np.zeros((len(vocabulary), 1), dtype=bool)

    for index in range(len(vocabulary)):
        if vocabulary[index] in english_stopwords:
            pruning_desition[index] = 1

        elif len(vocabulary[index]) < 2:
            pruning_desition[index] = 1

        elif len(vocabulary[index]) > 20:
            pruning_desition[index] = 1

        elif (index in highest_totaloccurrences_indeces[\
            0:int(np.floor(len(vocabulary)*0.01))]):
            pruning_desition[index] = 1

        elif highest_totaloccurrences_indeces[index] < 4:
            pruning_desition[index] = 1

    return pruning_desition


def main():

    # Retreve book contents and manipulate text
    # -----------------------------------------

    website_url = "https://www.gutenberg.org/ebooks/215"
    book_url = getbookcontents(website_url)
    bookpage_html = requests.get(book_url)
    bookpage_parsed = bs4.BeautifulSoup(bookpage_html.content, 'html.parser')
    text = getbooktext(bookpage_parsed)
    clean_text = cleanuptext(text)

    tokenized_text = nltk.word_tokenize(clean_text)
    nltk_text = nltk.Text(tokenized_text)

    lemmatized_nltktext = lematizetext(nltk_text)
    lemmatized_nltktext=nltk.Text(lemmatized_nltktext)


    uniqueresults = np.unique(lemmatized_nltktext, return_inverse=True)
    vocabulary = uniqueresults[0]
    text_in_indices = uniqueresults[1]

    # Word occurrences 
    # ------------------------------------------

    vocabulary_totaloccurrencecounts=np.zeros((len(vocabulary),1))
    vocabulary_meancounts=np.zeros((len(vocabulary),1))
    vocabulary_countvariances=np.zeros((len(vocabulary),1))

    # vocabulary_totaloccurrencescounts is a variable that has stored the count of the
    # word/index in the doccument (in this occation the book) so lets say the
    # word "book" is in index "0" vocabulary_totaloccurrencescounts will return the
    # occurence count of that word in index 0
    vocabulary_totaloccurrencecounts = np.bincount(text_in_indices, minlength=len(vocabulary))
    vocabulary_meancounts = vocabulary_totaloccurrencecounts / len(text_in_indices)

    vocabulary_countvariances = (len(vocabulary) - vocabulary_meancounts) ** 2

    # highest_totaloccurrences_indices is a variable that has stored the word/index that
    # occured the most in the document
    highest_totaloccurrences_indices = np.argsort(-1 * vocabulary_totaloccurrencecounts) # multiply bu -1 to get the most frequent

    # these lines of code will print the 100 most common words and the occurence counts
    # print(np.squeeze(vocabulary[highest_totaloccurrences_indices[0:100]]))
    # print(np.squeeze(vocabulary_totaloccurrencecounts[highest_totaloccurrences_indices[0:100]]))


    # Word pruning
    # -----------------------------------------
    
    print("Prune vocabulary\n")
    pruning_decision = prune_text(vocabulary, highest_totaloccurrences_indices)

    remainingindices = np.where(pruning_decision==0)[0]
    # Map old indices to new ones in the remaingn vocabulary
    old_to_new = -1 * np.ones(len(vocabulary), dtype=int)
    for new_idx, old_idx in enumerate(remainingindices):
        old_to_new[old_idx] = new_idx

    remapped_indices = [old_to_new[i] for i in remainingindices if old_to_new[i] != -1]
    remainingvocabulary=vocabulary[remapped_indices]

    remainingvocabulary_totaloccurrencecounts= \
        vocabulary_totaloccurrencecounts[remainingindices]
    remaining_highest_totaloccurrences_indices= \
        np.argsort(-1*remainingvocabulary_totaloccurrencecounts,axis=0)
    print(np.squeeze(remainingvocabulary[remaining_highest_totaloccurrences_indices[0:100]]))
    print(np.squeeze(remainingvocabulary_totaloccurrencecounts[
        remaining_highest_totaloccurrences_indices[0:100]
    ]))
            

    # Word pair occurance counting
    # ------------------------------------------

    vocab_size = len(remainingvocabulary)
    distanceoccurrences = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    sumdistances = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    sumabsdistances = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    sumdistancesquares = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)

    latestoccurencepositions = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)

    count_word_pair_occurances(remapped_indices, latestoccurencepositions, distanceoccurrences,
                     sumdistances, sumabsdistances, sumdistancesquares)


    # Statistics
    # ------------------------------------------

    distancemeans = scipy.sparse.lil_matrix((vocab_size, vocab_size))
    absdistancemeans = scipy.sparse.lil_matrix((vocab_size, vocab_size))
    distancevariances = scipy.sparse.lil_matrix((vocab_size, vocab_size))
    absdistancevariances = scipy.sparse.lil_matrix((vocab_size, vocab_size))

    count_statistics(vocab_size, distanceoccurrences, distancemeans, sumdistances, absdistancemeans,
                        sumabsdistances, distancevariances, sumdistancesquares, absdistancevariances)

    overalldistancecount=np.sum(distanceoccurrences)
    overalldistancesum=np.sum(sumdistances)
    overallabsdistancesum=np.sum(sumabsdistances)
    overalldistancesquaresum=np.sum(sumdistancesquares)
    overalldistancemean=overalldistancesum / overalldistancecount
    overallabsdistancemean=overallabsdistancesum / overalldistancecount
    overalldistancevariance=overalldistancesquaresum / (overalldistancecount-1) -\
                                overalldistancecount / (overalldistancecount - 1) * overalldistancemean
    overallabsdistancevariance=overalldistancesquaresum / (overalldistancecount-1) -\
                                overalldistancecount/(overalldistancecount - 1) * overallabsdistancemean


    mywordindex=findwordindex('foot', remainingvocabulary)
    tempindices=np.nonzero(distanceoccurrences[mywordindex,:]>1)[1]
    # Sort the pairs by lowest mean absolute distance
    lowest_meandistances_indices=np.argsort(np.squeeze(np.array(\
        absdistancemeans[mywordindex,tempindices].todense())),axis=0)
    N = 20
    # Print n lowest meandistance indices
    print_N_lowestmeandistance(N, absdistancemeans, distanceoccurrences, remainingvocabulary)

    absdistancepvalues = scipy.sparse.lil_matrix((vocab_size, vocab_size))

    compute_p_values(vocab_size, distanceoccurrences, absdistancemeans, absdistancevariances, overallabsdistancemean,
                        overallabsdistancevariance, overalldistancecount, absdistancepvalues)

    print_top_p_collections('dog', distanceoccurrences, absdistancepvalues, remainingvocabulary,
                            absdistancemeans, absdistancevariances, distancemeans,
                            distancevariances, remainingvocabulary)

    # ----------------------------------------------------
    # End of main



if __name__ == '__main__':
    main()
