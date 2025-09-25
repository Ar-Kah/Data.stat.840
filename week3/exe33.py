import tqdm
import scipy
import numpy as np
import nltk
import bs4
import requests
import re

# import functions that are used a lot
from exe32 import getbookcontents, getbooktext, cleanuptext, lematizetext, findwordindex

def compute_p_values(n_vocab, distanceoccurrences, absdistancemeans, absdistancevariances,
                     overallabsdistancemean, overallabsdistancevariance, overalldistancecount, absdistancepvalues ):
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
    for k in range(min(20,len(lowest_meandistances_indices))):
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


def print_N_lowestmeandistance(N, absdistancemeans, absdistancepvalues, distanceoccurrences, words):

    min_occurrences = 5

    all_pairs_nn_an = []
    words_tagged = nltk.pos_tag(words)

    for wordindex, word in enumerate(words):
        tempindices = np.nonzero(distanceoccurrences[wordindex,:] >= min_occurrences)[1]
        for otherwordindex in tempindices:

            tag_of_word = words_tagged[wordindex][1][0]
            tag_of_other_word = words_tagged[otherwordindex][1][0]

            # Skip over wordparis that are not noun-noun or noun-adjective pairs
            if not ((tag_of_word in ['N', 'J']) and (tag_of_other_word in ['N', 'J'])):
                continue

            if otherwordindex <= wordindex:   # skip duplicates
                continue
            
            mean_dist = absdistancemeans[wordindex, otherwordindex]
            # convert sparse to scalar safely
            p_value = absdistancepvalues[wordindex, otherwordindex]
            all_pairs_nn_an.append((wordindex, otherwordindex, mean_dist, p_value))

    # Sort by mean distance ascending
    all_pairs_sorted = sorted(all_pairs_nn_an, key=lambda x: x[2])

    # Take top N
    top_N_pairs = all_pairs_sorted[:N]

    print(f"Print the {N} word pairs with the lowest absmeandistance that occured atleast {min_occurrences} times in the text")
    for word_i, word_j, mean_d, p in top_N_pairs:
        print(f"{words[word_i]}, {words[word_j]}, absmeandistance: {mean_d:.2f}, p value: {p:f}")

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
            0:int(np.floor(len(vocabulary)*0.001))]):
            pruning_desition[index] = 1

        elif highest_totaloccurrences_indeces[index] < 2:
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

    # Map old indices to new ones in the remaining vocabulary
    old_to_new = -1 * np.ones(len(vocabulary), dtype=int)
    for new_idx, old_idx in enumerate(remainingindices):
        old_to_new[old_idx] = new_idx

    # Remap the text sequence to pruned vocabulary indices
    remapped_text_in_indices = [old_to_new[i] for i in text_in_indices if old_to_new[i] != -1]

    # Correct: slice vocabulary using remainingindices
    remainingvocabulary = vocabulary[remainingindices]

    remainingvocabulary_totaloccurrencecounts = vocabulary_totaloccurrencecounts[remainingindices]
    remaining_highest_totaloccurrences_indices = np.argsort(-1 * remainingvocabulary_totaloccurrencecounts, axis=0)

    print(np.squeeze(remainingvocabulary[remaining_highest_totaloccurrences_indices[0:100]]))
    print(np.squeeze(remainingvocabulary_totaloccurrencecounts[remaining_highest_totaloccurrences_indices[0:100]]))
            

    # Word pair occurance counting
    # ------------------------------------------

    vocab_size = len(remainingvocabulary)
    distanceoccurrences = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    sumdistances = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    sumabsdistances = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)
    sumdistancesquares = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)

    latestoccurencepositions = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)

    count_word_pair_occurances(remapped_text_in_indices, latestoccurencepositions, distanceoccurrences,
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

    absdistancepvalues = scipy.sparse.lil_matrix((vocab_size, vocab_size))

    compute_p_values(vocab_size, distanceoccurrences, absdistancemeans, absdistancevariances, overallabsdistancemean,
                        overallabsdistancevariance, overalldistancecount, absdistancepvalues)

    print_N_lowestmeandistance(N, absdistancemeans, absdistancepvalues, distanceoccurrences, remainingvocabulary)

    print_top_p_collections('dog', distanceoccurrences, absdistancepvalues, remainingvocabulary,
                            absdistancemeans, absdistancevariances, distancemeans,
                            distancevariances, remainingvocabulary)

    # ----------------------------------------------------
    # End of main



if __name__ == '__main__':
    main()


"""
Comments on exercise 3.3 output

Print the 20 word pairs with the lowest absmeandistance that occured atleast 5 times in the text
air, mid, absmeandistance: 1.00, p value: 0.000000
brother, wild, absmeandistance: 1.00, p value: 0.000000
club, fang, absmeandistance: 1.00, p value: 0.000000
half-breed, scotch, absmeandistance: 1.00, p value: 0.000000
red, sweater, absmeandistance: 1.00, p value: 0.000000
beast, primordial, absmeandistance: 1.20, p value: 0.000014
dominant, primordial, absmeandistance: 1.20, p value: 0.000014
neck, shoulder, absmeandistance: 1.20, p value: 0.000014
foot, stagger, absmeandistance: 1.40, p value: 0.000266
live, thing, absmeandistance: 1.50, p value: 0.000001
club, law, absmeandistance: 1.57, p value: 0.000249
toil, trace, absmeandistance: 1.57, p value: 0.000051
circle, silent, absmeandistance: 1.60, p value: 0.000323
dog, want, absmeandistance: 1.60, p value: 0.000047
hind, leg, absmeandistance: 1.60, p value: 0.001510
dog, outside, absmeandistance: 1.62, p value: 0.000240
eat, kill, absmeandistance: 1.67, p value: 0.000302
john, thornton, absmeandistance: 1.75, p value: 0.000000
beast, dominant, absmeandistance: 1.80, p value: 0.000026
dog, southland, absmeandistance: 1.80, p value: 0.001655

The book contains a lot of atmosferic and describing elements
and the book gives of a primitive vibe. For example the use of
adjectives such as wild, dominant and primordial coupled with words
like kill, beast and dog. In any case this fits well with the
title of the book "Call of the wild". You would expect somthing
like this in a book called like this
"""
