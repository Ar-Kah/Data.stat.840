import re
import requests
import nltk
import numpy as np
import scipy


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



def getbookparagraphs_from_rawtext(raw_text):
    """
    raw_text: the full ebook text as a single string (use response.text).
    Returns a list of paragraph strings (no empty ones), with internal newlines preserved
    until cleaning step.
    """
    # split into lines so we can find Gutenberg start/end markers reliably
    book_lines = raw_text.splitlines()

    start_index = 0
    end_index = len(book_lines)
    for i, line in enumerate(book_lines):
        if "START OF THE PROJECT GUTENBERG EBOOK" in line:
            start_index = i + 1
        elif "END OF THE PROJECT GUTENBERG EBOOK" in line:
            end_index = i
            break

    # join back to single string for paragraph-splitting
    book_body = "\n".join(book_lines[start_index:end_index])

    paragraphs = re.split(r'\n[ \n]*\n', book_body)

    # strip and remove empties
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs

def clean_paragraphs(paragraphs):
    """
    Collapse internal whitespace (including newlines) to single spaces,
    strip punctuation-ish characters from words, return cleaned paragraph strings.
    """
    cleaned = []
    for p in paragraphs:
        # collapse any whitespace (tabs/newlines/multiple spaces) into a single space
        p = re.sub(r'\s+', ' ', p).strip()

        # optional: more aggressive token-level cleanup
        words = p.split()
        cleaned_words = [w.strip('"[]_*(){}.,:;?!\'“”') .lower() for w in words]

        cleaned.append(" ".join(cleaned_words))
    return cleaned


def text_processing(paragraphs: list):

    nltk_paragraphs = []
    for p in paragraphs:
        tokenized_paragraph = nltk.word_tokenize(p)
        nltk_textp = nltk.Text(tokenized_paragraph)
        nltk_paragraphs.append(nltk_textp)
        
    return nltk_paragraphs



def lemmatize_paragraph(tockenized_paragraph):
    lemmatizer = nltk.stem.WordNetLemmatizer()

    word_tag_tuples = nltk.pos_tag(tockenized_paragraph)

    lemmatized_paragraph = []

    for index in range(len(word_tag_tuples)):
        # Lemmatize each word in the paragraph
        word_to_lemmatize = word_tag_tuples[index][0]       # first elemtent of tuple is the word
        wordnet_tag = tagtowordnet(word_tag_tuples[index][1])    # second element is the tag of the word

        if wordnet_tag != -1:
            lemmatized_word = lemmatizer.lemmatize(word_to_lemmatize, wordnet_tag)
        else:
            lemmatized_word = word_to_lemmatize

        # Store the lemmatized words
        lemmatized_paragraph.append(lemmatized_word)

    return lemmatized_paragraph



def remake_vocab_as_integers(paragraphs):
    vocabularies = []
    paragraphs_as_integers = []

    for paragraph in paragraphs:
        uniques, indeces = np.unique(paragraph, return_inverse=True)
        vocabularies.append(uniques)
        paragraphs_as_integers.append(indeces)
        
    return vocabularies, paragraphs_as_integers


def main():
    url = "https://www.gutenberg.org/cache/epub/55/pg55.txt"
    r = requests.get(url)
    raw_text = r.text

    # get paragraphs
    paragraphs = getbookparagraphs_from_rawtext(raw_text)

    cleaned = clean_paragraphs(paragraphs)

    # Format paragraphs into nltk Text format
    tokenized_paragrapths_nltk = text_processing(cleaned)

    # Lemmatize tokenized paragraphs
    print("lemmatizing words")
    lemmatized_paragraphs = []
    for tokenized_paragraph in tokenized_paragrapths_nltk:
        lemmatized_paragraph = lemmatize_paragraph(tokenized_paragraph)
        lemmatized_paragraphs.append(lemmatized_paragraph)
    print()
        

    print("Finding unique words")
    # Create vocabulary if each paragraph and recreate the original paragraph
    # with the vocabulary indices
    vocabularies, paragraphs_as_indices = remake_vocab_as_integers(lemmatized_paragraphs)
    print()

    # Make a unified vocabulary of the previously made individual paragraph vocabularies
    temp_vocab = []
    print("Making unified vocabulary")
    n_docs = len(lemmatized_paragraphs)
    for index in range(n_docs):
        temp_vocab.extend(vocabularies[index])
    print()

    # Concatenate all of the words into a single array
    concatenated_words = []
    for p in paragraphs_as_indices:
        concatenated_words.extend(p)

    unified_vocabulary, individual_vocabularies_as_indices = np.unique(temp_vocab, return_inverse=True)

    n_vocab = len(unified_vocabulary)
    
    tf_matrix = scipy.sparse.lil_matrix((n_docs, n_vocab))
    
    for index, paragraph in enumerate(paragraphs_as_indices):
        # word is now in the for of a integer
        count_of_words = len(paragraph)
        bins_of_occurrences = np.bincount(paragraph, minlength=n_vocab)
        tf_values = bins_of_occurrences / count_of_words

        # insert values into the matrix
        tf_matrix[index, :] = tf_values


if __name__ == "__main__":
    main()
