import numpy as np
import nltk
import re
import requests
import scipy
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

def tagtowordnet(postag):
    if postag[0]=='N':
        return 'n'
    elif postag[0]=='V':
        return 'v'
    elif postag[0]=='J':
        return 'a'
    elif postag[0]=='R':
        return 'r'
    return None

def getparagraphs_from_rawtext(raw_text: str):

    lines = raw_text.splitlines()
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        if "START OF THE PROJECT GUTENBERG EBOOK" in line:
            start = i + 1
        elif "END OF THE PROJECT GUTENBERG EBOOK" in line:
            end = i
            break
    body = '\n'.join(lines[start: end])
    paragraphs = re.split('\n[ \n]*\n', body)
    return paragraphs


def clean_paragraphs(paragraphs):
    cleaned = []
    for p in paragraphs:
        p = re.sub(r'\s+', ' ', p).strip()
        words = p.split()
        words = [w.strip('"[]_*(){}.,:;?!\'“”').lower() for w in words]
        cleaned.append(words)
    return cleaned


def getlongest_paragraph(paragraphs:list):
    current_max = 0
    max_length_index = -1
    for index, p in enumerate(paragraphs):
        paragraph_legth = len(p)
        if paragraph_legth > current_max:
            current_max = paragraph_legth
            max_length_index = index

    if max_length_index == -1:
        return

    return paragraphs[max_length_index]


def lemmatize_paragraph(paragraph):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    pos_tags = nltk.pos_tag(paragraph)
    lemmatized = []
    for word, tag in pos_tags:
        wn_tag = tagtowordnet(tag)
        if wn_tag:
            lemmatized.append(lemmatizer.lemmatize(word, wn_tag))
        else:
            lemmatized.append(word)
    return lemmatized

def prune_vocabulary(vocabulary, top_indices):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    mask = np.ones(len(vocabulary), dtype=bool)
    for i, word in enumerate(vocabulary):
        if word in stopwords or len(word)<2 or len(word)>20:
            mask[i] = False
        elif i in top_indices[:max(1, len(vocabulary)//100)]:
            mask[i] = False
    return mask


def build_tf_idf(paragraphs, vocabulary):
    word_to_idx = {w:i for i, w in enumerate(vocabulary)}
    n_docs = len(paragraphs)
    n_vocab = len(vocabulary)
    tf_matrix = scipy.sparse.lil_matrix((n_docs, n_vocab))
    
    for doc_id, doc in enumerate(paragraphs):
        indices = [word_to_idx[w] for w in doc if w in word_to_idx]
        if indices:
            counts = np.bincount(indices, minlength=n_vocab)
            tf_matrix[doc_id,:] = counts / len(doc)
    
    df_vector = np.array((tf_matrix > 0).sum(axis=0)).flatten()
    idf_vector = 1 + np.log((n_docs + 1) / (df_vector + 1))
    tfidf_matrix = tf_matrix.multiply(idf_vector)
    return scipy.sparse.csr_matrix(tfidf_matrix)

def build_td_idf_second(paragraphs, vocabulary):
    word_to_idx = {w:i for i, w in enumerate(vocabulary)}
    n_vocab = len(vocabulary)
    n_docs = len(paragraphs)
    tf_matrix = scipy.sparse.lil_matrix((n_docs, n_vocab))

    for doc_idx, doc in enumerate(paragraphs):
        indeces = [word_to_idx[w] for w in doc if w in word_to_idx]
        if indeces:
            counts = np.bincount(indeces, minlength=n_vocab)
            tf_matrix[doc_idx,:] = np.log(counts + 1)

    df_vector = np.array((tf_matrix > 0).sum(axis=0)).flatten()
    idf_vector = 1 + np.log((n_docs + 1) / (df_vector + 1))
    tfidf_matrix = tf_matrix.multiply(idf_vector)
    return scipy.sparse.csr_matrix(tfidf_matrix)

def build_td_idf_third(paragraphs, vocab):
    word_to_idx = {w:i for i, w in enumerate(vocab)}
    n_vocab = len(vocab)
    n_docs = len(paragraphs)

    tf_matrix = scipy.sparse.lil_matrix((n_docs, n_vocab))
    alpha = 0.1

    for doc_idx, doc in enumerate(paragraphs):
        indeces = [word_to_idx[w] for w in doc if w in word_to_idx]
        if indeces:
            counts = np.bincount(indeces, minlength=n_vocab)
            tf_matrix[doc_idx,:] = alpha + (1 - alpha) * (counts / np.max(counts))

    df_vector = np.array((tf_matrix > 0).sum(axis=0)).flatten() + 1
    idf_vector = np.log(df_vector.max() / 1 + df_vector)
    tfidf_matrix = tf_matrix.multiply(idf_vector)
    return scipy.sparse.csr_matrix(tfidf_matrix)

def print_top_words(N, list_of_tfidfvectors, vocab):
    for nparray in list_of_tfidfvectors:
        
        tfidf_sum = np.array(nparray.sum(axis=0)).flatten()
        top_indices_sorted = np.argsort(-tfidf_sum)[:20]
        top_words = vocab[top_indices_sorted]

        print(f"Top {N} words: {' '.join(top_words)}")
    
def main():
    # I have used AI for debuging purposses
    url = "https://www.gutenberg.org/cache/epub/55/pg55.txt"
    r = requests.get(url)
    raw_text = r.text

    paragraphs = getparagraphs_from_rawtext(raw_text)

    print("Longest paragraph:")
    print()
    print(getlongest_paragraph(paragraphs))
    print()

    cleaned_paragraphs = clean_paragraphs(paragraphs)
    lemmatized_paragraphs = [lemmatize_paragraph(p) for p in cleaned_paragraphs]

    # Build vocabulary
    all_words = [w for p in lemmatized_paragraphs for w in p]
    vocab, inverse = np.unique(all_words, return_inverse=True)
    
    counts = np.bincount(inverse)
    top_indices = np.argsort(-counts)

    mask = prune_vocabulary(vocab, top_indices)
    pruned_vocab = vocab[mask]

    pruned_paragraphs = [[w for w in p if w in pruned_vocab] for p in lemmatized_paragraphs]

    # Build the length-normalized frequency TH part and smoother logarithmic inverse document
    # frequency for the IDF part
    tfidf_vector = build_tf_idf(pruned_paragraphs, pruned_vocab)
    second_tfidf_vector = build_td_idf_second(pruned_paragraphs, pruned_vocab)
    third_tfidf_vector = build_td_idf_third(pruned_paragraphs, pruned_vocab)

    list_of_tfidfvectors = [tfidf_vector, second_tfidf_vector, third_tfidf_vector]

    print_top_words(20, list_of_tfidfvectors, pruned_vocab)

if __name__ == '__main__':
    main()

    """
OUTPUT
python3 exe42.py 
Longest paragraph:

She left Dorothy alone and went back to the others. These she also led
to rooms, and each one of them found himself lodged in a very pleasant
part of the Palace. Of course this politeness was wasted on the
Scarecrow; for when he found himself alone in his room he stood
stupidly in one spot, just within the doorway, to wait till morning. It
would not rest him to lie down, and he could not close his eyes; so he
remained all night staring at a little spider which was weaving its web
in a corner of the room, just as if it were not one of the most
wonderful rooms in the world. The Tin Woodman lay down on his bed from
force of habit, for he remembered when he was made of flesh; but not
being able to sleep, he passed the night moving his joints up and down
to make sure they kept in good working order. The Lion would have
preferred a bed of dried leaves in the forest, and did not like being
shut up in a room; but he had too much sense to let this worry him, so
he sprang upon the bed and rolled himself up like a cat and purred
himself asleep in a minute.

Top 20 words: ask scarecrow oz lion woodman go come shall tin girl answer get witch great chapter see back little make heart
Top 20 words: scarecrow come lion woodman oz go great make ask tin little witch one see could would get head back look
Top 20 words: scarecrow lion ask oz woodman come go great tin little make witch see get one back girl could shall answer

The top 20 words are very similar in in the different tfidf matrices, but the order differece a little
    """
