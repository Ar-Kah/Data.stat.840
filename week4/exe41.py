import re
import requests
import nltk
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

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

def getbookparagraphs_from_rawtext(raw_text):
    lines = raw_text.splitlines()
    start_index = 0
    end_index = len(lines)
    for i, line in enumerate(lines):
        if "START OF THE PROJECT GUTENBERG EBOOK" in line:
            start_index = i + 1
        elif "END OF THE PROJECT GUTENBERG EBOOK" in line:
            end_index = i
            break
    body = "\n".join(lines[start_index:end_index])
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
    return paragraphs

def clean_paragraphs(paragraphs):
    cleaned = []
    for p in paragraphs:
        p = re.sub(r'\s+', ' ', p).strip()
        words = p.split()
        words = [w.strip('"[]_*(){}.,:;?!\'“”').lower() for w in words]
        cleaned.append(words)
    return cleaned

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
    tf_matrix = lil_matrix((n_docs, n_vocab))
    
    for doc_id, doc in enumerate(paragraphs):
        indices = [word_to_idx[w] for w in doc if w in word_to_idx]
        if indices:
            counts = np.bincount(indices, minlength=n_vocab)
            tf_matrix[doc_id,:] = counts
    
    df_vector = np.array((tf_matrix > 0).sum(axis=0)).flatten()
    idf_vector = 1 + np.log((n_docs + 1) / (df_vector + 1))
    tfidf_matrix = tf_matrix.multiply(idf_vector)
    return csr_matrix(tfidf_matrix)

def main():
    url = "https://www.gutenberg.org/cache/epub/55/pg55.txt"
    r = requests.get(url)
    raw_text = r.text

    paragraphs = getbookparagraphs_from_rawtext(raw_text)
    cleaned = clean_paragraphs(paragraphs)
    lemmatized_paragraphs = [lemmatize_paragraph(p) for p in cleaned]

    # Build vocabulary
    all_words = [w for p in lemmatized_paragraphs for w in p]
    vocab, inverse = np.unique(all_words, return_inverse=True)
    
    counts = np.bincount(inverse)
    top_indices = np.argsort(-counts)

    # Prune vocabulary
    mask = prune_vocabulary(vocab, top_indices)
    pruned_vocab = vocab[mask]

    # Build TF-IDF
    pruned_paragraphs = [[w for w in p if w in pruned_vocab] for p in lemmatized_paragraphs]
    tfidf_matrix = build_tf_idf(pruned_paragraphs, pruned_vocab)

    # Normalize rows
    X = normalize(tfidf_matrix, norm='l2', axis=1)

    # Fit Gaussian Mixture
    gmm = GaussianMixture(n_components=10, covariance_type='diag', max_iter=100, init_params='random')
    gmm.fit(X.toarray())

    # Print top words per cluster
    for k in range(10):
        top_features = np.argsort(-gmm.means_[k,:])[:20]
        print(f"Cluster {k}: ", ' '.join(pruned_vocab[top_features]))

if __name__ == "__main__":
    main()

