import scipy
import nltk
import os
import numpy as np

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

def open_and_clean_documents(documents: list):
    # Exclude header lines from each message
    excludedlinemarkers = [
        'Xref:', 'Path:', 'From:', 'Newsgroups:', 'Subject:', 'Summary:',
        'Keywords:', 'Message-ID:', 'Date:', 'Expires:', 'Followup-To:',
        'Distribution:', 'Organization:', 'Approved:', 'Supersedes:', 'Lines:',
        'NNTP-Posting-Host:', 'References:', 'Sender:', 'In-Reply-To:',
        'Article-I.D.:', 'Reply-To:', 'Nntp-Posting-Host:'
    ]

    clean_documents = []

    for doc_path in documents:
        with open(doc_path, 'r', encoding='latin-1') as file:
            lines = file.read().splitlines()  # splits text into lines, removes line breaks

        # Filter out header lines and empty lines
        filtered_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line == '' or stripped_line.split(' ')[0] in excludedlinemarkers:
                continue  # skip empty or header lines

            # Clean each word in the line: strip punctuation and lowercase
            words = stripped_line.split()
            cleaned_words = [w.strip('"-[]_*(){}.,:;?!\'“”').lower() for w in words]

            # Reconstruct the cleaned line
            cleaned_line = ' '.join(cleaned_words)
            filtered_lines.append(cleaned_line)

        # Restore line structure
        clean_text = '\n'.join(filtered_lines)
        clean_documents.append(clean_text)

    return clean_documents



def get_directories(path):
    
    valid = []

    for root,d_names,f_names in os.walk(path):
        for d in d_names:
            if not (d == "rec.sport.baseball" or \
               d == "rec.autos" or d == "rec.motorcycles" or \
               d == "rec.sport.hockey"):
                continue

            valid.append(os.path.join(root, d))


    return valid

def get_files_from_dirs(dirs):

    documents = []
    for d in dirs:
        for root,d_names,f_names in os.walk(d):
            for f in f_names:
                documents.append(root + '/' + f)

    return documents


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

def lemmatize_document(document):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    pos_tags = nltk.pos_tag(document.split())
    lemmatized = []
    for word, tag in pos_tags:
        wn_tag = tagtowordnet(tag)
        if wn_tag:
            lemmatized.append(lemmatizer.lemmatize(word, wn_tag))
        else:
            lemmatized.append(word)
    return lemmatized

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


def prune_vocabulary(vocabulary, top_indices):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    mask = np.ones(len(vocabulary), dtype=bool)
    for i, word in enumerate(vocabulary):
        if word in stopwords or len(word)<2 or len(word)>20:
            mask[i] = False
        elif i in top_indices[:max(1, len(vocabulary)//100)]:
            mask[i] = False
    return mask


def main():
    path = "/home/omakone3/Programming/Data.stat.840/week6/data/20news-bydate-train"

    dirs = get_directories(path)

    documents = get_files_from_dirs(dirs)

    cleand_documents = open_and_clean_documents(documents)
    lemmatized_documents = [lemmatize_document(d) for d in cleand_documents]

    all_words = [w for d in lemmatized_documents for w in d]
    vocab, inverse = np.unique(all_words, return_inverse=True)

    counts = np.bincount(inverse)
    top_indices = np.argsort(-counts)

    mask = prune_vocabulary(vocab, top_indices)
    pruned_vocab = vocab[mask]

    pruned_paragraphs = [[w for w in p if w in pruned_vocab] for p in lemmatized_documents]

    # Build the length-normalized frequency TF part and smoother logarithmic inverse document
    # frequency for the IDF part
    tfidf_vector = build_tf_idf(pruned_paragraphs, pruned_vocab)
    print(tfidf_vector)
    
if __name__ == "__main__":
    main()
