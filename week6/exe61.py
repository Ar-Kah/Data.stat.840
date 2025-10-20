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
            cleaned_words = [w.strip('"-<>[]_*(){}.,:;?!\'“”').lower() for w in words]

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

def build_tf_idf(documents, vocabulary):
    word_to_idx = {w:i for i, w in enumerate(vocabulary)}
    n_docs = len(documents)
    n_vocab = len(vocabulary)
    tf_matrix = scipy.sparse.lil_matrix((n_docs, n_vocab))
    
    for doc_id, doc in enumerate(documents):
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


def print_results(N, UrightT, remainingvocabulary, highesttotals):

    for x in range(N):
        topweights_indices=np.argsort(-1*np.abs(UrightT[x,:]))
        print(remainingvocabulary[highesttotals[topweights_indices[0:10]]])

def main():
    path = "/home/omakone3/Programming/Data.stat.840/week6/data/20news-bydate-train"

    dirs = get_directories(path)

    documents = get_files_from_dirs(dirs)

    cleaned_documents = open_and_clean_documents(documents)
    lemmatized_documents = [lemmatize_document(d) for d in cleaned_documents]

    all_words = [w for d in lemmatized_documents for w in d]
    vocab, inverse = np.unique(all_words, return_inverse=True)

    counts = np.bincount(inverse)
    top_indices = np.argsort(-counts)

    mask = prune_vocabulary(vocab, top_indices)
    pruned_vocab = vocab[mask]

    pruned_documents = [[w for w in p if w in pruned_vocab] for p in lemmatized_documents]
    all_remaining_words = [w for d in pruned_documents for w in d]
    remainingvocabulary = np.unique(all_remaining_words)


    # Build the length-normalized frequency TF part and smoother logarithmic inverse document
    # frequency for the IDF part
    tfidf_matrix = build_tf_idf(pruned_documents, pruned_vocab)
    
    dimensiontotals=np.squeeze(np.array(np.sum(tfidf_matrix,axis=0)))
    highesttotals=np.argsort(-1*dimensiontotals)
    Xsmall=tfidf_matrix[:,highesttotals[0:500]].todense()
    # Compute 10 factors from LSA
    n_low_dimensions=15
    Uleft,D,UrightT=scipy.sparse.linalg.svds(Xsmall,k=n_low_dimensions)


    print_results(n_low_dimensions, UrightT, remainingvocabulary, highesttotals)


if __name__ == "__main__":
    main()

"""
exe6.1 (e)
output:
['stats' 'blah' 'individual' 'al' 'cub' 'tie' 'breaker' 'e-mail' 'record' 'advance']
['blah' 'stats' 'group' 'individual' "let's" 'al' 'advance' 'e-mail' 'subject' 'cub']
['jewish' 'kingman' 'stats' 'dave' 'cub' 'advance' 'e-mail' 'subject' 'email' 'address']
['jewish' 'kingman' 'dave' 'cub' 'stats' 'dolven' 'advance' 'e-mail' 'subject' 'wonder']
['dolven' 'doug' 'hall' 'yankee' 'tom' 'sign' 'jewish' 'e-mail' 'appreciate' 'stats']
['state/maine' 'finals...who' 'lake' 'maine' 'roster' 'hall' 'ryan' 'college' 'luck' 'reply']
['luck' 'jim' 'pitching' 'version' 'ticket' 'tin' 'x-newsreader' 'advantage' 'staff' 'address']
['test' 'hello' 'message' 'become' 'ticket' 'push' 'wonder' 'different' 'dealer' 'comment']
['appreciate' 'advance' 'captain' 'park' 'wonder' 'dan' 'kind' 'series' 'job' 'news']
['advantage' 'hawk' 'luck' 'detroit' 'alomar' 'dog' 'beat' 'brian' 'round' 'champ']

Some of the outputs seem to be talking about baseball atleast. But its not clear atleast from
these word groups to really easily tell what the resulting factors are about.
"""

"""
['sam' 'vesterman' 'nl' 'tin' 'x-newsreader' '1.1' 'charles' 'al' 'e-mail' 'version']
['hello' 'message' 'o&o' 'tie' 'breaker' 'group' 'record' 'tin' 'x-newsreader' 'blah']
['o&o' 'tie' 'breaker' 'record' 'cub' 'stats' 'tin' 'x-newsreader' '1.1' 'individual']
['o&o' 'hello' 'tie' 'breaker' 'cub' 'record' 'advance' 'stats' 'tin' 'x-newsreader']
['cub' 'advance' 'tie' 'blah' 'breaker' 'email' 'record' 'subject' 'stats' 'roster']
['stats' 'blah' 'individual' 'al' 'cub' 'tie' 'breaker' 'e-mail' 'record' 'advance']
['blah' 'stats' 'group' 'individual' "let's" 'al' 'advance' 'e-mail' 'subject' 'cub']
['jewish' 'kingman' 'stats' 'dave' 'cub' 'advance' 'e-mail' 'subject' 'email' 'address']
['jewish' 'kingman' 'dave' 'cub' 'stats' 'dolven' 'advance' 'e-mail' 'subject' 'wonder']
['dolven' 'doug' 'hall' 'yankee' 'tom' 'sign' 'jewish' 'e-mail' 'appreciate' 'stats']
['appreciate' 'advance' 'captain' 'park' 'wonder' 'dan' 'kind' 'series' 'job' 'news']
['advantage' 'hawk' 'luck' 'detroit' 'alomar' 'dog' 'beat' 'brian' 'round' 'champ']
['state/maine' 'finals...who' 'lake' 'maine' 'roster' 'hall' 'ryan' 'college' 'luck' 'reply']
['luck' 'jim' 'pitching' 'version' 'ticket' 'tin' 'x-newsreader' 'advantage' 'staff' 'address']
['test' 'hello' 'message' 'become' 'ticket' 'push' 'wonder' 'different' 'dealer' 'comment']

Atleast the outputs didn't improve. There is a lot of stuff about emails for some reason.
"""
