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

def plsa(document_to_word_matrix, n_topics, n_iterations):
    n_docs=np.shape(document_to_word_matrix)[0] # Number of documents and vocabulary words
    n_vocab=np.shape(document_to_word_matrix)[1]
    theta = scipy.stats.uniform.rvs(size=(n_vocab,n_topics)) # Prob of words per topic: random init
    theta = theta/np.matlib.repmat(np.sum(theta,axis=0),n_vocab,1)
    psi = scipy.stats.uniform.rvs(size=(n_topics,n_docs)) # Probs topics per document: random init
    psi = psi/np.matlib.repmat(np.sum(psi,axis=0),n_topics,1)
    n_words_in_docs = np.squeeze(np.array(np.sum(document_to_word_matrix,axis=1)))
    n_totalwords = np.sum(n_words_in_docs) # Total number of words: computed once
    pi = n_words_in_docs/n_totalwords # Document probs: computed once
    for myiter in range(n_iterations): # Perform Expectation-Maximization iterations
        # ===Perform E-step====
        doc_word_to_topics = [] # Compute theta_{v|t}psi_{t|d}/sum_t' theta_{v|t'}psi_{t'|d}
        doc_word_to_topic_sum = np.zeros((n_docs,n_vocab))
        for t in range(n_topics):
            doc_word_to_topict = \
                np.matlib.repmat(theta[:,t],n_docs,1) * \
                np.matlib.repmat(psi[t,:],n_vocab,1).T
            myepsilon=1e-14 # Add a positive number to avoid divisions by zero
            doc_word_to_topict += myepsilon
            doc_word_to_topics.append(doc_word_to_topict)
            doc_word_to_topic_sum += doc_word_to_topict
        for t in range(n_topics):
            doc_word_to_topics[t] /= doc_word_to_topic_sum
        # =======Perform M-step=======
        # Add a small number to word counts to avoid divisions by zero
        for t in range(n_topics): # Compute document-to-topic probabilities.
            psi[t,:] = np.squeeze(np.array(np.sum( \
                np.multiply(document_to_word_matrix+myepsilon,doc_word_to_topics[t]),axis=1)))
        psi /= np.matlib.repmat(np.sum(psi,axis=0),n_topics,1)
        for t in range(n_topics): # Compute topic-to-word probabilities
            theta[:,t]= np.squeeze(np.array(np.sum( \
                np.multiply(document_to_word_matrix,doc_word_to_topics[t]),axis=0).T))
        theta /= np.matlib.repmat(np.sum(theta,axis=0),n_vocab,1)
    return(pi,psi,theta)


def print_results(N, theta, remainingvocabulary, highesttotals):

    for x in range(N):
        topweights_indices=np.argsort(-1*np.abs(theta[:,x]))
        print(remainingvocabulary[highesttotals[topweights_indices[0:20]]])

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

    # Run PLSA
    n_topics=10
    n_iterations=200
    pi,psi,theta=plsa(Xsmall, n_topics, n_iterations)

    print_results(10, theta, remainingvocabulary, highesttotals)
if __name__ == "__main__":
    main()

    """
['jewish' 'group' 'sox' 'advantage' 'expect' 'mind' 'kid' 'seat' 'white'
 'air' 'gear' 'paint' 'mets' "what's" 'auto' 'bring' 'exactly' 'tom'
 'originator' 'kevin']
['oil' 'local' 'advance' 'michael' 'park' 'tiger' 'al' 'net' 'coverage'
 'pitching' 'tv' 'vesterman' 'york' 'wonder' 'key' 'abc' 'puck' 'penalty'
 'cop' 'paul']
['x-newsreader' 'tin' 'version' '1.1' 'pen' 'phillies' 'montreal' 'april'
 'draft' 'gerald' 'philadelphia' 'flyer' 'defense' 'wagon' 'round'
 'anybody' 'winner' 'news' 'jeff' 'deal']
['cub' 'appreciate' 'wave' 'rider' 'left' 'answer' 'foot' 'standard'
 'disclaimer' 'law' 'center' 'stick' 'person' 'steve' 'room' 'kill'
 'street' 'window' 'later' 'learn']
['bmw' "let's" 'previous' 'rear' 'design' 'light' 'alan' 'vote' 'dodger'
 'finish' 'level' 'keith' 'effect' 'handle' 'convertible' 'mask'
 'straight' 'cannot' 'talent' 'apr']
['luck' 'honda' 'model' 'experience' 'guess' 'idea' 'check' 'else' 'stuff'
 'jim' 'rid' 'richard' 'fact' 'european' 'ok' 'canadian' 'north' 'saw'
 'type' 'fine']
['test' 'stats' 'insurance' 'money' 'hawk' 'company' 'devil' 'message'
 'jet' 'penguin' 'hello' 'subject' 'islander' 'save' 'manager'
 'individual' 'jack' 'gary' 'box' '91']
['e-mail' 'mail' 'dealer' 'interested' 'owner' 'mile' 'helmet' 'area'
 'cost' 'address' 'email' 'andrew' 'figure' 'info' 'kind' 'hole' 'ticket'
 'cover' 'sale' 'saturn']
['dave' 'record' 'dog' 'tie' 'leaf' 'detroit' 'david' 'captain' 'espn'
 'ice' 'throw' 'stupid' 'field' 'ottawa' 'eric' 'roger' 'ryan' 'lock'
 'lemieux' 'breaker']
['sound' 'brave' 'nice' 'yankee' 'ball' 'smith' 'doug' 'hitter' 'pitcher'
 'hall' 'somebody' 'information' 'jay' 'alomar' 'lake' 'dolven' 'bat'
 'defensive' 'aaa' 'ron']
    """
