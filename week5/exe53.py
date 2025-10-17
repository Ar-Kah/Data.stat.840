#!/usr/bin/env python3

from nltk.lm import MLE, preprocessing
import nltk
import requests
import nltk


def cleantexts(texts):
    cleaned_texts = []
    for text in texts:
        split_text = text.splitlines()
        clean_words = [w.strip('[]_*(){}').lower() for w in split_text]
        clean_text = '\n'.join(clean_words)
        cleaned_texts.append(clean_text)

    return cleaned_texts


def cuttexts(texts):
    book_contents = []
    for text in texts:
        split_lines = text.splitlines()
        start_index = 0
        end_index = len(split_lines)
        for idx, line in enumerate(split_lines):
            if "START OF THE PROJECT GUTENBERG EBOOK" in line:
                start_index = idx + 1
            elif "END OF THE PROJECT GUTENBERG EBOOK" in line:
                end_index = idx
                break
        
        body = '\n'.join(split_lines[start_index: end_index])
        # Strip empty lines of text
        body = [line.strip() for line in body.splitlines() if line.strip()]
        body = '\n'.join(body)
        book_contents.append(body)

    return book_contents




def generate_ngram_themoon(book_tokenized_text):
    x = [2, 3, 5]
    for n in x:
        nltk_ngramtraining_data, nltk_padded_sentences = preprocessing.padded_everygram_pipeline(n, book_tokenized_text)
        nltk_ngrammodel = MLE(n)
        nltk_ngrammodel.fit(nltk_ngramtraining_data, nltk_padded_sentences)
        print(f"paragraphs with n: {n}")
        print(' '.join(nltk_ngrammodel.generate(num_words=20, text_seed=['the', 'moon'])))


def generate_ngram(book_tokenized_text):
    x = [1, 2, 3, 5]
    for n in x:
        nltk_ngramtraining_data, nltk_padded_sentences = preprocessing.padded_everygram_pipeline(n, book_tokenized_text)
        nltk_ngrammodel = MLE(n)
        nltk_ngrammodel.fit(nltk_ngramtraining_data, nltk_padded_sentences)
        print(f"paragraphs with n: {n}")
        print(' '.join(nltk_ngrammodel.generate(num_words=20, text_seed=['the'])))
        print(' '.join(nltk_ngrammodel.generate(num_words=20, text_seed=['the'])))
        

def main():

    # Raw text of Robin Hood
    rh_url = "https://www.gutenberg.org/cache/epub/10148/pg10148.txt"
    rh_r = requests.get(rh_url)
    rh_raw_text = rh_r.text

    # Raw text of A Martian Odyssey
    mo_url = "https://www.gutenberg.org/cache/epub/23731/pg23731.txt"
    mo_r = requests.get(mo_url)
    mo_raw_text = mo_r.text

    raw_texts = [rh_raw_text, mo_raw_text]
    # Cut book contents from the lines START OF THE
    # PROJECT... to END OF THE PROJECT...
    cut_texts = cuttexts(raw_texts)
    # print(cut_texts[0][0: 100]) 

    clean_texts = cleantexts(cut_texts)
    # print(clean_texts[0][0:500])

    rh_tokenized_text = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(clean_texts[0])]
    mo_tokenized_text = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(clean_texts[1])]

    print("Robin hood")
    generate_ngram(rh_tokenized_text)
    print("------------------")
    print()
    print("A Martina Odyssey")
    generate_ngram(mo_tokenized_text)
    print("------------------")
    print()
    # end of (c) in exercise 5.3

    generate_ngram_themoon(rh_tokenized_text)
    generate_ngram_themoon(mo_tokenized_text)


if __name__ == '__main__':
    main()

"""
comments on (c)
output of seed word 'the':

Robin hood
paragraphs with n: 1
, from give holy and no as `` but saints very sir . they life deer of till to grass
tired a met beggar he ran dost he looked that a the while jolly went was . though , split
paragraphs with n: 2
tanner could gather his doings among the other , with a pot . </s> `` here the lass , or
king , but will be silent in tones : '' said sir richard 's flank with hills and smote him
paragraphs with n: 3
glade that lay hidden so near that they bow down ; but fat abbot , knight , or the other
world would i draw a string betwixt the fingers . </s> </s> </s> </s> </s> </s> </s> </s> </s> </s>
paragraphs with n: 5
dawn the dew was always the brightest , and the song of the deserted shepherdess ? '' </s> </s> </s>
beggar had gone . </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s>
------------------

A Martina Odyssey
paragraphs with n: 1
`` inclusive . dealing he rabbit 'd pointed was cliffs cliff i aim floor straps then the , me of
leafless , if nuts time feathery hundred and keep , '' soup `` left sand dumping tried and ! with
paragraphs with n: 2
barrels playing train , just at the air hit my words of the center of the floor from then was
whole chorus of gas that queer ! </s> of those block-long nose ! </s> '' </s> if he would .
paragraphs with n: 3
creatures were intelligent . </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s>
creatures ' development . </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s>
paragraphs with n: 5
pyramid creature and plowed along through xanthus . </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s>
planet , or a dozen others . </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s>
------------------


Comments:
After running and reading the output a few times it is clear that the bigger the n is
the more structure the text has and the more sence it makes. Atleats with n=5 the ngram
starts to remember paragraphs. You can find the exact paragraphs in the book.
"""

"""
Comments on (d)

output:
paragraphs with n: 2
arose and i say , scratching in thought to himself ! </s> forth to merry withal , with a harp
paragraphs with n: 3
was full and the strong waters mixed with the dust from off thy bones , though he meant it with
paragraphs with n: 5
was full and the dun deer in season ; so that the king 's archers were fairly beaten , and
paragraphs with n: 2
, or trying to my point that understanding was something , then i got weak man -- '' </s> </s>
paragraphs with n: 3
is visible from here on i 'll wager ! </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s>
paragraphs with n: 5
is visible from here as a fifth magnitude star . </s> </s> </s> </s> </s> </s> </s> </s> </s> </s>


I could not tell in which book the paragraphs where generated if I didn't myself know the order.
"""
