import csv,re,os
import glob
PATH = "data/tweets/lexicons/*.tsv"

def check_sentiment_word(word, PATH):
    with open(PATH, 'r') as f:
        data = [row for row in csv.reader(f.read().splitlines(), delimiter='\t')]
        for i in range(0, len(data)):
            if data[i][0] == word.lower():
                if float(data[i][1]) < 0:
                    return -1
                elif float(data[i][1]) > 0:
                    return 1
                else:
                    return 0

    return 0

def all_word_sentiment(word):
    all_sent = []
    for filename in glob.glob(PATH):
        sent = check_sentiment_word(word, filename)
        all_sent.append(sent)

    return all_sent

def count_sentiment_per_word(word):
    all_sent = all_word_sentiment(word)
    sum = float(all_sent[0])
    for i in range(0, len(all_sent)):
        sum += float(all_sent[i])

    return sum


def count_sentiment_per_sentence(sentence):
    finall_sent = 0
    listt = []
    for word in sentence:
        kk = count_sentiment_per_word(word)
        listt.append(kk)
        finall_sent += kk

    return "{0:.2f}".format(finall_sent/float(len(sentence)))


def count_sentiment(sent):
    e_sent = re.sub(r'[^\w\s]', '', sent)
    words = e_sent.split(" ")
    return count_sentiment_per_sentence(words)

