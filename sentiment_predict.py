import glob, csv, os, re
from sentiment_lexicon import count_sentiment
from nltk import tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import datetime
PATH = "data/tweets/influencers/*.tsv"
PATH_SAVE = "data/tweets/predicted"

def vader_paragraph(paragraph, analyzer):
    text = paragraph
    text = re.sub(r'http\S+', '', text)
    sentence_list = tokenize.sent_tokenize(text)
    paragraphSentiments=0.0
    for sentence in sentence_list:
        vs = analyzer.polarity_scores(sentence)

        paragraphSentiments += vs["compound"]

    if len(sentence_list)==0:
        return str(0)
    else:
        return str(round(paragraphSentiments / len(sentence_list), 4))

def predict_sentiment_per_file_vader(data):
    all_data = []
    analyzer = SentimentIntensityAnalyzer()
    for d in data:
        sent = vader_paragraph(d[1], analyzer)
        d.append(sent)
        all_data.append(d)
    return all_data

def predict_sentiment_per_file(data):
    all_data = []
    analyzer = SentimentIntensityAnalyzer()
    for d in data:
        text = d[1].lower()
        if d[0] != "" or ("crypto" or "block" or "btc" or "ltc" or "etc" or "xrp" or "currency" or "miner") in text:
            sent = count_sentiment(d[1])
            d.append(sent)
            sent = vader_paragraph(d[1], analyzer)
            d.append(sent)
            sent = 0    #TO DO - cnn_sent
            d.append(sent)
            #d[5] = int(time.mktime(datetime.datetime.strptime(d[5], "%Y-%m-%d %H:%M:%S").timetuple()))
            all_data.append(d)
    return all_data

def take_path():
    for path in glob.glob(PATH):
        with open(path, 'r') as f:
            data = [row for row in csv.reader(f.read().splitlines(), delimiter='\t')]
            all_data = predict_sentiment_per_file(data)
            dirname = os.path.basename(path)
            new_path = os.path.join(PATH_SAVE, dirname)
            print (new_path)
            with open(new_path, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow(['currency', 'tweet', 'username', 'favorites', 'retweets', 'date', 'id',
                                 'permalink', 'geo', 'hashtags', 'mentions', 'lex_sent', 'vader_sent', 'cnn_sent'])

                for d in all_data:
                    writer.writerow(d)

take_path()
