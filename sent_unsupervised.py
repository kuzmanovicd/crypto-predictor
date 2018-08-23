import nltk, glob, csv, re
from vaderSentiment import vaderSentiment

lexicons = "data/sentiment_analysis/lexicons/*.tsv"
dataset = "data/sentiment_analysis/tweets/*.tsv"

ps = nltk.stem.PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))
vader_analyzer = vaderSentiment.SentimentIntensityAnalyzer()


def pretprocess(tweet):
    tweet = re.sub(r'@\w+', '', tweet)  # remove mentions
    tweet = re.sub(r'@(\s+)\w+', '', tweet)
    tweet = re.sub(r'http\S+', '', tweet)  # remove links
    tweet = re.sub(r'\w*\\.\w*', '', tweet)
    tweet = re.sub(r'/\w*', '', tweet)
    tweet = re.sub(r'([^\s\w]|_)+', '', tweet)  # only alfanumeric and space
    tweet = re.sub(r'\W*\b\w{18,60}\b', '', tweet)  # remove big words
    tokenize_tweet = nltk.word_tokenize(tweet)
    tweet = [word for word in tokenize_tweet if word not in stop_words]  # stop
    tweet = [ps.stem(word) for word in tweet]  # stem
    tweet = [word for word in tweet if len(word) > 2]  # small words
    return tweet


def check_sentiment_per_lexicon(words, lexicon_path):
    sent = 0
    with open(lexicon_path, 'r') as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter='\t')]
        for word in words:
            for row in rows:
                if row[0] == word:
                    if int(row[1]) == 1:
                        sent += 1
                    elif int(row[1]) == -1:
                        sent -= 1
    if sent<0:
        return -1
    elif sent>0:
        return 1
    else:
        return 0


def lex(tweet):
    words = nltk.word_tokenize(tweet)
    final_sent = 0
    for lex in sorted(glob.glob(lexicons)):
        final_sent += check_sentiment_per_lexicon(words, lex)

    if final_sent<0:
        return -1
    elif final_sent>0:
        return 1
    else:
        return 0


def vader(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  # remove links
    sentences = nltk.sent_tokenize(tweet)
    sent = 0
    for sentence in sentences:
        sent += vader_analyzer.polarity_scores(sentence)['compound']

    if len(sentences)!=0:
        sent /= len(sentences)
        if sent < 0:
            return '{0:.2f}'.format(abs(sent)), -1
        elif sent > 0:
            return '{0:.2f}'.format(sent), 1
        else:
            return 0, 0
    else:
        return 0,0


def tweets_processing(folder_path, lex_path):

    for path in sorted(glob.glob(folder_path)):
        tweets = []
        with open(path, 'r') as csvfile:
            rows = [row for row in csv.reader(csvfile, delimiter='\t')]

            for row in rows:
                if row[1] != '':
                    tweet = row[1].lower()
                    if ("btc" or "bitcoin" or "ltc" or "litecoin" or "xrp" or "ripple" or "etc" or "ethereum") in tweet:
                        sent1 = lex(tweet)
                        v_int, v_pol = vader(row[1])
                        row.extend([sent1, v_int, v_pol])
                        tweets.append(row)


        path = path.replace('tweets', 'predicted_tweets')
        print(path)

        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            for row in tweets:
                writer.writerow(row)
