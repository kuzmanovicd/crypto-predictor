import tweepy
import requests
from langdetect import detect
import datetime
import time
from textblob import TextBlob

consumer_key="ogi5obFBXqZZI04Ipdermt4xu"
consumer_secret="zuOj0sCi0oABTZ7olyyxG9RAVvCJZFmgR3zhyaFkNkvc8VHyIO"
access_token="939891962420846594-HluQ9x1saRkL1JpcYGArCEbnrsAggaq"
access_token_secret="BR2g2hTN6TBnijLaidGnmRcRKzRTLcGHq3F4SbwfVW1lD"
text_processing_url = 'http://text-processing.com/api/sentiment/'


def api_service():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api


def adapt_dates(start_timestamp, end_timestamp):
    dt1 = datetime.datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
    dt = datetime.datetime.fromtimestamp(end_timestamp)
    y = dt.year
    m = dt.month
    d = dt.day+1
    dt2 = datetime.datetime(year=y, month=m, day=d).strftime('%Y-%m-%d')

    return dt1, dt2


def get_text_api_sentiment(text):

    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment


def get_sentiment(text, algorithm):
    if algorithm=="textprocessing.org":
        return get_text_api_sentiment(text)


def get_tweets_sentiment_per_hour(start_timestamp, end_timestamp, currency):
    api = api_service()
    dt1, dt2 = adapt_dates(start_timestamp, end_timestamp)
    sentiment = {'neg': 0, 'pos': 0, 'neutral': 0}
    i = 0
    for status in tweepy.Cursor(api.search, q=currency, lang='en', since=dt1, until=dt2).items():
        linux_timestamp = int(time.mktime(status.created_at.timetuple()))

        if start_timestamp < linux_timestamp < end_timestamp:
            probability = get_sentiment(status.text, "textprocessing.org")
            sentiment['neg'] = sentiment['neg'] + probability['neg']
            sentiment['pos'] = sentiment['pos'] + probability['pos']
            sentiment['neutral'] = sentiment['neutral'] + probability['neutral']
            i += 1

    if i != 0:
        sentiment['neg'] = sentiment['neg']/i
        sentiment['pos'] = sentiment['pos']/i
        sentiment['neutral'] = sentiment['neutral']/i
    return sentiment


def get_tweets_sentiment_per_period(start_timestamp, end_timestamp, currency):

    api = api_service()
    dt1 = datetime.datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
    dt2 = datetime.datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d')
    sentiment = 0
    i = 0
    temp_timestamp = end_timestamp-3600
    statuss = tweepy.Cursor(api.search, q=currency, lang='en', since=dt1, until=dt2).items()

    while True:
        try:
            status = statuss.next()
            linux_timestamp = int(time.mktime(status.created_at.timetuple()))

            if linux_timestamp < temp_timestamp:
                if i != 0:
                    sentiment = sentiment / i
                dt2 = datetime.datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M')
                dt1 = datetime.datetime.fromtimestamp(temp_timestamp).strftime('%Y-%m-%d %H:%M')
                end_timestamp = temp_timestamp
                temp_timestamp -= 3600
                file = open(currency+'_sssentiment.txt', 'a')
                file.write(dt1 + "," + dt2 + "," + str(sentiment) + "," + str(i) + "\n")
                file.close()
                print(dt1 + " , " + dt2 + " , " + str(sentiment) + "-----------ISPIS\n")
                sentiment = 0
                i = 0

            probability = get_sentiment(status.text, "textprocessing.org")
            file = open(currency + str(dt1)+'_ssstatus.txt', 'a')
            file.write("START_OF_NEW_STATUS," + str(status.created_at) + "," + status.text + "\n")
            file.close()
            sentiment += probability
            i += 1
            print(str(status.created_at) + "-----------\n")
        except tweepy.TweepError:
            print("Sleep untill new call")
            time.sleep(60 * 15)
            continue
        except StopIteration:
            break


def get_tweets(start_timestamp, end_timestamp, currency):
    api = api_service()
    dt1 = datetime.datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
    dt2 = datetime.datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d')
    pages = tweepy.Cursor(api.search, q=currency, lang='en', since=dt1, until=dt2).pages()
    while True:
        try:
            tweets = pages.next()
            for tweet in tweets:
                file = open(currency + '_statuses.txt', 'a')
                file.write("START_OF_NEW_STATUS," + str(tweet.created_at) + "," + tweet.text + "\n")
                file.close()
        except tweepy.TweepError:
            print("Sleep untill new call")
            time.sleep(60 * 15)
            continue
        except StopIteration:
            break

#sentiment_hour = get_tweets_sentiment_per_hour(1513364400, 1513368000, 'tensorflow')

#get_tweets_sentiment_per_period(1513983600, 1514070000, 'ripple')
get_tweets_sentiment_per_period(1513983600, 1514070000, 'litecoin')
get_tweets_sentiment_per_period(1513983600, 1514070000, 'ethereum')
get_tweets_sentiment_per_period(1514070000, 1514156400, 'bitcoin')
get_tweets_sentiment_per_period(1513983600, 1514070000, 'bitcoin')
#get_tweets(1513206000, 1513810800, 'ethereum')