#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
from textblob import TextBlob
import os
import time
from langdetect import detect

consumer_key="ogi5obFBXqZZI04Ipdermt4xu"
consumer_secret="zuOj0sCi0oABTZ7olyyxG9RAVvCJZFmgR3zhyaFkNkvc8VHyIO"
access_token="939891962420846594-HluQ9x1saRkL1JpcYGArCEbnrsAggaq"
access_token_secret="BR2g2hTN6TBnijLaidGnmRcRKzRTLcGHq3F4SbwfVW1lD"
BITCOIN_PATH = os.path.abspath("/home/mica/PycharmProjects/data/bitcoin")
LITECOIN_PATH = os.path.abspath("/home/mica/PycharmProjects/data/litecoin")
ETHEREUM_PATH = os.path.abspath("/home/mica/PycharmProjects/data/ethereum")
RIPPLE_PATH = os.path.abspath("/home/mica/PycharmProjects/data/ripple")

def get_sentiment(text):

    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment

#This is a basic listener that just prints received tweets to stdout.
class myStreamListener(StreamListener):
    def on_status(self, status):
        dt = status.created_at
        y = dt.year
        mm = dt.month
        d = dt.day
        h = dt.hour
        m = dt.minute
        dt = str(y) + "-" + str(mm) + "-" + str(d) + "_" + str(h) + ':' + str(m)

        try:
            lang = detect(status.text)
            if lang == 'en':
                if "bitcoin" in status.text:
                    sentiment = get_sentiment(status.text)
                    name = "bitcoin" + dt
                    path = os.path.join(BITCOIN_PATH, name)
                    file = open(path, 'a')
                    file.write("NEW STATUS," + str(sentiment) + "," + str(status.created_at) + "," + status.text + "\n")
                    file.close()
                    print("NEW STATUS," + str(sentiment) + "," + str(status.created_at) + "," + status.text)

                if "litecoin" in status.text:
                    sentiment = get_sentiment(status.text)
                    name = "litecoin" + dt
                    path = os.path.join(LITECOIN_PATH, name)
                    file = open(path, 'a')
                    file.write("NEW STATUS," + str(sentiment) + "," + str(status.created_at) + "," + status.text + "\n")
                    file.close()
                    print("NEW STATUS," + str(sentiment) + "," + str(status.created_at) + "," + status.text)

                if "ethereum" in status.text:
                    sentiment = get_sentiment(status.text)
                    name = "ethereum" + dt
                    path = os.path.join(ETHEREUM_PATH, name)
                    file = open(path, 'a')
                    file.write("NEW STATUS," + str(sentiment) + "," + str(status.created_at) + "," + status.text + "\n")
                    file.close()
                    print("NEW STATUS," + str(sentiment) + "," + str(status.created_at) + "," + status.text)

                if "ripple" in status.text:
                    sentiment = get_sentiment(status.text)
                    name = "ripple" + dt
                    path = os.path.join(RIPPLE_PATH, name)
                    file = open(path, 'a')
                    file.write("NEW STATUS," + str(sentiment) + "," + str(status.created_at) + "," + status.text + "\n")
                    file.close()
                    print("NEW STATUS," + str(sentiment) + "," + str(status.created_at) + "," + status.text)
        except:
            print("Error")



if __name__ == '__main__':


    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, listener=myStreamListener())


    try:
        stream.filter(track=['bitcoin', 'litecoin', 'ripple', 'ethereum'], async=True)
    except:
        print ("error!")
        stream.disconnect()
