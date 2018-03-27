import datetime, csv, time, os, sys
if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

PATH = "data/tweets/100_inf.txt"
PATH_INF = "data/tweets/influencers"


def check_currencies(text):
    mentions = ""
    lower_text = text.lower()
    if ("bitcoin" or "btc") in lower_text:
        mentions += "bitcoin "
    if ("litecoin" or "ltc") in lower_text:
        mentions += "litecoin "
    if ("ethereum" or "etc") in lower_text:
        mentions += "ethereum "
    if ("ripple" or "xrp") in lower_text:
        mentions += "ripple "
    return mentions

def adapt_tweet(tweet):
    text = str(tweet.text.encode('utf-8'))
    mentions = check_currencies(text)
    s_tweet = []
    s_tweet.append(mentions)
    s_tweet.append(text)
    s_tweet.append(tweet.username)
    s_tweet.append(str(tweet.favorites))
    s_tweet.append(str(tweet.retweets))
    s_tweet.append(str(int(time.mktime(tweet.date.timetuple()))))
    s_tweet.append(tweet.id)
    s_tweet.append(str(tweet.permalink.encode('utf-8')))
    s_tweet.append(tweet.geo)
    s_tweet.append(str(tweet.hashtags))
    s_tweet.append(tweet.mentions)
    return s_tweet


def save_tweet(tweet, FILE_PATH):

    s_tweet = adapt_tweet(tweet)
    with open(FILE_PATH, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t')
        spamwriter.writerow(s_tweet)


def adapt_dates(start_timestamp):
    dt1 = datetime.datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
    return dt1


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def get_influencers(P):
    influencers = []
    with open(P) as f:
        lines = f.readlines()
        for line in lines:
            influencers.append(line.rstrip())
    return influencers


def main():
    influencers = get_influencers(PATH)
    for influencer in influencers:

        date1 = datetime.date(2016, 1, 1)
        date2 = datetime.date(2018, 1, 1)

        tweet_criteria = got.manager.TweetCriteria().setUsername(influencer).setSince(str(date1)).setUntil(str(date2))
        tweets = got.manager.TweetManager.getTweets(tweet_criteria)

        print (influencer)
        for i in range(len(tweets)):
            new_date = int(time.mktime(tweets[i].date.timetuple()))
            day = adapt_dates(new_date)
            FILE_PATH = os.path.join(PATH_INF, day + ".tsv")
            save_tweet(tweets[i], FILE_PATH)


if __name__ == '__main__':
    main()