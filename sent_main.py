import glob, csv, sys, os, re, json, nltk
import time, datetime
import numpy as np
import sent_cnn, sent_unsupervised

PATH = "data/sentiment_analysis/tweets/*.tsv"
test_date = "2018-06-01"
val_date = "2018-05-01"

stop_words = set(nltk.corpus.stopwords.words("english"))
ps = nltk.stem.PorterStemmer()


if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen


def get_dates_timestamp(test_date, val_date):
    test_timestamp = int(time.mktime(datetime.datetime.strptime(test_date, "%Y-%m-%d").timetuple()))
    val_timestamp = int(time.mktime(datetime.datetime.strptime(val_date, "%Y-%m-%d").timetuple()))
    return  test_timestamp, val_timestamp


def get_data(FILE_PATH):
    all_data = []
    num_of_days = len(glob.glob(FILE_PATH))
    for path in sorted(glob.glob(FILE_PATH)):
        with open(path, 'r') as f:
            all_data.extend([row for row in csv.reader(f, delimiter='\t') if row[0] != ""])
    return all_data, num_of_days


def pretprocess_tweet(tweet):
    text = tweet.lower()
    text = re.sub(r'@\w+', '', text)  # remove mentions
    text = re.sub(r'@(\s+)\w+', '', text)
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub(r'\w*\\.\w*', '', text)
    text = re.sub(r'/\w*', '', text)
    text = re.sub(r'([^\s\w]|_)+', '', text)  # only alfanumeric and space
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'\W*\b\w{18,60}\b', '', text)  # remove big words
    text_tokenize = nltk.word_tokenize(text)
    text = [word for word in text_tokenize if word not in stop_words]
    text = [ps.stem(word) for word in text]
    text = [word for word in text if len(word) > 2]
    text = ' '.join(word for word in text)
    return text


def prepare_text(data):                 #prepare text for text processing

    pp_data = []
    i = 0
    for row in data:
        text = pretprocess_tweet(row[1])
        data[i][1] = text
        data[i][5] = int(data[i][5])
        if text!='':
            pp_data.append(data[i])
        i += 1

    return pp_data


def check_num(FILE_PATH, terminal_path):
    i = 0
    for path in sorted(glob.glob(FILE_PATH)):
        if terminal_path not in path:
            i +=1
        else:
            break

    return i


def save_results(terminal_path, results, FILE_PATH, mode):

    i = 0
    path_num = check_num(FILE_PATH, terminal_path)
    for path in sorted(glob.glob(FILE_PATH)):
        if path_num<=0 and mode == "train":
            break
        elif path_num>0 and mode == "test":
            path_num -= 1
            continue

        with open(path, 'r') as f:
            all_data = []
            data = [row for row in csv.reader(f, delimiter='\t') if row[0]!=""]
            for d in data:
                tweet = pretprocess_tweet(d[1])
                if tweet != "" and i<len(results):
                    lex = sent_unsupervised.lex(tweet)
                    v_int, v_pol = sent_unsupervised.vader(d[1])
                    d.extend([lex, v_int, v_pol])
                    d.extend(results[i])
                    all_data = np.append(all_data, d).reshape(-1, 16)
                    i += 1

        path_num -= 1

        path2 = path.replace('tweets', 'tweets_predicted')

        with open(path2, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['currency', 'tweet', 'username', 'favorites', 'retweets', 'date', 'id', 'permalink', 'geo',
                             'hashtags', 'mentions', 'lex_sent', 'vad_sent', 'vad_pol', 'cnn_sent', 'cnn_pol'])
            for d in all_data:
                writer.writerow(d)


def save_test_results(terminal_path, results, FILE_PATH):

    i = 0
    for pred in predicted:
        if pred[1]>0.5:
            predicted[i][1] = 1
        else:
            predicted[i][1] = -1
        i += 1

    save_results(terminal_path, results, FILE_PATH, "test")


if __name__ == '__main__':
    all_data, num_of_days = get_data(PATH)
    test_timestamp, val_timestamp = get_dates_timestamp(test_date, val_date)

    all_data_pp = prepare_text(all_data)

    train_data = [row for row in all_data_pp if row[5] < val_timestamp]
    val_data = [row for row in all_data_pp if (row[5] > val_timestamp and row[5] < test_timestamp)]
    test_data = [row for row in all_data_pp if row[5] > test_timestamp]

    train_labels = sent_cnn.get_labels(train_data)
    val_labels = sent_cnn.get_labels(val_data)
    save_labels = np.vstack([train_labels, val_labels])
    save_results(test_date, save_labels, PATH, "train")

    predicted = sent_cnn.cnn_process(train_data, val_data, test_data, train_labels, val_labels)

    save_test_results(test_date, predicted, PATH)
