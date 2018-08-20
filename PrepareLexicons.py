import csv, re, nltk

dataset1 = 'data/tweets/lexicon_dataset/bitcointweets.csv'
dataset2 = 'data/tweets/lexicon_dataset/sent140_1.csv'
dataset3 = 'data/tweets/lexicon_dataset/sent140_2.csv'
dataset4 = 'data/tweets/lexicon_dataset/twitter2.csv'
dataset5 = 'data/tweets/lexicon_dataset/twitter4242.txt'
dataset6 = 'data/tweets/lexicon_dataset/senticnet.tsv'


stop_words = set(nltk.corpus.stopwords.words('english'))
ps = nltk.stem.PorterStemmer()


def pretprocess_dataset1(path):
    pp_dataset = []
    with open(path, 'r') as csvfile:
        rows = [row for row in csv.reader(csvfile)]
        if all([len(row)==8 for row in rows]):
            print('svi tvitovi imaju isti broj clanova')

            for row in rows:
                tweet = row[1].lower()
                tweet = re.sub(r'@\w+', '', tweet)  # remove mentions
                tweet = re.sub(r'@(\s+)\w+', '', tweet)
                tweet = re.sub(r'http\S+', '', tweet)  # remove links
                tweet = re.sub(r'\w*\\.\w*', '', tweet)
                tweet = re.sub(r'/\w*', '', tweet)
                tweet = re.sub(r'([^\s\w]|_)+', '', tweet)  # only alfanumeric and space
                tweet = re.sub(r'[0-9]+', '', tweet)
                tweet = re.sub(r'\W*\b\w{18,60}\b', '', tweet)  # remove big words
                tokenize_tweet = nltk.word_tokenize(tweet)
                tweet = [word for word in tokenize_tweet if word not in stop_words] #remove stop words
                tweet = [ps.stem(word) for word in tweet]
                tweet = [word for word in tweet if len(word)>2] #remove small words
                tweet = ' '.join(word for word in tweet)   #make sentence again
                if tweet != '':
                    if 'positive' in row[7]:
                        label = 1
                    elif 'negative' in row[7]:
                        label = -1
                    else:
                        label = 0

                    pp_dataset.append([tweet, label])

        else:
            print('neki tvit nema 8 elemenata')

    path = path.replace('.csv', '_new.tsv')

    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for row in pp_dataset:
            writer.writerow(row)


def pretprocess_dataset23(path):
    pp_dataset = []
    with open(path, 'r', encoding="ISO-8859-1") as csvfile:
        rows = [row for row in csv.reader(csvfile)]
        if all(len(row)==6 for row in rows):
            print('svi tvitovi su iste duzine')
        else:
            print('tvitovi su razlicite duzine')
        for row in rows:
            if len(row)==6:
                tweet = row[5].lower()
                tweet = re.sub(r'@\w+', '', tweet)  # remove mentions
                tweet = re.sub(r'@(\s+)\w+', '', tweet)
                tweet = re.sub(r'http\S+', '', tweet)  # remove links
                tweet = re.sub(r'\w*\\.\w*', '', tweet)
                tweet = re.sub(r'/\w*', '', tweet)
                tweet = re.sub(r'([^\s\w]|_)+', '', tweet)  # only alfanumeric and space
                tweet = re.sub(r'[0-9]+', '', tweet)
                tweet = re.sub(r'\W*\b\w{18,60}\b', '', tweet)  # remove big words
                tokenize_tweet = nltk.word_tokenize(tweet)
                tweet = [word for word in tokenize_tweet if word not in stop_words] #stop
                tweet = [ps.stem(word) for word in tweet]  #stem
                tweet = [word for word in tweet if len(word)>2] #small words
                tweet = ' '.join(word for word in tweet) #make sentence

                if tweet != '':
                    if int(row[0])==4:
                        label = 1
                    elif int(row[0])==0:
                        label = -1
                    else:
                        label = 0
                    pp_dataset.append([tweet, label])



    path = path.replace('.csv', '_new.tsv')

    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for row in pp_dataset:
            writer.writerow(row)


def pretprocess_dataset4(path):
    pp_dataset = []
    with open(path, 'r') as csvfile:
        rows = [row for row in csv.reader(csvfile)]
        if all([len(row)==17 for row in rows]):
            print ('svi tvitovi imaju jednak broj clanova')
            i = 0
            for row in rows:
                if i<2:
                    i += 1
                    continue
                tweet = row[4].lower()
                tweet = re.sub(r'@\w+', '', tweet)  # remove mentions
                tweet = re.sub(r'@(\s+)\w+', '', tweet)
                tweet = re.sub(r'http\S+', '', tweet)  # remove links
                tweet = re.sub(r'\w*\\.\w*', '', tweet)
                tweet = re.sub(r'/\w*', '', tweet)
                tweet = re.sub(r'([^\s\w]|_)+', '', tweet)  # only alfanumeric and space
                tweet = re.sub(r'[0-9]+', '', tweet)
                tweet = re.sub(r'\W*\b\w{18,60}\b', '', tweet)  # remove big words
                tokenize_tweet = nltk.word_tokenize(tweet)
                tweet = [word for word in tokenize_tweet if word not in stop_words] #stop word
                tweet = [ps.stem(word) for word in tweet] #stem
                tweet = [word for word in tweet if len(word)>2] #small word
                tweet =' '.join(word for word in tweet) #make sentence
                if tweet != '':
                    if row[0] == 'POS':
                        label = 1
                    elif row[0] == 'NEG':
                        label = -1
                    else:
                        label = 0

                    pp_dataset.append([tweet, label])

        else:
            print ('tvitovi imaju razlicit broj clanova')

    path = path.replace('.csv', '_new.tsv')

    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter ='\t')
        for row in pp_dataset:
            writer.writerow(row)


def pretprocess_dataset5(path):
    pp_dataset = []
    with open(path, 'r', encoding="ISO-8859-1") as file:
        rows = file.read().splitlines()
        i = 0
        for row in rows:
            if i<1:
                i += 1
                continue

            pos, neg, tweet = row.split('\t')
            tweet = tweet.lower()
            tweet = re.sub(r'@\w+', '', tweet)  # remove mentions
            tweet = re.sub(r'@(\s+)\w+', '', tweet)
            tweet = re.sub(r'http\S+', '', tweet)  # remove links
            tweet = re.sub(r'\w*\\.\w*', '', tweet)
            tweet = re.sub(r'/\w*', '', tweet)
            tweet = re.sub(r'([^\s\w]|_)+', '', tweet)  # only alfanumeric and space
            tweet = re.sub(r'[0-9]+', '', tweet)
            tweet = re.sub(r'\W*\b\w{18,60}\b', '', tweet)  # remove big words
            tokenize_tweet = nltk.word_tokenize(tweet)
            tweet = [word for word in tokenize_tweet if word not in stop_words]
            tweet = [ps.stem(word) for word in tweet]
            tweet = [word for word in tweet if len(word)>2]
            tweet = ' '.join(word for word in tweet)

            if tweet != '':
                pos = float(pos)
                neg = float(neg)
                if abs(neg-pos) <= 1.0:
                    label = 0
                elif pos/neg>1.5:
                    label = 1
                else:
                    label = -1

                pp_dataset.append([tweet, label])


    path = path.replace('.txt', '_new.tsv')

    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for row in pp_dataset:
            writer.writerow(row)


def pretprocess_dataset6(path):
    words = []
    with open(path, 'r') as csvfile:
        rows = [row for row in csv.reader(csvfile, delimiter='\t')]
        for row in rows:
            if '_' not in row[0] and row[0] not in stop_words:
                word = ps.stem(row[0])
                if len(word)<3:
                    continue
                if float(row[1])<0:
                    label = -1
                elif float(row[1])>0:
                    label = 1
                else:
                    label = 0
                words.append([word, label])

    path = path.replace('.tsv', '_new.tsv')
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for word in words:
            writer.writerow(word)


pretprocess_dataset1(dataset1)
pretprocess_dataset23(dataset2)
pretprocess_dataset23(dataset3)
pretprocess_dataset4(dataset4)
pretprocess_dataset5(dataset5)
pretprocess_dataset6(dataset6)