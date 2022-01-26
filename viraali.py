import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def datan_muokkaus(twiitit):
    twiitit['is_viral'] = np.where(twiitit['retweet_count'] > twiitit['retweet_count'].median() ,1 , 0)
    twiitit['tweet_lenght'] = twiitit.apply(lambda tweet : len(tweet['text']), axis = 1)
    twiitit['followers_count'] = twiitit['user'].apply(lambda d : d.get('followers_count'))
    twiitit['friends_count'] = twiitit['user'].apply(lambda d : d.get('friends_count'))
    twiitit['hastags'] = twiitit.apply(lambda tweet : tweet['text'].count('#'), axis=1)
    labels = twiitit['is_viral']
    features = twiitit[['tweet_lenght', 'followers_count', 'friends_count', 'hastags']]
    return features, labels

def datan_skaalaus(features, labels):
    scaled_features = scale(features, axis=0)
    train_features, test_features, train_labels, test_labels = train_test_split(scaled_features, labels, test_size= 0.2, random_state=1)
    return train_features, test_features, train_labels, test_labels

def luokittelija(train_features, test_features, train_labels, test_labels, k = 38):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_features, train_labels)
    score = classifier.score(test_features, test_labels)
    return score


def main():
    twiitit = pd.read_json("Codeacademy/Twitter/data/random_tweets.json", lines=True)
    features, labels = datan_muokkaus(twiitit)
    train_features, test_features, train_labels, test_labels = datan_skaalaus(features, labels)
    score = luokittelija(train_features, test_features, train_labels, test_labels, k = 38)
    print(score)

def etsi_k():
    twiitit = pd.read_json("Codeacademy/Twitter/data/random_tweets.json", lines=True)
    features, labels = datan_muokkaus(twiitit)
    train_features, test_features, train_labels, test_labels = datan_skaalaus(features, labels)
    
    scores=[]

    for k in range(1,201):
        score = luokittelija(train_features, test_features, train_labels, test_labels, k = k)
        scores.append(score)
    print(f'Parhaimman luokittelijan accuracy {max(scores)}, jolloin luokittelijan k-arvo on {scores.index(max(scores))}')
    plt.plot(range(1,201), scores)
    plt.title('Viraali-twiitti luokittelijan accurary')
    plt.ylabel('accurary')
    plt.xlabel('k')
    plt.show()

main()
etsi_k()