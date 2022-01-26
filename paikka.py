import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def lataa_data():
    new_yorkin_twiitit = pd.read_json("Codeacademy/Twitter/data/new_york.json", lines=True)
    lontoon_twiitit = pd.read_json('Codeacademy/Twitter/data/london.json', lines=True)
    pariisin_twiitit = pd.read_json('Codeacademy/Twitter/data/paris.json', lines=True)
    return new_yorkin_twiitit, lontoon_twiitit, pariisin_twiitit

def muokkaa_data(new_yorkin_twiitit, lontoon_twiitit, pariisin_twiitit):
    features = pd.concat([new_yorkin_twiitit['text'], lontoon_twiitit['text'], pariisin_twiitit['text']])
    labels = [0] * len(new_yorkin_twiitit) + [1] * len(lontoon_twiitit) + [2] * len(pariisin_twiitit)
    train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=1)

    counter = CountVectorizer()
    counter.fit(train_data, train_labels)
    train_counts = counter.transform(train_data)
    test_counts = counter.transform(test_data)

    
    return train_counts, test_counts, train_labels, test_labels

def malli(train_counts, test_counts, train_labels, test_labels):
    classifier = MultinomialNB()
    classifier.fit(train_counts, train_labels)
    predictions = classifier.predict(test_counts)
    con_matrix = confusion_matrix(predictions, test_labels)
    acc = accuracy_score(predictions, test_labels)
    return con_matrix, acc

def piirra_con_matrix(con_matrix, acc):
    tick_labels = ['New York 0', 'Lontoo 1', 'Pariisi 2']
    tick_loc = [0.5, 1.5, 2.5]
    plt.figure(figsize=(8,8))
    sns.heatmap(con_matrix, cmap='Blues', cbar=False, annot = True, fmt='g', annot_kws={'fontsize' : 14})
    plt.title(f'Confusion matrix, accuracy {round(acc, 2)}', fontsize = 20)
    plt.xlabel('Predicted labels', fontsize = 14)
    plt.ylabel('True labels', fontsize = 14)
    plt.xticks(tick_loc, labels=tick_labels)
    plt.yticks(tick_loc, labels=tick_labels)
    plt.show()


def main():
    new_yorkin_twiitit, lontoon_twiitit, pariisin_twiitit = lataa_data()
    train_counts, test_counts, train_labels, test_labels = muokkaa_data(new_yorkin_twiitit, lontoon_twiitit, pariisin_twiitit)
    con_matrix, acc = malli(train_counts, test_counts, train_labels, test_labels)
    print('Mallin accuracy arvo:', acc)
    piirra_con_matrix(con_matrix, acc)

main()
