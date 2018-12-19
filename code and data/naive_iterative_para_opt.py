"""
naive_iterative_para_opt.py: Runs Multinomial Naive Bayes Classifier to predict a user selected personality trait.

Performs iterative evaluation over a range of alpha to figure out good values of alpha

The data essays is converted to lowercase before use

Example usage:  python naive_iterative_para_opt.py <trait>

                <trait> can take vales form 0 to 4 based on the trait for which the user wants to run model denoted by:
                0: Extraversion
                1: Neuroticism
                2: Agreeableness
                3: Conscientiousness
                4: Openness

"""
__author__ = "Chirayu Desai"

import sys
import operator
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# read data from csv
df = pd.read_csv('essays.csv', names=['author_id', 'essay', 'Extraversion', 'Neuroticism',
                                      'Agreeableness', 'Conscientiousness', 'Openness'], encoding='ansi')


def get_choice(choice):
    """
    Get the users choice for which trait to predict based on provided command line option
    :param choice: the value of command line option
    :return: the trait label and default alpha value
    """
    return {
        '0': ('Extraversion', 0.07),
        '1': ('Neuroticism', 0.27),
        '2': ('Agreeableness', 0.11),
        '3': ('Conscientiousness', 0.09),
        '4': ('Openness', 0.45)
    }.get(choice, (None, None))


def classify(trait_arg):
    """
    Runs Naive Bayes classifier with iterative search for provided trait
    :param trait_arg: the trait to predict
    """
    print("Predicting for trait: ", trait_arg)
    x = df['essay'][1:]
    x = x.str.lower()
    y = df[trait_arg][1:]

    # Range of alpha values
    params = np.arange(0.01, 2.01, 0.01, dtype=float).tolist()
    params = [float(f'{x:.3f}') for x in params]

    print("Test set, Train Set ratio: 1:3")

    # Test train split in 25 : 75 ratio
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=11)
    scores = dict()

    for alpha in params:

        # TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        xx_train = vectorizer.fit_transform(x_train)
        xx_test = vectorizer.transform(x_test)

        # Multinomial Naive Bayes Classifier
        classifier = MultinomialNB(alpha=alpha)
        classifier.fit(xx_train, y_train)

        predictions = classifier.predict(xx_test)

        # print(classification_report(y_test, predictions))
        score = accuracy_score(y_test, predictions)
        print("Alpha: ", alpha, " \t Accuracy: ", score)
        scores[alpha] = score

    print('Best Alpha: ', max(scores.items(), key=operator.itemgetter(1))[0], ' with Accuracy : ', max(scores.values()))


if __name__ == "__main__":

    if not len(sys.argv) > 1:
        print("No command line Arguments Provided")
    else:
        trait_index = sys.argv[1]

    trait, default_alpha = get_choice(trait_index)
    if trait is None:
        print("Trait index value should be between 0 and 4")
    else:
        classify(trait)
