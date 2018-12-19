"""
naive_simple.py: Runs Simple Multinomial Naive Bayes Classifier to predict a user selected personality trait.

The data essays are converted to lowercase before use

Example usage:  python naive_simple.py <trait> <custom_alpha>

                <trait> can take vales form 0 to 4 based on the trait for which the user wants to run model denoted by:
                0: Extraversion
                1: Neuroticism
                2: Agreeableness
                3: Conscientiousness
                4: Openness

                <custom_alpha> is the smoothing parameter allows values from 0.0001 to 2.0000
"""
__author__ = "Chirayu Desai"

import sys
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
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


def classify(trait_arg, alpha):
    """
    Runs Naive Bayes classifier with provided parameters
    :param trait_arg: the trait to predict
    :param alpha: the alpha value to be used for smoothing
    """
    x = df['essay'][1:]
    x = x.str.lower()
    y = df[trait_arg][1:]

    print("Predicting ", trait_arg, " with alpha = ", alpha)
    print("Test set, Train Set ratio: 1:3")

    # Test train split in 25 : 75 ratio
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=11)

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    xx_train = vectorizer.fit_transform(x_train)
    xx_test = vectorizer.transform(x_test)

    # Multinomial Naive Bayes Classifier
    classifier = MultinomialNB(alpha=alpha)
    classifier.fit(xx_train, y_train)

    predictions = classifier.predict(xx_test)
    print("Confusion Matrix:")
    print(classification_report(y_test, predictions))
    score = accuracy_score(y_test, predictions)
    print("Accuracy:", score)


if __name__ == "__main__":

    if not len(sys.argv) > 1:
        print("No command line Arguments Provided")
    else:
        trait_index = sys.argv[1]

    if len(sys.argv) > 2:
        custom_alpha = float(sys.argv[2])
    else:
        custom_alpha = None

    if custom_alpha is not None:
        if 0.0001 <= custom_alpha < 2.0001:
            trait, default_alpha = get_choice(trait_index)
            if trait is None:
                print("Trait index value should be between 0 and 4")
            else:
                classify(trait, custom_alpha)
        else:
            print("Please Enter Alpha Values between 0.0001 and 2.0000")
    else:
        trait, default_alpha = get_choice(trait_index)
        if trait is None:
            print("Trait index value should be between 0 and 4")
        else:
            classify(trait, default_alpha)
