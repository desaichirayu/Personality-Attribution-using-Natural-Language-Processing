"""
mlp_iterative.py: Runs Multilayer Perceptron Neural Network to predict a user selected personality trait.

Performs iterative scoring to estimate best parameters
The data essays are converted to lowercase before use

Example usage:  1) python mlp_iterative.py <trait>

                <trait> can take vales form 0 to 4 based on the trait for which the user wants to run model denoted by:
                0: Extraversion
                1: Neuroticism
                2: Agreeableness
                3: Conscientiousness
                4: Openness

"""
__author__ = "Chirayu Desai"

import sys
import warnings
import pandas as pd
from itertools import product
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

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
        '0': ('Extraversion', ('tanh', 'adaptive', 'lbfgs')),
        '1': ('Neuroticism', ('tanh', 'adaptive', 'lbfgs')),
        '2': ('Agreeableness', ('tanh', 'adaptive', 'lbfgs')),
        '3': ('Conscientiousness', ('relu', 'invscaling', 'lbfgs')),
        '4': ('Openness', ('relu', 'invscaling', 'lbfgs'))
    }.get(choice, (None, None))


def classify(trait_arg, activation_arg, learning_rate_arg, solver_arg):
    """
    Runs MLP classifier with provided parameters
    :param trait_arg: the trait to predict
    :param activation_arg: the activation function
    :param learning_rate_arg: the type of learning for neural network
    :param solver_arg: the type of solver to be used
    """
    x = df['essay'][1:]
    x = x.str.lower()
    y = df[trait_arg][1:]

    print("Predicting ", trait_arg, " with arguments = ", activation_arg, "\t", learning_rate_arg, "\t", solver_arg)
    print("Test set, Train Set ratio: 1:3")

    # Test train split in 25 : 75 ratio
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=11)

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    xx_train = vectorizer.fit_transform(x_train)
    xx_test = vectorizer.transform(x_test)

    # Multilayer Perceptron Classifier with single hidden layer size = 60
    classifier = MLPClassifier(activation=activation_arg, alpha=0.0001, hidden_layer_sizes=(60),
                               learning_rate=learning_rate_arg, max_iter=20, solver=solver_arg)
    classifier.fit(xx_train, y_train)

    predictions = classifier.predict(xx_test)
    print("Confusion Matrix:")
    print(classification_report(y_test, predictions))
    score = accuracy_score(y_test, predictions)
    print("Accuracy:", score)


if __name__ == "__main__":

    if not len(sys.argv) > 1:
        print("No command line Arguments Provided")
    elif len(sys.argv) == 2:
        trait_index = sys.argv[1]
        trait, params = get_choice(trait_index)
        if trait is None:
            print("Trait index value should be between 0 and 4")
        else:
            performance = dict()

            activation_types = ['identity', 'logistic', 'tanh', 'relu']
            learning_rates = ['constant', 'invscaling', 'adaptive']
            solver_types = ['lbfgs', 'sgd', 'adam']

            a_l_s = product(activation_types, learning_rates, solver_types)
            for params in a_l_s:
                ac, lr, sl = params
                classify(trait, ac, lr, sl)
    else:
        print("Incorrect command line arguments")
