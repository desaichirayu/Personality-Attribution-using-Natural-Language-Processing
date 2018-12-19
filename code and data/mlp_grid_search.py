"""
mlp_grid_search.py: Runs Multilayer Perceptron Neural Network to predict a user selected personality trait.

Performs grid-search cross validation to figure out optimal values of hyper-parameters

The data essays are converted to lowercase before use.
The labels are Binarized

Example usage:  1) python mlp_grid_search.py <trait>

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
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
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


def classify(trait_arg):
    """
    Runs MLP classifier with provided parameters
    :param trait_arg: the trait to predict
    """
    x = df['essay'][1:]
    x = x.str.lower()
    y = df[trait_arg][1:]
    # binarize labels
    y = np.where(y == 'n', 0, 1)

    print("Predicting ", trait_arg)
    print("Test set, Train Set ratio: 1:3")

    # Test train split in 25 : 75 ratio
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=11)

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    xx_train = vectorizer.fit_transform(x_train)
    xx_test = vectorizer.transform(x_test)

    # Lists of Possible Hyper-parameter values
    activation_types = ['identity', 'logistic', 'tanh', 'relu']
    learning_rates = ['constant', 'invscaling', 'adaptive']
    solver_types = ['lbfgs', 'sgd', 'adam']
    hidden_layers = [(20), (40), (60), (6, 10), (10, 20), (20, 40), (50, 100), (75, 150, 300), (50, 100, 150, 200)]
    iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500]

    parameters = [{'activation': activation_types, 'learning_rate': learning_rates,
                   'solver': solver_types, 'hidden_layer_sizes': hidden_layers, 'max_iter': iterations}]

    scorers = ['accuracy', 'precision', 'recall', 'f1']

    # Tune for each scorer
    for scorer in scorers:
        print("Tuning hyper-parameters for %s" % scorer)
        print()

        # Grid Search with 10-fold cross validation
        clf = GridSearchCV(MLPClassifier(), parameters, cv=10, scoring=scorer, n_jobs=4)
        clf.fit(xx_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        standard_deviation = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, standard_deviation, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        predictions = clf.predict(xx_test)
        print(classification_report(y_test, predictions))
        print(accuracy_score(y_test, predictions))
        print()


if __name__ == "__main__":

    if not len(sys.argv) > 1:
        print("No command line Arguments Provided")
    elif len(sys.argv) == 2:
        trait_index = sys.argv[1]
        trait, params = get_choice(trait_index)
        if trait is None:
            print("Trait index value should be between 0 and 4")
        else:
            classify(trait)
    else:
        print("Incorrect command line arguments")
