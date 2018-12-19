"""
mlp_kfold.py: Runs Multilayer Perceptron Neural Network to predict a user selected personality trait.

Performs k-fold cross validation to figure out optimal values of hyper-parameters

The data essays are converted to lowercase before use.
The labels are Binarized

Example usage:  1) python mlp_kfold.py <trait>

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
import operator
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
from itertools import product


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


def parse_classification_report(classification_report_arg):
    """
    Source StackOverflow
    Parse a sklearn classification report into a dict keyed by class name
    and containing a tuple (precision, recall, fscore, support) for each class
    :param classification_report_arg: the generated classification report
    """
    lines = classification_report_arg.split('\n')
    # Remove empty lines
    lines = list(filter(lambda l: not len(l.strip()) == 0, lines))

    # Starts with a header, then score for each class and finally an average
    header = lines[0]
    cls_lines = lines[1:-1]
    avg_line = lines[-1]

    assert header.split() == ['precision', 'recall', 'f1-score', 'support']
    assert avg_line.split()[0] == 'avg'

    # We cannot simply use split because class names can have spaces. So instead
    # figure the width of the class field by looking at the indentation of the
    # precision header
    cls_field_width = len(header) - len(header.lstrip())

    # Now, collect all the class names and score in a dict

    def parse_line(l):
        """Parse a line of classification_report"""
        cls_name = l[:cls_field_width].strip()
        precision, recall, fscore, support = l[cls_field_width:].split()
        precision = float(precision)
        recall = float(recall)
        fscore = float(fscore)
        support = int(support)
        return (cls_name, precision, recall, fscore, support)

    data = OrderedDict()
    for l in cls_lines:
        ret = parse_line(l)
        cls_name = ret[0]
        scores = ret[1:]
        data[cls_name] = scores

    # average
    data['avg'] = parse_line(avg_line)[1:]
    return data


def classify(trait_arg, params_arg, performance_dict):
    """
    Runs a classifier on given activation type
    :param trait_arg:  the trait to predict
    :param params_arg: the values of activation_type,  learning_rate_type, solver_type
    :param performance_dict: to store performance for each combination
    """

    activation_type,  learning_rate_type, solver_type = params_arg
    print("Using Parameters : ", params_arg)
    x = df['essay'][1:]
    x = x.str.lower()
    y = df[trait_arg][1:]

    print("Predicting ", trait_arg)

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # 10 fold
    kf = KFold(n_splits=10)

    # MLP classifier
    classifier = MLPClassifier(activation=activation_type, batch_size='auto',
                               hidden_layer_sizes=(60), learning_rate=learning_rate_type, max_iter=20,
                               random_state=None, solver=solver_type)

    ind = 0
    precision_dict = dict()
    recall_dict = dict()
    accuracy_dict = dict()
    for train_indices, test_indices in kf.split(x, y):
        x_train, x_test, y_train, y_test = x[train_indices][1:], x[test_indices].tolist()[1:], \
                                           y[train_indices][1:], y[test_indices].tolist()[1:]
        train_x_vector = vectorizer.fit_transform(x_train)
        test_X_vector = vectorizer.transform(x_test)
        classifier.fit(train_x_vector, y_train)
        guess = classifier.predict(test_X_vector)
        rep = classification_report(y_test, guess)
        a, b, c, d = dict(parse_classification_report(rep))['avg']
        precision_dict[ind] = a
        recall_dict[ind] = b
        accuracy_dict[ind] = accuracy_score(y_test, guess)
        ind = ind + 1

    p, r, a = (float(sum(precision_dict.values())) / 10), (float(sum(recall_dict.values())) / 10)\
        , (float(sum(accuracy_dict.values())) / 10)
    performance_dict[params_arg] = (p + r + a) * 33.33
    print("Precision : ", p * 100, ",  Recall : ", r * 100,  ", Accuracy : ", a * 100)
    return performance_dict


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

            for param in a_l_s:
                performance = classify(trait, param, performance)

            best = max(performance.items(), key=operator.itemgetter(1))[0]
            print(best, " seems to perform the best.")
    else:
        print("Incorrect command line arguments")
