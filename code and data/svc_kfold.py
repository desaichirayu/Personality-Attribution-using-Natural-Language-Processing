import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# Read data from csv
df = pd.read_csv('essays.csv', names=['author_id', 'essay', 'Extraversion', 'Neuroticism',
                                      'Agreeableness', 'Conscientiousness', 'Openness'], encoding='latin-1')

def get_choice(choice):
    """
    Get the users choice for which trait to predict based on provided command line option
    :param choice: the value of command line option
    :return: the trait label and default alpha value
    """
    return {
        '0': 'Extraversion',
        '1': 'Neuroticism',
        '2': 'Agreeableness',
        '3': 'Conscientiousness',
        '4': 'Openness'
    }.get(choice, (None, None))

def fetch_prediction_results(y_test, predicted):
	"""
	Compute the evaluation metrics
	"param y_test : The test label vector"
	"param predicted: The predicted label vector"
	"return: the metrics computed as a multiple argument return value"
	"""
	tp = 0
	fp = 0
	tn = 0
	fn = 0

	for i in range(len(predicted)):
	    if predicted[i] == 'y' and y_test[i] == 'y': tp += 1
	    elif predicted[i] == 'y' and y_test[i] == 'n': fp += 1
	    elif predicted[i] == 'n' and y_test[i] == 'n': tn += 1
	    else: fn += 1

	# Calculate metrics from raw counts of the confusion matrix
	accuracy = accuracy_score(y_test, predicted)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = (2 * precision * recall) / (precision + recall)
	return accuracy, precision, recall, f1

def classify(trait_arg):
	"""
	Do SVC classification with k-fold cross validation performed over 10 iterations
	:param trait_arg: the index of the personality trait for which we want to do prediction
	"""
	print("Predicting ", trait_arg)
	X = np.array(df['essay'])
	y = df[trait_arg]

	# Initialize k-fold cross validator
	k = 10
	kf = KFold(n_splits=k)
	mlb = preprocessing.MultiLabelBinarizer()

	accuracy_l = []
	precision_l = []
	recall_l = []
	
	# Run the classifier prediction for k splits
	for train_indices, test_indices in kf.split(X, y):

		X_train, X_test, y_train, y_test = X[train_indices][1:], X[test_indices].tolist()[1:], \
                                           y[train_indices][1:], y[test_indices].tolist()[1:]

		classifier = Pipeline([
		    ('vectorizer', CountVectorizer()),
		    ('tfidf', TfidfTransformer()),
		    ('clf', OneVsRestClassifier(LinearSVC()))
		])

		classifier.fit(X_train, y_train)

		predicted = classifier.predict(X_test)
		a, p, r, f = fetch_prediction_results(y_test, predicted)
		accuracy_l.append(a)
		precision_l.append(p)
		recall_l.append(r)

	# Average the results
	acc = sum(accuracy_l) / k
	prec = sum(precision_l) / k
	recall = sum(recall_l) / k
	f1 = (2 * prec * recall) / (prec + recall)

	return acc, prec, recall, f1

if __name__ == "__main__":

    if not len(sys.argv) > 1:
        print("No command line Arguments Provided")
    elif len(sys.argv) == 2:
        trait_index = sys.argv[1]
        trait = get_choice(trait_index)
        if trait is None:
            print("Trait index value should be between 0 and 4")
        else:
            results = classify(trait)
            print(list(zip(["Accuracy", "Precision", "Recall", "F1 score"], results)))
    else:
        print("Incorrect command line arguments")

