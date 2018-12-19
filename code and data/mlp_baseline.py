import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras import layers, Sequential
from matplotlib import pyplot

def get_common_words(text):
    text = nltk.word_tokenize(text)
    text = nltk.Text(text)
    informative_words = [word.lower() for word in text if len(word) > 5]
    return set(informative_words)

X = []
y = []

# Read data from file
with open('essays.csv', encoding='latin-1') as f:
    i = 0
    for row in f:
        entry = []
        for j in range(5):
            if j == 0:
                val = (row.rsplit(",", 1)[-1])[:-1]
            else:
                val = row.rsplit(",", 1)[-1]
            entry.insert(0, val)
            row = row.rsplit(",")[:-1]
            row = ",".join(row)
        entry.insert(0, row.split(",",1)[0])
        entry.insert(1, row.split(",",1)[1])
        i += 1
        res = get_common_words(entry[1])
        X.append(res)
        y.append(entry[2:])
X = X[1:]
y = y[1:]


# Use w2v aggregated word vectors
l = []
with open("w2v_features.txt") as f:
    contents = f.read();
    contents = contents.split("\n")
    for i in contents:
        if len(i) > 0:
            vec = i.split(",")
            vec = [float(x) for x in vec]
            l.append(np.array(vec))
X = np.array(l)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# Function used to vectorize label vectors
def vectorize_labels(labels, dimension=5):
    results = zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        for j in range(5):
            if label[j] == 'y':
                results[i, j] = 1
            else:
                results[i, j] = 0
    return results

y_train_labels = vectorize_labels(y_train)
y_test_labels = vectorize_labels(y_test)


# Create model using Keras
model = Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(X.shape[1], )))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(X_train, y_train_labels, epochs=100, batch_size=100, validation_split=0.1)

# Evaluation
predictions = model.predict(X_test)
labels = ['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']
for i, test_rec in enumerate(X_test):
    s = ""
    for j, val in enumerate(predictions[i]):
        if val > 0.5:
            s = s + labels[j] + " "

# Plot the accuracy and loss curves
def plot_accuracy(history):
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('model accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['training', 'validation'], loc='lower right')
    pyplot.show()

def plot_loss(history):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['training', 'validation'], loc='upper right')
    pyplot.show()

# Summarize history for accuracy
plot_accuracy(history)

# Summarize history for loss
plot_loss(history)

# Read results for the test set predictions
results = model.evaluate(X_test, y_test_labels)
print(model.metrics_names)
print(results)