# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
This project aims to implement an LSTM-based model for named entity recognition (NER) in text, targeting the identification of entities like persons,
organizations, and locations. By leveraging deep learning techniques, we seek to develop a robust system capable of accurately labeling named 
entities in unstructured text data

![Screenshot 2024-10-04 111632](https://github.com/user-attachments/assets/7cf50320-9e85-4a60-949d-b92c079a68ba)

## DESIGN STEPS

### STEP 1: Import the necessary packages

### STEP 2: Read the dataset, and fill the null values using forward fill.

### STEP 3: Create a list of words, and tags. Also find the number of unique words and tags in the dataset.

### STEP 4: Create a dictionary for the words and their Index values. Do the same for the tags as well,Now we move to moulding the data for training and testing.

### STEP 5: We do this by padding the sequences,This is done to acheive the same length of input data.

## PROGRAM
### Name: Praveen V
### Register Number:21222233004
```
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model


data = pd.read_csv("/content/ner_dataset.csv", encoding="latin1")


data.head(50)


data = data.fillna(method="ffill")


data.head(50)


print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())

words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())

print("Unique tags are:", tags)

num_words = len(words)
num_tags = len(tags)

num_words



class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = getter.sentences

len(sentences)

sentences[0]

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

word2idx

plt.hist([len(s) for s in sentences], bins=50)
plt.show()

X1 = [[word2idx[w[0]] for w in s] for s in sentences]


type(X1[0])


X1[0]

max_len = 50


X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)

X[0]

y1 = [[tag2idx[w[2]] for w in s] for s in sentences]

y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)

X_train[0]


y_train[0]


input_word = layers.Input(shape=(max_len,))
embedding_layer = layers.Embedding(input_dim=num_words,
                         output_dim=50,
                         input_length=max_len)(input_word)
dropout_layer = layers.SpatialDropout1D(0.1)(embedding_layer)
bi_lstm = layers.Bidirectional(layers.LSTM(units=100,
                                  return_sequences=True,
                                  recurrent_dropout=0.1))(dropout_layer)
output=layers.TimeDistributed(layers.Dense(num_tags, activation="softmax"))(bi_lstm)
model = Model(input_word, output)


print('Name: Praveen.V Register Number: 212222233004   ')
model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test,y_test),
    batch_size=32,
    epochs=3,
)

metrics = pd.DataFrame(model.history.history)
metrics.head()

print('Name: Praveen V  Register Number: 212222233004    ')
metrics[['accuracy','val_accuracy']].plot()

print('Name: Praveen V  Register Number: 212222233004    ')
metrics[['loss','val_loss']].plot()

i = 20
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print('Name: Praveen V  Register Number: 212222233004    ')
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![download](https://github.com/user-attachments/assets/302dbf9b-a607-41c5-a7c0-d7978694a9df)

![Screenshot 2024-10-04 104539](https://github.com/user-attachments/assets/87f35678-540e-4cb3-84d5-0c79f545687c)

![Screenshot 2024-10-04 104645](https://github.com/user-attachments/assets/a399615d-4cad-435c-91eb-104c3231929d)

![Screenshot 2024-10-04 104827](https://github.com/user-attachments/assets/efc55dbc-e088-40ae-a37d-2f0deea2daaf)

![Screenshot 2024-10-04 104944](https://github.com/user-attachments/assets/e8559bcf-e6e0-4dff-babd-47e9ab0678a9)

![Screenshot 2024-10-04 105130](https://github.com/user-attachments/assets/6472feb7-82c0-4cfa-8399-caca9055a8aa)

## RESULT
Thus, an LSTM-based model (bi-directional) for recognizing the named entities in the text is developed Successfully.
