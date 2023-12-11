import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.backend import clear_session
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

filepath_dict = {'yelp':   'sentiment_labelled_sentences/yelp_labelled.txt',
                 'amazon': 'sentiment_labelled_sentences/amazon_cells_labelled.txt',
                 'imdb':   'sentiment_labelled_sentences/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)


df_yelp = df[df['source'] == 'yelp']
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values
# sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
sentences_train = sentences[:749]
sentences_test = sentences[750:]
y_train = y[:749]
y_test = y[750:]

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

print(X_train[0].A)

from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]  # Number of features

print("shape of x_train is " +  str(X_train.shape[0]))
print("shape of x_train is " +  str(X_train.shape[1]))

print("shape of x_test is " +  str(X_test.shape[0]))
print("shape of x_test is " +  str(X_test.shape[1]))

model = Sequential()
model.add(layers.Dense(5, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
            epochs=5, verbose=False,
            validation_data=(X_test, y_test),
            batch_size=10)

#clear_session()

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
cnt = 0
for i in X_test:
    y = model(i.A)
    print(cnt)
    print("y is " + str(y))
    print(sentences_test[cnt])
    print()
    cnt = cnt + 1
#plot_history(history)