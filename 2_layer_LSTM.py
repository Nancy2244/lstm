from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from dicts import one_hot_dict
import keras.backend as K
from dicts import temporal


# one_hot = one_hot_dict()
# X_train, X_test = train_test_split(one_hot, test_size=0.3, random_state=5)
# X_train, y_train, X_test, y_test = temporal()
one_hot = one_hot_dict()
X_train, X_test = train_test_split(one_hot, test_size=0.3, random_state=5)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def lstm_64(X_train):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=50, verbose=1)
    print(history.history.keys())
    model.save('lstm_64.h5')
    # demonstrate prediction
    plt.plot(history.history['accuracy'])
    plt.title('lstm_4 accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('lstm_4 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def lstm_6(X_train):
    model = Sequential()
    model.add(LSTM(18, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(6, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(6, activation='relu', return_sequences=True))
    model.add(LSTM(18, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=50, verbose=1, validation_data=(X_test, X_test))
    print(history.history.keys())
    model.save('lstm_18_6.h5')

def lstm_44_4(X_train):
    model = Sequential()
    model.add(LSTM(44, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(44, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=250, verbose=1, validation_data=(X_test, X_test))
    print(history.history.keys())
    model.save('lstm_44_4.h5')

lstm_44_4(X_train)