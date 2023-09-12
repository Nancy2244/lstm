from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from dicts import temporal
import keras.backend as K

one_hot = temporal()
X_train, X_test = train_test_split(one_hot, test_size=0.3, random_state=5)
print(X_train.shape)
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

def lstm_4(X_train):
    model = Sequential()
    model.add(LSTM(4, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, X_train, batch_size=33, epochs=50, verbose=1)
    print(history.history.keys())
    model.save('lstm_4.h5')
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
    model.add(LSTM(6, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    # model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    # model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(6, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=50, verbose=1)
    print(history.history.keys())
    model.save('lstm_6.h5')
    # demonstrate prediction
    plt.plot(history.history['accuracy'])
    plt.title('lstm_6 accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('lstm_6 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def lstm_8(X_train):
    model = Sequential()
    model.add(LSTM(8, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    # model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    # model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(8, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=100, verbose=1)
    print(history.history.keys())
    model.save('lstm_8.h5')
    # demonstrate prediction
    plt.plot(history.history['accuracy'])
    plt.title('lstm_8 accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('lstm_8 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def lstm_10(X_train):
    model = Sequential()
    model.add(LSTM(10, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    # model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    # model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=100, verbose=1)
    print(history.history.keys())
    model.save('lstm_10.h5')
    # demonstrate prediction
    plt.plot(history.history['accuracy'])
    plt.title('lstm_10 accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('lstm_10 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def lstm_12(X_train):
    model = Sequential()
    model.add(LSTM(84, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(24, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(24, activation='relu', return_sequences=True))
    model.add(LSTM(84, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1,  activation='relu')))
    model.add(TimeDistributed(Dense(X_train.shape[2], activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=100, verbose=1)
    print(history.history.keys())
    model.save('lstm_12.h5')
    # demonstrate prediction
    plt.plot(history.history['accuracy'])
    plt.title('lstm_12 accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('lstm_12 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def lstm_14(X_train, X_test):
    model = Sequential()
    #encoder
    model.add(LSTM(14, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    #decoder
    model.add(LSTM(14, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    #compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m, recall_m, precision_m])
    model.summary()
    #fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=300, verbose=1, validation_data=(X_test, X_test))
    model.save('lstm_14.h5')
    a = history.history
    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, X_test, verbose=1)
    print(loss, accuracy, f1_score, precision, recall)
    # demonstrate prediction
    # plt.plot(history.history['acc'])
    # plt.title('lstm_14 accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # plt.plot(history.history['f1_m'])
    # plt.title('lstm_14 f1')
    # plt.ylabel('f1')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.title('lstm_14 loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
def lstm_24(X_train, X_test):
    model = Sequential()
    model.add(LSTM(24, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    # model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    # model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(24, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',f1_m, recall_m, precision_m])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=300, verbose=1, validation_data=(X_test, X_test))
    model.save('lstm_24.h5')
    a = history.history
    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, X_test, verbose=1)
    print(loss, accuracy, f1_score, precision, recall)

def lstm_44(X_train, X_test):
    model = Sequential()
    model.add(LSTM(44, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    # model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    # model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(44, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', f1_m, recall_m, precision_m])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=300, verbose=1, validation_data=(X_test, X_test))
    print(history.history.keys())
    model.save('lstm_44.h5')
    a = history.history
    loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, X_test, verbose=1)
    print(loss, accuracy, f1_score, precision, recall)

def lstm_15_temporal(X_train, X_test):
    model = Sequential()
    model.add(LSTM(15, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    # model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    # model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(15, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', f1_m, recall_m, precision_m])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=300, verbose=1, validation_data=(X_test, X_test))
    print(history.history.keys())
    model.save('lstm_15_temporal.h5')

def lstm_10_temporal(X_train, X_test):
    model = Sequential()
    model.add(LSTM(10, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    # model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    # model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', f1_m, recall_m, precision_m])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=140, verbose=1, validation_data=(X_test, X_test))
    print(history.history.keys())
    model.save('lstm_10_temporal.h5')

def lstm_64_temporal(X_train, X_test):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    # model.add(LSTM(4, activation='relu', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    # model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2],  activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', f1_m, recall_m, precision_m])
    model.summary()
    # # fit model
    history = model.fit(X_train, X_train, batch_size=33, epochs=140, verbose=1, validation_data=(X_test, X_test))
    print(history.history.keys())
    model.save('lstm_50_temporal.h5')


# LSTMs = [lstm_14]
# def choose():
#     for i in range(len(LSTMs)):
#         LSTMs[i](X_train, X_test)
#
# choose()

LSTMs_temporal = [lstm_15_temporal, lstm_10_temporal]
def choose_temporal():
    from dicts import temporal
    X_train, y_train, X_test, y_test = temporal()
    print(X_train.shape)
    for i in range(len(LSTMs_temporal)):
        LSTMs_temporal[i](X_train, X_test)

choose_temporal()