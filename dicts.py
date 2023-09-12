# dense dict
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def one_hot_united_df():
    actions = pd.read_csv("/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/Апробация COR/actions.csv",
                          encoding='utf-8')
    # actions = actions.drop(['Unnamed: 0', "Action_num"], axis=1)
    actions.dropna(inplace=True)
    ids = actions.Login.unique()
    actions = actions.set_index('Login')
    unique_actions = actions.Action_zone.unique()
    unique_actions = list(unique_actions)
    word2index = {}
    i = 0
    for i, act in enumerate(unique_actions):
        i += 1
        word2index[act] = i
    for word in word2index:
        actions["Action_zone"].replace(word, word2index[word], inplace=True)
    unique_numbers = actions.Action_zone.unique()
    encoded = to_categorical(unique_numbers)
    num2index = {}
    for i, action in enumerate(unique_numbers):
        num2index[action] = encoded[i]
    acts = []
    for index in ids:
        df = actions.loc[index]
        df_list = list(df["Action_zone"])
        act = []
        for i in df_list:
            a = num2index[i]
            act.append(a)
        acts.extend(act)
    return acts


one_hot_united_df()


def temporal(seq_size=1):
    one_hot = one_hot_united_df()
    X_train, X_test = train_test_split(one_hot, test_size=0.3, random_state=5)
    seq_size = 30

    x_values = []
    y_values = []
    x_test_values = []
    y_test_values = []
    for i in range(len(X_train) - seq_size):
        if i == len(one_hot) - seq_size - 1:
            break
        else:
            x_values.append(one_hot[i:(i + seq_size)])
            y_values.append(one_hot[i + seq_size])

    for i in range(len(X_test) - seq_size):
        if i == len(one_hot) - seq_size - 1:
            break
        else:
            x_test_values.append(one_hot[i:(i + seq_size)])
            y_test_values.append(one_hot[i + seq_size])
    print()
    arr = np.array(x_values)
    # df = pd.DataFrame(arr)
    # df.to_csv('/Users/anastasiabelaeva/Desktop/Postgraduate/мои статьи/LSTM/data/temporal.csv')

    return (np.array(x_values))
    # return np.array(y_values), np.array(x_test_values), np.array(y_test_values)


temporal(one_hot)
# def dense():
#     actions = pd.read_csv("/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/оценки рейтеров/score+actions.csv", encoding='utf-8')
#     score = actions
#     actions = actions.drop(['Unnamed: 0', "Action_num"], axis=1)
#     ids = actions.Student_Id.unique()
#     actions = actions.set_index('Student_Id')
#     unique_actions = actions.Action_zone.unique()
#     unique_actions = list(unique_actions)
#     #вот здесь как раз словарь уникальное действие:уникальный номер
#     word2index = {}
#     for i, act in enumerate(unique_actions):
#         i += 1
#         word2index[act] = i
#     for word in word2index:
#         actions["Action_zone"].replace(word, word2index[word], inplace=True)
#     act = []
#     for index in ids:
#         df = actions.loc[index]
#         act.append(list(df["Action_zone"]))
#     #padding the df
#     padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(act, padding="post")
#     max_len= len(padded_inputs)
#     unique_actions = len(unique_actions)
#     n_features = len(padded_inputs[6])
#     return padded_inputs, unique_actions, max_len, n_features
# dense()

# def one_dict():
#     actions = pd.read_csv("/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/оценки рейтеров/score+actions.csv",
#                           encoding='utf-8')
#     actions = actions.drop(['Unnamed: 0', "Action_num"], axis=1)
#     actions.dropna(inplace=True)
#     ids = actions.Student_Id.unique()
#     actions = actions.set_index('Student_Id')
#     unique_actions = actions.Action_zone.unique()
#     unique_actions = list(unique_actions)
#     acts = []
#     for index in ids:
#         act =[]
#         df = actions.loc[index]
#         act = df["Action_zone"]
#         a = ' '.join(act)
#         acts.append(a)
#     # padding the df
#     encoded_docs = [one_hot(d, len(unique_actions)) for d in acts]
#     padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, padding="post")
#     vocab_size = len(unique_actions)
#     n_samples = len(padded_inputs)
#     input_length = len(padded_inputs[1])
#     return padded_inputs, vocab_size, n_samples, input_length
# one_dict()


# def one_hot_dict():
#
#     actions = pd.read_csv("/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/Апробация COR/actions.csv",
#                           encoding='utf-8')
#     # actions = actions.drop(['Unnamed: 0', "Action_num"], axis=1)
#     actions.dropna(inplace=True)
#     ids = actions.Login.unique()
#     actions = actions.set_index('Login')
#     unique_actions = actions.Action_zone.unique()
#     unique_actions = list(unique_actions)
#     word2index = {}
#     i = 0
#     for i, act in enumerate(unique_actions):
#         i+=1
#         word2index[act] = i
#     for word in word2index:
#         actions["Action_zone"].replace(word, word2index[word], inplace=True)
#     unique_numbers = actions.Action_zone.unique()
#     encoded = to_categorical(unique_numbers)
#     num2index = {}
#     for i, action in enumerate(unique_numbers):
#         num2index[action] = encoded[i]
#     acts = []
#     for index in ids:
#         df = actions.loc[index]
#         df_list = list(df["Action_zone"])
#         a = []
#         act = []
#         for i in df_list:
#             a = num2index[i]
#             act.append(a)
#         acts.append(act)
#     #padding the df
#     padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(acts, padding="post", maxlen=88)
#     return padded_inputs
# one_hot_dict()

# actions = pd.read_csv("/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/Апробация COR/actions.csv",
#                           encoding='utf-8')
# actions = actions.set_index('Login')
# a = actions.groupby('Login').agg({'Action_zone':'count'}).mean()
