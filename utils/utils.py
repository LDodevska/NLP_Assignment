import re
import pandas as pd
import numpy as np
import json
from future.utils import iteritems
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.optimizers import Adam
from keras_contrib.layers import CRF
import keras


def get_sentence_dict(file_name):
    with open(file_name, 'r') as fp:
        sentence_dict = json.load(fp)
    return sentence_dict


def get_input_dict(file_name, lemma=False):
    sentence_dict = get_sentence_dict(file_name)
    input_dict = {}
    if lemma:
        idx = 2
    else:
        idx = 0
    for key, sentence in sentence_dict.items():
        new_sentence = []
        for word in sentence:
            new_sentence.append((word[idx], word[1], word[3]))
        input_dict[int(float(key))] = new_sentence
    return input_dict


def parse_conll2003_dataset(file_name, start_index=0, save_file=False):
    with open(file_name, "r") as file:
        content = file.read()
    find_sentences = re.compile(r'(.*?)(?:\n{2})', re.MULTILINE | re.DOTALL)
    sentences = find_sentences.findall(content)
    parse_sentence = re.compile(r"(\S+)\s(\S+)\s\S+\s(\S+)")
    df = pd.DataFrame(columns=["sentence_idx", "word", "pos", "tag"])
    words = []
    poss = []
    tags = []
    idxs = []
    results = {}
    for i, s in enumerate(sentences):
        tokens = parse_sentence.findall(s)
        results[start_index + i] = tokens
        for token in tokens:
            word, pos, tag = token
            idxs.append(i)
            words.append(word)
            poss.append(pos)
            tags.append(tag)
    df["sentence_idx"] = idxs
    df["word"] = words
    df["pos"] = poss
    df["tag"] = tags
    if save_file:
        df.to_csv("parsed_{}.csv".format(file_name.split(".")[1]))
    return df, results


def get_dataset_info(dataset="conll2003", train_sets="train", test_sets="eng_a"):
    if dataset == "conll2003":
        train_sets_parts = train_sets.split(",")
        test_sets_parts = test_sets.split(",")
        input_dict_test = {}
        df_test = pd.DataFrame(columns=["sentence_idx", "word", "pos", "tag"])
        df, input_dict = parse_conll2003_dataset(file_name="../eng.train")
        if "eng_a" in train_sets_parts:
            df_test_a, input_dict_test_a = parse_conll2003_dataset("../eng.testa", start_index=len(input_dict) + 1)
            input_dict.update(input_dict_test_a)
            df = df.append(df_test_a, ignore_index=True)
        if "eng_b" in train_sets_parts:
            df_test_b, input_dict_test_b = parse_conll2003_dataset("../eng.testb", start_index=len(input_dict) + 1)
            df = df.append(df_test_b, ignore_index=True)
            input_dict.update(input_dict_test_b)
        if "eng_a" in test_sets_parts:
            df_test_a, input_dict_test_a = parse_conll2003_dataset("../eng.testa")
            input_dict_test.update(input_dict_test_a)
            df_test = df_test.append(df_test_a, ignore_index=True)
        if "eng_b" in test_sets_parts:
            df_test_b, input_dict_test_b = parse_conll2003_dataset("../eng.testb")
            input_dict_test.update(input_dict_test_b)
            df_test = df_test.append(df_test_b, ignore_index=True)
        return df, df_test, input_dict, input_dict_test
    if dataset == "lodi":
        df = pd.read_csv("../ner_first_preprocessing.csv")
        df = df[['sentence_idx', 'word', 'lemma', 'pos', 'tag']]
        input_dict = get_input_dict("../sentence_dict.json")
        return df, None, input_dict, None


def convert_predictions(y_pred, indices_to_tag):
    conv_pred = []
    for pred in y_pred:
        conv_tmp = []
        for val in pred:
            val_arg_max = np.argmax(val)
            conv_tmp.append(indices_to_tag[val_arg_max])
        conv_pred.append(conv_tmp)
    return conv_pred


def get_words_and_tags(df_train, df_test=None, dataset="conll2003"):
    tags = set(df_train["tag"].values)
    tags_without_O = list(tags - {"O"})
    tags = list(tags)
    # List with unique words in the dataset
    words = list(set(df_train["word"].values))
    words_test = []

    if df_test is not None:
        words_test = list(set(df_test["word"].values))

    words_total = list(set(words + words_test))
    words_total.append("EOL")
    # Number of unique words
    number_of_words_total = len(words_total)
    # Number of tags
    number_of_tags = len(tags)
    return tags, tags_without_O, words_total, words_test, number_of_words_total, number_of_tags


def get_indices(input_dict, tags, words_train, words_test, dataset="conll2003", input_dict_test=None):
    word_indices = {w: idx for idx, w in enumerate(words_train)}
    tag_indices = {t: idx for idx, t in enumerate(tags)}
    indices_to_tag = {v: k for k, v in iteritems(tag_indices)}
    len_max = max([len(s) for s in input_dict.values()])
    word_indices_test = None
    if dataset == "conll2003":
        len_max_test = max([len(s) for s in input_dict_test.values()])
        if len_max_test > len_max:
            len_max = len_max_test
        word_indices_test = {w: idx for idx, w in enumerate(words_test)}
    return word_indices, word_indices_test, tag_indices, indices_to_tag, len_max


def generate_train_and_test_lists(word_indices, tag_indices, input_dict, input_dict_test, tags, len_max,
                                  dataset="conll2003"):
    input_pad = word_indices["EOL"]
    if dataset == "conll2003":
        x_train = [[word_indices[word[0]] for word in s] for s in list(input_dict.values())]
        y_train = [[tag_indices[t[2]] for t in s] for s in list(input_dict.values())]

        X_train = pad_sequences(maxlen=len_max, sequences=x_train, padding="post", value=input_pad)
        y_train = pad_sequences(maxlen=len_max, sequences=y_train, padding="post", value=tag_indices["O"])
        y_train = [to_categorical(tag, num_classes=len(tags)) for tag in y_train]

        x_test = [[word_indices[word[0]] for word in s] for s in list(input_dict_test.values())]
        y_test = [[tag_indices[t[2]] for t in s] for s in list(input_dict_test.values())]

        X_test = pad_sequences(maxlen=len_max, sequences=x_test, padding="post", value=input_pad)
        y_test = pad_sequences(maxlen=len_max, sequences=y_test, padding="post", value=tag_indices["O"])
        y_test = [to_categorical(tag, num_classes=len(tags)) for tag in y_test]
    else:
        input_final = [[word_indices[word[0]] for word in s] for s in list(input_dict.values())]
        output_final = [[tag_indices[t[2]] for t in s] for s in list(input_dict.values())]

        input_final = pad_sequences(maxlen=len_max, sequences=input_final, padding="post", value=input_pad)
        output_final = pad_sequences(maxlen=len_max, sequences=output_final, padding="post", value=tag_indices["O"])
        output_final = [to_categorical(tag, num_classes=len(tags)) for tag in output_final]
        X_train, X_test, y_train, y_test = train_test_split(input_final, output_final, test_size=0.2)
    return X_train, y_train, X_test, y_test

def create_model(number_of_words_total, number_of_tags, len_max, embedding_size, lstm_units, dropout, recurrent_dropout):
    # Input
    input_layer = Input(shape=(len_max,))

    # Embedding Layer
    model = Embedding(input_dim=number_of_words_total, output_dim=embedding_size, input_length=len_max)(input_layer)
    # BI-LSTM Layer
    model = Bidirectional(LSTM(units=lstm_units, return_sequences=True,
                               dropout=dropout, recurrent_dropout=recurrent_dropout,
                               kernel_initializer=keras.initializers.he_normal()))(model)
    # TimeDistributed layer
    model = TimeDistributed(Dense(number_of_tags, activation="relu"))(model)
    # CRF Layer
    crf = CRF(number_of_tags)

    # Output
    output_layer = crf(model)
    model = Model(input_layer, output_layer)

    # Optimiser
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])
    model.summary()
    return model

def run_model(model, X_train, y_train, X_test, y_test, indices_to_tag, epochs):
    history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=epochs, validation_split=0.1, verbose=1)
    y_pred = model.predict(X_test, verbose=1)

    pred_labels = convert_predictions(y_pred, indices_to_tag)
    real_labels = convert_predictions(y_test, indices_to_tag)
    return history, pred_labels, real_labels

