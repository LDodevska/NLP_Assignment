import argparse

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import model_from_yaml
import warnings

import logging

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger('tensorflow').disabled = True
# tf.get_logger().setLevel('INFO')

elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


def predict_gmb(sentence):
    yaml_file = open('elmo_gmb_1.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    batch_size = 32
    max_len = 81
    # loaded_model = model_from_yaml(loaded_model_yaml, custom_objects={"elmo_model": elmo_model, "tf": tf, "batch_size": batch_size, "max_len": max_len})
    loaded_model = model_from_yaml(loaded_model_yaml, custom_objects={"elmo_model": elmo_model, "tf": tf, "hub": hub,
                                                                      "batch_size": batch_size, "max_len": max_len})
    # load weights into new model
    loaded_model.load_weights("elmo_gmb_1.h5")

    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    sent = sentence
    sentences_to_predict = []
    words_to_predict = sent.split()
    input_pad = ["PAD"]
    len_max = 81
    words_to_predict.extend(input_pad * (len_max - len(words_to_predict)))

    sentences_to_predict.append(words_to_predict)

    # This is necessary because the model expects input with fixed length and batch size
    for idx in range(0, 31):
        sentences_to_predict.append(["PAD"] * len_max)

    sentences_to_predict = np.array(sentences_to_predict)

    predicted_result = loaded_model.predict(sentences_to_predict)

    sentence_result = predicted_result[0]
    sentence_result = np.argmax(sentence_result, axis=-1)
    # indices_to_tag_gmb = {0: 'B-geo', 1: 'B-tim', 2: 'I-gpe', 3: 'I-art', 4: 'B-per', 5: 'I-eve', 6: 'B-gpe',
    #                       7: 'I-geo', 8: 'B-eve', 9: 'I-nat', 10: 'B-nat', 11: 'I-org', 12: 'I-tim', 13: 'I-per',
    #                       14: 'B-org', 15: 'B-art', 16: 'O'}

    indices_to_tag_gmb = {0: 'B-geo', 1: 'B-tim', 2: 'I-gpe', 3: 'I-art', 4: 'B-per', 5: 'I-eve', 6: 'B-gpe',
                          7: 'I-geo', 8: 'B-eve', 9: 'I-nat', 10: 'O', 11: 'I-org', 12: 'I-tim', 13: 'I-per',
                          14: 'B-org', 15: 'B-art', 16: 'B-nat'}

    print("{:15} {:5}".format("Word", "Pred"))
    print("=" * 25)
    for word, res in zip(words_to_predict, sentence_result):
        if word != "PAD":
            print("{:15}:{:5}".format(word, indices_to_tag_gmb[res]))


def predict_conll(sentence):
    # yaml_file = open('/content/drive/My Drive/ELMO/elmo_gmb_1.yaml', 'r')
    yaml_file = open('elmo_conll_1.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    batch_size = 32
    max_len = 140
    # loaded_model = model_from_yaml(loaded_model_yaml, custom_objects={"elmo_model": elmo_model, "tf": tf, "batch_size": batch_size, "max_len": max_len})
    loaded_model = model_from_yaml(loaded_model_yaml, custom_objects={"elmo_model": elmo_model, "tf": tf, "hub": hub,
                                                                      "batch_size": batch_size, "max_len": max_len})
    # load weights into new model
    # loaded_model.load_weights("/content/drive/My Drive/ELMO/elmo_gmb_1.h5")
    loaded_model.load_weights("elmo_conll_1.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    sent = sentence
    sentences_to_predict = []
    words_to_predict = sent.split()
    input_pad = ["PAD"]
    len_max = 140
    words_to_predict.extend(input_pad * (len_max - len(words_to_predict)))

    sentences_to_predict.append(words_to_predict)

    # This is necessary because the model expects input with fixed length and batch size
    for idx in range(0, 31):
        sentences_to_predict.append(["PAD"] * len_max)

    sentences_to_predict = np.array(sentences_to_predict)

    predicted_result = loaded_model.predict(sentences_to_predict)

    sentence_result = predicted_result[0]
    sentence_result = np.argmax(sentence_result, axis=-1)
    indices_to_tag_conll = {5: 'O', 1: 'I-PER', 4: 'I-ORG', 6: 'I-MISC', 0: 'I-LOC', 3: 'B-ORG', 7: 'B-MISC',
                            2: 'B-LOC'}

    print("{:15} {:5}".format("Word", "Pred"))
    print("=" * 25)
    for word, res in zip(words_to_predict, sentence_result):
        if word != "PAD":
            print("{:15}:{:5}".format(word, indices_to_tag_conll[res]))


def show_examples():
    # yaml_file = open('/content/drive/My Drive/ELMO/elmo_gmb_1.yaml', 'r')

    input_pad = ["PAD"]
    len_max = 140
    max_len = 140
    batch_size = 32

    yaml_file = open('elmo_conll_1.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    # loaded_model = model_from_yaml(loaded_model_yaml, custom_objects={"elmo_model": elmo_model, "tf": tf, "batch_size": batch_size, "max_len": max_len})
    loaded_model = model_from_yaml(loaded_model_yaml, custom_objects={"elmo_model": elmo_model, "tf": tf, "hub": hub,
                                                                      "batch_size": batch_size, "max_len": max_len})
    # load weights into new model
    # loaded_model.load_weights("/content/drive/My Drive/ELMO/elmo_gmb_1.h5")
    loaded_model.load_weights("elmo_conll_1.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    sent = "As Harry rides the Hogwarts Express on his journey back from school after his fourth year and the dire Triwizard Tournament, he dreads disembarking from the train"
    sent_2 = "European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices"
    sent_3 = "I just had breakfast in London in Blue Cafe"

    sentences_to_predict = []

    words_to_predict = sent.split()
    words_to_predict.extend(input_pad * (len_max - len(words_to_predict)))
    words_to_predict_2 = sent_2.split()
    words_to_predict_2.extend(input_pad * (len_max - len(words_to_predict_2)))

    sentences_to_predict.append(words_to_predict)
    sentences_to_predict.append(words_to_predict_2)

    # This is necessary because the model expects input with fixed length and batch size
    for idx in range(0, 30):
        sentences_to_predict.append(["PAD"] * len_max)

    sentences_to_predict = np.array(sentences_to_predict)

    predicted_result = loaded_model.predict(sentences_to_predict)

    sentence_result = predicted_result[0]
    sentence_result = np.argmax(sentence_result, axis=-1)
    sentence_result_2 = predicted_result[1]
    sentence_result_2 = np.argmax(sentence_result_2, axis=-1)

    indices_to_tag_conll = {5: 'O', 1: 'I-PER', 4: 'I-ORG', 6: 'I-MISC', 0: 'I-LOC', 3: 'B-ORG', 7: 'B-MISC',
                            2: 'B-LOC'}

    print("{:15} {:5}".format("Word", "Pred"))
    print("=" * 25)
    for word, res in zip(words_to_predict, sentence_result):
        if word != "PAD":
            print("{:15}:{:5}".format(word, indices_to_tag_conll[res]))

    print("{:15} {:5}".format("Word", "Pred"))
    print("=" * 25)
    for word, res in zip(words_to_predict_2, sentence_result_2):
        if word != "PAD":
            print("{:15}:{:5}".format(word, indices_to_tag_conll[res]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", default=None, help='One sentence that will go through NER pipeline')
    parser.add_argument("--type", default="CONLL", help='Choose the model. Accepted values: CONLL, GMB')
    parser.add_argument("--show_example", default=False,
                        help='True if you want to see some examples without entering your own sentences')
    args = parser.parse_args()

    print("Please wait until the model is loaded")

    if args.show_example:
        show_examples()
    else:
        if args.type == "GMB" and args.sentence is not None and isinstance(args.sentence, str):
            predict_gmb(args.sentence)
        elif args.sentence is not None and isinstance(args.sentence, str):
            predict_conll(args.sentence)
