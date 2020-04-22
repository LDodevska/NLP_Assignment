from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import re
import pandas as pd
from collections import Counter


def parse_dataset(file_name, save_file=False):
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
        results[i] = tokens
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


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )


def train_and_evaluate(x_train, y_train, x_test, y_test, example_sent, c1=0.1, c2=1e-3, max_iterations=100,
                       output_file='conll2003-eng.crfsuite'):
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': c1,  # coefficient for L1 penalty
        'c2': c2,  # coefficient for L2 penalty
        'max_iterations': max_iterations,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(output_file)
    tagger = pycrfsuite.Tagger()
    tagger.open(output_file)
    y_pred = [tagger.tag(xseq) for xseq in x_test]

    print(' '.join(sent2tokens(example_sent)), end='\n\n')

    print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    print("Correct:  ", ' '.join(sent2labels(example_sent)))

    print(bio_classification_report(y_test, y_pred))

    info = tagger.info()
    print("Top likely transitions:")
    print_transitions(Counter(info.transitions).most_common(10))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(info.transitions).most_common()[-10:])

    print("\nTop positive:")
    print_state_features(Counter(info.state_features).most_common(10))

    print("\nTop negative:")
    print_state_features(Counter(info.state_features).most_common()[-10:])


if __name__ == '__main__':
    df, input_dict = parse_dataset("./eng.train")
    df_test, input_dict_test = parse_dataset("./eng.testa")
    x_train = []
    y_train = []
    for idx, sentence in input_dict.items():
        x_train.append(sent2features(sentence))
        y_train.append(sent2labels(sentence))
    x_test = []
    y_test = []
    for idx, sentence in input_dict_test.items():
        x_test.append(sent2features(sentence))
        y_test.append(sent2labels(sentence))
    train_and_evaluate(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, example_sent=input_dict_test[1])
