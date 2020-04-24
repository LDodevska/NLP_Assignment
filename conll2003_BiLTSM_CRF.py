from seqeval.metrics import f1_score, classification_report, accuracy_score
from sklearn_crfsuite.metrics import flat_classification_report
from utils.utils import *

if __name__ == '__main__':

    dataset = "conll2003"
    df, df_test, input_dict, input_dict_test = get_dataset_info(dataset, train_sets="train,eng_a", test_sets="eng_b")

    tags, tags_without_O, words_train, \
    words_test, number_of_words_total, number_of_tags = get_words_and_tags(df, df_test, dataset=dataset)

    word_indices, word_indices_test, \
    tag_indices, indices_to_tag, \
    len_max = get_indices(input_dict, tags, words_train, words_test, dataset, input_dict_test)

    X_train, y_train, X_test, y_test = generate_train_and_test_lists(word_indices, tag_indices, input_dict,
                                                                     input_dict_test, tags, len_max,
                                                                     dataset=dataset)

    embedding_size = 40
    # lstm_units = embedding_size * 2
    lstm_units = 50
    dropout = 0.1
    recurrent_dropout = 0.1
    epochs = 8

    model = create_model(number_of_words_total, number_of_tags, len_max, embedding_size, lstm_units, dropout,
                         recurrent_dropout)

    history, pred_labels, real_labels = run_model(model, X_train, y_train, X_test, y_test, indices_to_tag, epochs)

    with open("../BiLTSM_CRF.log", "a") as file:
        file.write("\n##############################################################\n\n")
        file.write(f"Dataset: {dataset}\n")
        file.write(f"Embedding size: {embedding_size} | Dropout: {dropout} | Recurrent dropout: {recurrent_dropout}\
    | Epochs: {epochs} | LSTM units: {lstm_units} | Train: train, eng_a | Test: eng_b\n")
        file.write("Accuracy: {:.2%}\n".format(accuracy_score(real_labels, pred_labels)))
        file.write("F1-score: {:.2%}\n\n".format(f1_score(real_labels, pred_labels)))

        file.write(classification_report(real_labels, pred_labels))

        report = flat_classification_report(y_pred=pred_labels, y_true=real_labels, labels=tags_without_O)
        file.write(report)
