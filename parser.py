import re
import json
import pandas as pd
import argparse

def parse_dataset(file_name, save_file=True):
    with open(file_name, "r") as file:
        content = file.read()
    find_sentences = re.compile(r'(.*?)(?:\n{2})', re.MULTILINE | re.DOTALL)
    sentences = find_sentences.findall(content)
    parse_sentence = re.compile(r"(\S+)\s(\S+)\s\S+\s(\S+)")
    df = pd.DataFrame(columns=["sentence_idx", "sentences", "tags"])
    comb_sentences = []
    all_tags = []
    idxs = []
    for i, s in enumerate(sentences):
        tokens = parse_sentence.findall(s)
        curr_sentence = []
        curr_tags = []
        for token in tokens:
            word, pos, tag = token
            curr_sentence.append(word)
            curr_tags.append(tags[tag])
        idxs.append(i)
        comb_sentences.append(curr_sentence)
        all_tags.append(curr_tags)
    df["sentence_idx"] = idxs
    df["sentences"] = comb_sentences
    
    df["tags"] = all_tags
    if save_file:
        df.to_json("parsed_sl_{}.json".format(file_name.split(".")[-1]), orient='records')
    return df


def parse_ner_dataset(file_name, save_file=True):
    ner_df = pd.read_csv(file_name)
    ner_df = ner_df[['sentence_idx', 'word', 'tag']]
    
    df_grouped = ner_df.groupby('sentence_idx')
    
    comb_sentences = []
    final_tags = []
    idxs = []
    
    for s_group in df_grouped:
        sentence_group = s_group[1]
        
        s_idx = sentence_group['sentence_idx'].iloc[0]
        words = sentence_group['word'].tolist()
        tags = sentence_group['tag'].tolist()
        
        comb_sentences.append(words)
        final_tags.append(tags)
        idxs.append(s_idx)
    
    df = pd.DataFrame(columns=["sentence_idx", "sentences", "tags"])
    df["sentence_idx"] = idxs
    df["sentences"] = comb_sentences
    df["tags"] = final_tags
    
    if save_file:
        df.to_json("parsed_gmb_bert.json", orient='records')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help='choose which dataset to parse conll or gmb')
    parser.add_argument("--file_name", help='the path to the dataset file that you want to parse')
    args = parser.parse_args()
    if args.dataset == "conll":
    	parse_dataset(args.file_name, True)
    elif args.dataset == "gmb":
    	parse_ner_dataset(args.file_name, True)
    
    
 
