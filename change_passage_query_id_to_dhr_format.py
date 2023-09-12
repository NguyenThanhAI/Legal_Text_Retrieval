import os
import argparse
import json
import pickle
from tqdm import tqdm
import numpy as np
from rank_bm25 import BM25Plus, BM25Okapi, BM25L, BM25

from transformers import AutoTokenizer

from utils import load_json


def string_to_word_id(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    tokens_id = tokenizer.convert_tokens_to_ids(tokens)
    
    return tokens_id


def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--corpus_path", type=str, default="./generated_data/tevatron_data/corpus/corpus.jsonl")
    parser.add_argument("--train_path", type=str, default="./generated_data/tevatron_data/train_dir/train.jsonl")
    parser.add_argument("--test_queries_path", type=str, default="./generated_data/tevatron_data/dev_dir/train.jsonl")
    parser.add_argument("--train_queries_path", type=str, default="./generated_data/tevatron_data/train_dir_origin/train.jsonl")
    parser.add_argument("--encoding_mode", type=str, default="utf-8")
    parser.add_argument("--save_dir", type=str, default="dhr_dataset")
    
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = get_args()
    
    corpus_path = args.corpus_path
    train_path = args.train_path
    test_queries_path = args.test_queries_path
    train_queries_path = args.train_queries_path
    encoding_mode = args.encoding_mode
    save_dir = args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        
    corpus = load_json(corpus_path, encoding=encoding_mode)
    train_json = load_json(train_path, encoding=encoding_mode)
    train_queries = load_json(train_queries_path, encoding=encoding_mode)
    test_queries = load_json(test_queries_path, encoding=encoding_mode)
    
    if not os.path.exists(os.path.join(save_dir, "corpus")):
        os.makedirs(os.path.join(save_dir, "corpus"), exist_ok=True)
            
    if not os.path.exists(os.path.join(save_dir, "train_dir")):
        os.makedirs(os.path.join(save_dir, "train_dir"), exist_ok=True)
        
    if not os.path.exists(os.path.join(save_dir, "original_train_dir")):
        os.makedirs(os.path.join(save_dir, "original_train_dir"), exist_ok=True)
        
    if not os.path.exists(os.path.join(save_dir, "dev_dir")):
        os.makedirs(os.path.join(save_dir, "dev_dir"), exist_ok=True)
        
    print("Save for corpus")
    count = 0
    passage_original_id_to_int = {}
    data = []
    for passage in tqdm(corpus):
        save_dict = {}
        passage_id = passage["docid"]
        new_id = str(count)
        passage_original_id_to_int[passage_id] = new_id
        save_dict["docid"] = new_id
        save_dict["title"] = passage["title"]
        save_dict["text"] = passage["text"]
        data.append(save_dict)
        count += 1
        
    with open(os.path.join(save_dir, "corpus", "corpus.jsonl"), "w", encoding=encoding_mode) as f:
        json.dump(data, f, indent=4)
        
    print("Save for positive and negatives")
    train_data = []
    count = 0
    query_original_id_to_int = {}
    for que in tqdm(train_json):
        query_text = que["query"]
        query_id = que["query_id"]
        save_json = {}
        new_id = str(count)
        query_original_id_to_int[query_id] = new_id
        save_json["query_id"] = new_id
        save_json["query"] = query_text
        save_json["positive_passages"] = []
        for pos in que["positive_passages"]:
            save_json["positive_passages"].append({"docid": passage_original_id_to_int[pos["docid"]],
                                                   "title": pos["title"],
                                                   "text": pos["text"]})
        save_json["negative_passages"] = []
        for neg in que["negative_passages"]:
            save_json["negative_passages"].append({"docid": passage_original_id_to_int[neg["docid"]],
                                                   "title": neg["title"],
                                                   "text": neg["text"]})
        train_data.append(save_json)
        count += 1
        
    with open(os.path.join(save_dir, "train_dir", "train.jsonl"), "w", encoding=encoding_mode) as f:
        json.dump(train_data, f, indent=4)
        
    print("Save for train queries")
    train_data = []
    for que in tqdm(train_queries):
        save_json = {}
        query_text = que["query"]
        query_id = que["query_id"]
        new_id = query_original_id_to_int[query_id]
        save_json["query_id"] = new_id
        save_json["query"] = query_text
        save_json["positive_passages"] = que["positive_passages"]
        save_json["negative_passages"] = que["negative_passages"]
        train_data.append(save_json)
        
    with open(os.path.join(save_dir, "original_train_dir", "train.jsonl"), "w", encoding=encoding_mode) as f:
        json.dump(train_data, f, indent=4)
        
    print("Save for test queries")
    test_data = []
    for que in tqdm(test_queries):
        save_json = {}
        query_text = que["query"]
        #query_id = que["query_id"]
        query_id = str(count)
        save_json["query_id"] = query_id
        save_json["query"] = query_text
        save_json["positive_passages"] = que["positive_passages"]
        save_json["negative_passages"] = que["negative_passages"]
        test_data.append(save_json)
        count += 1
        
    with open(os.path.join(save_dir, "dev_dir", "train.jsonl"), "w", encoding=encoding_mode) as f:
        json.dump(test_data, f, indent=4)
        
        