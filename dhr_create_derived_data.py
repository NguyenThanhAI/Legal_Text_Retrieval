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
        
    models_to_code = {"phobert_base": "vinai/phobert-base",
                      "condenser_phobert_base": "NlpHUST/Condenser-phobert-base",
                      "phobert_large": "vinai/phobert-large",
                      "vibert_base_cased": "FPTAI/vibert-base-cased"}
    
    
    corpus = load_json(corpus_path, encoding=encoding_mode)
    train_json = load_json(train_path, encoding=encoding_mode)
    train_queries = load_json(train_queries_path, encoding=encoding_mode)
    test_queries = load_json(test_queries_path, encoding=encoding_mode)
    
    for model in models_to_code.keys():
        print("Models: {}".format(model))
        tokenizer = AutoTokenizer.from_pretrained(models_to_code[model])
        
        if not os.path.exists(os.path.join(save_dir, model, "corpus")):
            os.makedirs(os.path.join(save_dir, model, "corpus"), exist_ok=True)
            
        if not os.path.exists(os.path.join(save_dir, model, "train")):
            os.makedirs(os.path.join(save_dir, model, "train"), exist_ok=True)
            
        if not os.path.exists(os.path.join(save_dir, model, "train_queries")):
            os.makedirs(os.path.join(save_dir, model, "train_queries"), exist_ok=True)
            
        if not os.path.exists(os.path.join(save_dir, model, "test_queries")):
            os.makedirs(os.path.join(save_dir, model, "test_queries"), exist_ok=True)
            
        corp_f = open(os.path.join(save_dir, model, "corpus", "corpus.jsonl"), "w", encoding=encoding_mode)
        train_que_f = open(os.path.join(save_dir, model, "train_queries", "train.jsonl"), "w", encoding=encoding_mode)
        test_que_f = open(os.path.join(save_dir, model, "test_queries", "train.jsonl"), "w", encoding=encoding_mode)
        train_json_f = open(os.path.join(save_dir, model, "train", "train.jsonl"), "w", encoding=encoding_mode)
        print("Save for corpus")
        count = 0
        passage_original_id_to_int = {}
        for passage in tqdm(corpus):
            passage_id = passage["docid"]
            new_id = str(count)
            passage_original_id_to_int[passage_id] = new_id
            vocab_ids = string_to_word_id(tokenizer=tokenizer, text=passage["text"])
            corp_f.write(json.dumps({"text_id": new_id, "text": vocab_ids}) + "\n")
            count += 1
        
        
        corp_f.close()
        print("Save for positive and negatives")
        for que in tqdm(train_json):
            query_text = que["query"]
            save_json = {}
            save_json["query"] = string_to_word_id(tokenizer=tokenizer, text=query_text)
            save_json["positive_pids"] = []
            for passage in que["positive_passages"]:
                save_json["positive_pids"].append(passage_original_id_to_int[passage["docid"]])
            
            save_json["negative_pids"] = []
            for passage in que["negative_passages"]:
                save_json["negative_pids"].append(passage_original_id_to_int[passage["docid"]])
                
            train_json_f.write(json.dumps(save_json) + "\n")
            
        train_json_f.close()
        
        print("Save for train queries")
        
        count = 0
        for que in tqdm(train_queries):
            query_text = que["query"]
            #query_id = que["query_id"]
            query_id = str(count)
            vocab_ids = string_to_word_id(tokenizer=tokenizer, text=query_text)
            train_que_f.write(json.dumps({"text_id": query_id, "text": vocab_ids}) + "\n")
            count += 1
        
        train_que_f.close()
            
        print("Save for test queries")
        
        for que in tqdm(test_queries):
            query_text = que["query"]
            #query_id = que["query_id"]
            query_id = str(count)
            vocab_ids = string_to_word_id(tokenizer=tokenizer, text=query_text)
            test_que_f.write(json.dumps({"text_id": query_id, "text": vocab_ids}) + "\n")
            count += 1
        
        test_que_f.close()
                    