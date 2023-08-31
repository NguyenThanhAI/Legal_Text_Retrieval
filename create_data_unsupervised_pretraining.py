import os
import json
import argparse
import random
from tqdm import tqdm
import numpy as np
from utils import tokenizing


def load_json(path, encoding):
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--legal_dict_json", type=str, default="saved_model/legal_dict_tokens.json")
    parser.add_argument("--save_dir", type=str, default="unsupervised_data")
    parser.add_argument("--encoding_mode", default="utf-8", type=str)
    parser.add_argument("--num_doc_chosen", default=10000, type=int)
    parser.add_argument("--num_neg_doc", type=int, default=25)
    parser.add_argument("--window", type=int, default=200)
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    args = get_args()
    
    legal_dict_json = args.legal_dict_json
    save_dir = args.save_dir
    encoding_mode = args.encoding_mode
    num_doc_chosen = args.num_doc_chosen
    num_neg_doc = args.num_neg_doc
    window = args.window
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    doc_data = load_json(path=legal_dict_json, encoding=encoding_mode)
    
    text_doc = list(doc_data.keys())
    
    random.shuffle(text_doc)
    
    chosen_text_doc = text_doc[:num_doc_chosen]
    
    train_data = []
    
    for doc in tqdm(chosen_text_doc):
        query_data = {}
        
        new_text_doc = text_doc.copy()
        new_text_doc.remove(doc)
        
        tokens = doc_data[doc]["text"]
        
        if len(tokens) <= 2 * window:
            seg_len = int(0.7 * len(tokens))
        else:
            seg_len = window
        
        pos_1 = random.randint(0, int(0.2 * len(tokens)))
        pos_2 = random.randint(int(0.4 * len(tokens)), int(0.6 * len(tokens)))
        
        if np.random.rand() < 0.5:
            que = pos_1
            pos = pos_2
        else:
            que = pos_2
            pos = pos_1
            
        
        query_data["query_id"] = doc + "_" + "pos{}".format(que)
        query_data["query"] = " ".join(tokens[que:que+seg_len])
        
        query_data["positive_passages"] = []
        
        save_dict_teva = {}
        save_dict_teva["docid"] = doc + "_" + "pos{}".format(pos)
        save_dict_teva["title"] = doc_data[doc]["title"]
        save_dict_teva["text"] = " ".join(tokens[pos:pos+seg_len])
        
        query_data["positive_passages"].append(save_dict_teva)
        
        neg_doc_list = random.choices(new_text_doc, k=num_neg_doc)
        
        query_data["negative_passages"] = []
        for n_doc in neg_doc_list:
            tokens = doc_data[n_doc]["text"]
            
            if len(tokens) <= int(1.2 * window):
                seg_len = int(0.7 * len(tokens))
            else:
                seg_len = window
                
            pos = random.randint(0, int(0.25 * len(tokens)))
            
            save_dict_teva = {}
            save_dict_teva["docid"] = n_doc + "_" + "pos{}".format(pos)
            save_dict_teva["title"] = doc_data[n_doc]["title"]
            save_dict_teva["text"] = " ".join(tokens[pos:pos+seg_len])
                
            query_data["negative_passages"].append(save_dict_teva)
            
        train_data.append(query_data)
            
    if not os.path.exists(os.path.join(save_dir, "train_dir")):
        os.makedirs(os.path.join(save_dir, "train_dir"), exist_ok=True)
        
    with open(os.path.join(save_dir, "train_dir", "train.jsonl"), "w", encoding=encoding_mode) as f:
        json.dump(train_data, f, indent=3)
        