import os
import argparse
import json
import pickle
import numpy as np

from rank_bm25 import *
from utils import load_json, basic_tokenizer

def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--legal_dict_segments", type=str, default="saved_model/legal_dict_segments.json")
    parser.add_argument("--passage_id_to_index_group_path", type=str, default="generated_data/tevatron_passage_id_to_index_group.json")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--encoding_mode", type=str, default="utf-8")
    
    args = parser.parse_args()
    
    return args



if __name__ == "__main__":
    
    args = get_args()
    
    legal_dict_segments = args.legal_dict_segments
    passage_id_to_index_group_path = args.passage_id_to_index_group_path
    save_dir = args.save_dir
    encoding_mode = args.encoding_mode
    
    doc_data = load_json(path=legal_dict_segments, encoding=encoding_mode)
    passage_id_to_index_group = load_json(path=passage_id_to_index_group_path, encoding=encoding_mode)
    
    
    documents = []
    count = 0
    for concat_id in doc_data:
        segments = doc_data[concat_id]["text"]
        
        indices = []
        for segment in segments:
            indices.append(count)
            count += 1
            seg = doc_data[concat_id]["text"][segment]
            tokenized_seg = basic_tokenizer(seg)
            #print(seg, tokenized_seg)
            documents.append(tokenized_seg)
            
        #print(passage_id_to_index_group[concat_id], indices)
        assert passage_id_to_index_group[concat_id] == indices
        
        
    print("Num segments: {}".format(len(documents)))
        
        
    bm25 = BM25Plus(corpus=documents, k1=0.4, b=0.6)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    with open(os.path.join(save_dir, "bm25_Plus_segments"), "wb") as f:
        pickle.dump(bm25, f)
        