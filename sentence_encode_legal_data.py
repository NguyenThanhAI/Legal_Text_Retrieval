from typing import List, Dict

import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import os
import pickle
import glob

from sentence_transformers import SentenceTransformer, util

from utils import load_legal_data, load_model, load_json, encode_text_data, load_question_data
def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--legal_dict_segments", type=str, default="saved_model/legal_dict_segments.json")
    parser.add_argument("--train_question_path", type=str, default=None)
    parser.add_argument("--test_question_path", type=str, default=None)
    parser.add_argument("--passage_id_to_index_group_path", type=str, default="generated_data/tevatron_passage_id_to_index_group.json")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--encoding_mode", type=str, default="utf-8")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--models_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    
    args = get_args()
    
    legal_dict_segments = args.legal_dict_segments
    train_question_path = args.train_question_path
    test_question_path = args.test_question_path
    passage_id_to_index_group_path = args.passage_id_to_index_group_path
    save_dir = args.save_dir
    encoding_mode = args.encoding_mode
    model_dir = args.model_dir
    models_name = args.models_name.split(",")
    batch_size = args.batch_size
    
    #model_paths = list(map(lambda x: os.path.join(model_dir, x), models_name))
    
    
    passage_id_to_index_group = load_json(path=passage_id_to_index_group_path, encoding=encoding_mode)
    
    print("Load legal data and question")
    legal_list = load_legal_data(legal_dict_segments_json=legal_dict_segments, passage_id_to_index_group=passage_id_to_index_group,
                                 encoding_mode=encoding_mode)
    
    
    train_question_list = load_question_data(questions_path=train_question_path, encoding_mode=encoding_mode)
    test_question_list = load_question_data(questions_path=test_question_path, encoding_mode=encoding_mode)
    
    for name in models_name:
        print("Load model {}".format(name))
        model = load_model(model_dir=model_dir, model_path=name)
        
        print("Encode legal data model: {}".format(name))
        emb_legal_data = encode_text_data(passage_list=legal_list, model=model, batch_size=batch_size)
        print("Encode train questions model: {}".format(name))
        emb_train_question = encode_text_data(passage_list=train_question_list, model=model, batch_size=batch_size)
        print("Encode test question model: {}".format(name))
        emb_test_question = encode_text_data(passage_list=test_question_list, model=model, batch_size=batch_size)
        
        dir_to_save = os.path.join(save_dir, name)
        if not os.path.exists(dir_to_save):
            os.makedirs(dir_to_save, exist_ok=True)
        
        with open(os.path.join(dir_to_save, "{}_corpus_emb.pkl".format(name)), "wb") as f:
            pickle.dump(emb_legal_data, f)
            
        with open(os.path.join(dir_to_save, "{}_train_question_emb.pkl".format(name)), "wb") as f:
            pickle.dump(emb_train_question, f)
            
        with open(os.path.join(dir_to_save, "{}_test_question_emb.pkl".format(name)), "wb") as f:
            pickle.dump(emb_test_question, f)
            