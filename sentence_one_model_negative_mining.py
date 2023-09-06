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
from utils import bm25_tokenizer, calculate_f2, basic_tokenizer, load_bm25, load_json, compute_overall_emb_legal_data, compute_overall_score

from sentence_transformers import util


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default=r"E:\ZALO2021\zac2021-ltr-data\zac2021-ltr-data")
    parser.add_argument("--train_json_path", type=str, default="for_train_question_answer.json")
    parser.add_argument("--legal_dict_json_segments", default="saved_model/legal_dict_segments.json", type=str)
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--doc_refers_path", default="saved_model/doc_refers_saved", type=str)
    parser.add_argument("--encoded_data_dir", type=str, default=r"F:\Programming\PythonProjects\Natural_Language_Processing\embeddings\sentence_bert_segments_round_1")
    parser.add_argument("--model_name", type=str, default="phobert-base")
    parser.add_argument("--range_score", default=2.6, type=float)
    parser.add_argument("--passage_id_to_index_group_path", type=str, default="generated_data/tevatron_passage_id_to_index_group.json")
    parser.add_argument("--encoding_mode", default="utf-8", type=str)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--num_neg_segments_per_passage", type=int, default=5)
    parser.add_argument("--round_turn", type=int, default=1)
    parser.add_argument("--save_dir", default=r"F:\Programming\PythonProjects\Natural_Language_Processing\sentence_bert_segments_round_1_negative_mining", type=str)
    
    args = parser.parse_args()
    
    return args

def load_encoded_legal_data(encoded_legal_path):
    print("Start loading embedding of legal data")
    with open(encoded_legal_path, "rb") as f:
        emb_legal_data = pickle.load(f)
    return emb_legal_data


def load_encoded_question_data(question_encoded_path):
    print("Start loading embedding of question data")
    with open(question_encoded_path, "rb") as f:
        emb_question_data = pickle.load(f)

    return emb_question_data


if __name__ == "__main__":
    
    args = get_args()
    
    data_dir = args.data_dir
    train_json_path = args.train_json_path
    legal_dict_json_segments = args.legal_dict_json_segments
    bm25_path = args.bm25_path
    doc_refers_path = args.doc_refers_path
    encoded_data_dir = args.encoded_data_dir
    model_name = args.model_name
    range_score = args.range_score
    passage_id_to_index_group_path = args.passage_id_to_index_group_path
    encoding_mode = args.encoding_mode
    top_n = args.top_n
    num_neg_segments_per_passage = args.num_neg_segments_per_passage
    round_turn = args.round_turn
    save_dir = args.save_dir
    
    train_path = os.path.join(data_dir, train_json_path)
    data = json.load(open(train_path))
    items = data["items"]
    print("Num train question: {}".format(len(items)))
    
    print("Load BM25 model")
    bm25 = load_bm25(bm25_path=bm25_path)
    
    print("Load legal dict segments")
    doc_data = load_json(legal_dict_json_segments, encoding=encoding_mode)
        
    print("Load doc refer")
    with open(doc_refers_path, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)
        
    passage_id_to_index_group = load_json(path=passage_id_to_index_group_path, encoding=encoding_mode)
        
    encoded_legal_path = os.path.join(encoded_data_dir, "sentence-bert-contrastive-{}-segments-round-{}".format(model_name, round_turn), "sentence-bert-contrastive-{}-segments-round-{}_corpus_emb.pkl".format(model_name, round_turn))
    question_encoded_path = os.path.join(encoded_data_dir, "sentence-bert-contrastive-{}-segments-round-{}".format(model_name, round_turn), "sentence-bert-contrastive-{}-segments-round-{}_train_question_emb.pkl".format(model_name, round_turn))
    
    emb_legal_data = load_encoded_legal_data(encoded_legal_path=encoded_legal_path)
    question_emb_dict = load_encoded_question_data(question_encoded_path=question_encoded_path)
    
    print("Shape of question embedding: {}".format(question_emb_dict.shape))
    # assert emb_legal_data.shape[0] == 115683
    
    # emb_legal_data_new = []
    
    # for concat_id in tqdm(passage_id_to_index_group):
    #     group_emb = emb_legal_data[passage_id_to_index_group[concat_id]]
    #     assert group_emb.shape[0] == len(passage_id_to_index_group[concat_id])
    #     emb_legal_data_new.append(np.mean(group_emb, axis=0))
        
    # emb_legal_data = np.stack(emb_legal_data_new, axis=0)
    # assert emb_legal_data.shape[0] == 61425
    
    #emb_legal_data = compute_overall_emb_legal_data(emb_legal_data=emb_legal_data, passage_id_to_index_group=passage_id_to_index_group)
    
    save_pairs = []
    for idx, item in tqdm(enumerate(items)):
        
        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        preprocessed_query = " ".join(basic_tokenizer(question))
        
        quest_emb = question_emb_dict[idx]
        scores = util.cos_sim(torch.from_numpy(quest_emb), torch.from_numpy(emb_legal_data))
        scores = scores.squeeze(0).numpy()
        scores = compute_overall_score(scores=scores, passage_id_to_index_group=passage_id_to_index_group, ensemble_type="mean")
        new_scores = scores
        
        predictions = np.argsort(new_scores)[::-1][:top_n]
        
        for article in relevant_articles:
            concat_id = article["law_id"] + "_" + article["article_id"]
            segments = doc_data[concat_id]["text"]
            
            for seg in segments:
                save_dict = {}
                save_dict["question"] = preprocessed_query
                save_dict["document"] = doc_data[concat_id]["title"] + " " + segments[seg]
                save_dict["relevant"] = 1
                save_pairs.append(save_dict)
                
        
        # Save negative pairs
        for idx, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]
            
            check = 0
            for article in relevant_articles:
                if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                    check += 1
                    break
            
            if check == 0:
                concat_id = pred[0] + "_" + pred[1]
                segments = doc_data[concat_id]["text"]
                
                seg_list = list(segments.keys())
                if len(seg_list) > num_neg_segments_per_passage:
                    chosen_seg = np.random.choice(seg_list, size=num_neg_segments_per_passage, replace=False)
                else:
                    chosen_seg = seg_list
                    
                for seg in chosen_seg:
                    
                    save_dict = {}
                    save_dict["question"] = preprocessed_query
                    save_dict["document"] = doc_data[concat_id]["title"] + " " + segments[seg]
                    save_dict["relevant"] = 0
                    save_pairs.append(save_dict)
                    
    
    save_dir = os.path.join(save_dir, model_name, "sentence_bert_{}_segments_round_{}".format(model_name, round_turn))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    with open(os.path.join(save_dir, "sentence_bert_{}_segments_round_{}_bm25_pairs_top{}.pkl".format(model_name, round_turn, top_n)), "wb") as f:
        pickle.dump(save_pairs, f)