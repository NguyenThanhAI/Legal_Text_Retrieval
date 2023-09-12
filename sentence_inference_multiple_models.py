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
from utils import bm25_tokenizer, calculate_f2, load_bm25, load_json, compute_overall_emb_legal_data, compute_overall_score, basic_tokenizer, load_json

from sentence_transformers import util


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default=r"E:\ZALO2021\zac2021-ltr-data\zac2021-ltr-data")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--legal_dict_json", default="generated_data/legal_dict.json", type=str)
    parser.add_argument("--bm25_path", default="./saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--doc_refers_path", default="saved_model/doc_refers_saved", type=str)
    parser.add_argument("--encoded_data_dir", type=str, default=r"F:\Programming\PythonProjects\Natural_Language_Processing\embeddings\sentence_bert_segments_round_2")
    parser.add_argument("--model_names", type=str, default="sentence-bert-contrastive-phobert-base-segments-round-2,sentence-bert-contrastive-condenser-phobert-base-segments-round-2,sentence-bert-contrastive-vibert-base-cased-segments-round-2")
    parser.add_argument("--range_score", default=2.6, type=float)
    parser.add_argument("--passage_id_to_index_group_path", type=str, default="generated_data/tevatron_passage_id_to_index_group.json")
    parser.add_argument("--encoding_mode", default="utf-8", type=str)
    parser.add_argument("--weights", type=str, default="0.4,0.4,0.2")
    parser.add_argument("--question", type=str, default="Khi ly hôn tài sản chia như thế nào")
    
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
    model_dir = args.model_dir
    legal_dict_json = args.legal_dict_json
    bm25_path = args.bm25_path
    doc_refers_path = args.doc_refers_path
    encoded_data_dir = args.encoded_data_dir
    model_names = args.model_names.split(",")
    range_score = args.range_score
    passage_id_to_index_group_path = args.passage_id_to_index_group_path
    encoding_mode = args.encoding_mode
    weights = list(map(lambda x: float(x), args.weights.split(",")))
    question = args.question
    
    
    print("Load BM25 model")
    bm25 = load_bm25(bm25_path=bm25_path)
    
    print("Load legal dict")
    doc_data = load_json(legal_dict_json, encoding=encoding_mode)
    
    print("Load doc refer")
    with open(doc_refers_path, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)
        
    passage_id_to_index_group = load_json(path=passage_id_to_index_group_path, encoding=encoding_mode)
    
    emb_legal_data: Dict[str, np.ndarray] = {}
    question_emb_dict: Dict[str, np.ndarray] = {}
    print("Load embedding of corpus")
    top_n = 61425
    #top_n = 115683
    
    for model_name in tqdm(model_names):
        encoded_legal_path = os.path.join(encoded_data_dir, model_name, "{}_corpus_emb.pkl".format(model_name))
        emb_legal_data[model_name] = load_encoded_legal_data(encoded_legal_path=encoded_legal_path)
        
        #emb_legal_data[model_name] = compute_overall_emb_legal_data(emb_legal_data=emb_legal_data[model_name], passage_id_to_index_group=passage_id_to_index_group)
    
    print("Loading models")
    
    models_dict = {}
    for model_name in tqdm(model_names):
        print("Load model {}".format(model_name))
        model = load_model(model_dir=model_dir, model_path=model_name)
        models_dict[model_name] = model
        
    
    tokenized_query = bm25_tokenizer(question)
    doc_scores = bm25.get_scores(tokenized_query)
    
    preprocessed_query = " ".join(basic_tokenizer(question))

    cos_sim = []

    for idx_2, model_path in enumerate(model_paths):
        emb1 = models_dict[model_path].encode(preprocessed_query)
        emb2 = emb_legal_data[model_path]

        scores = util.cos_sim(torch.from_numpy(emb1), torch.from_numpy(emb2))
        scores = scores.squeeze(0).numpy()
        scores = torch.from_numpy(scores).unsqueeze(0)
        cos_sim.append(weights[idx_2] * scores)

    cos_sim = torch.cat(cos_sim, dim=0)

    cos_sim = torch.sum(cos_sim, dim=0).squeeze(0).numpy()
    new_scores = doc_scores * cos_sim
    max_score = np.max(new_scores)
    
    #predictions = np.argpartition(new_scores, len(new_scores) - top_n)[-top_n:]
    predictions = np.argsort(new_scores)[::-1][:top_n]
    new_scores = new_scores[predictions]
    
    new_predictions = np.where(new_scores >= (max_score - range_score))[0]
    
    map_ids = predictions[new_predictions]
    
    new_scores = new_scores[new_scores >= (max_score - range_score)]
    
    if new_scores.shape[0] > 5:
        #predictions_2 = np.argpartition(new_scores, len(new_scores) - 5)[-5:]
        predictions_2 = np.argsort(new_scores)[::-1][:5]
        map_ids = map_ids[predictions_2]
    
    # true_positive = 0
    # false_positive = 0
    for idx_3, idx_pred in enumerate(map_ids):
        pred = doc_refers[idx_pred]
        concat_id = pred[0] + "_" + pred[1]
        print("ID: {}".format(concat_id))
        print("Nội dung: {}".format(doc_data[concat_id]["title"] + " " + doc_data[concat_id]["text"]))
        print("=====================================================================================")