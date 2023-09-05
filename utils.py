import re
import argparse
import string
from underthesea import word_tokenize
import os
import json

import pickle
import numpy as np

from typing import Dict, List
from tqdm import tqdm

from rank_bm25 import *

from sentence_transformers import SentenceTransformer

number = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
chars = ["a", "b", "c", "d", "đ", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]
stop_word = number + chars + ["của", "và", "các", "có", "được", "theo", "tại", "trong", "về", 
            "hoặc", "người",  "này", "khoản", "cho", "không", "từ", "phải", 
            "ngày", "việc", "sau",  "để",  "đến", "bộ",  "với", "là", "năm", 
            "khi", "số", "trên", "khác", "đã", "thì", "thuộc", "điểm", "đồng",
            "do", "một", "bị", "vào", "lại", "ở", "nếu", "làm", "đây", 
            "như", "đó", "mà", "nơi", "”", "“"]

def remove_stopword(w):
    return w not in stop_word
def remove_punctuation(w):
    return w not in string.punctuation
def lower_case(w):
    return w.lower()

def bm25_tokenizer(text):
    tokens = word_tokenize(text)
    tokens = list(map(lower_case, tokens))
    tokens = list(filter(remove_punctuation, tokens))
    tokens = list(filter(remove_stopword, tokens))
    return tokens

def tokenizing(text):
    tokens = word_tokenize(text)
    tokens = list(map(lower_case, tokens))
    tokens = list(filter(remove_punctuation, tokens))
    return tokens


def calculate_f2(precision, recall):        
    return (5 * precision * recall) / (4 * precision + recall + 1e-20)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def segmentize(sentence: str, stride: int=100, window:int=200) -> Dict[str, str]:
    tokenized_sent = tokenizing(sentence)
    
    derived_sent: Dict[int, str] = {}
    if len(tokenized_sent) <= window:
        derived_sent["pos{}".format(0)] = " ".join(tokenized_sent)
    else:
        for j in range(0, len(tokenized_sent), stride):
            seg = tokenized_sent[j:j+window]
            if len(seg) <= int(0.5 * window):
                continue
            derived_sent["pos{}".format(j)] = " ".join(seg)
            
    return derived_sent

# with open(os.path.join("generated_data", "legal_dict.json"), "r", encoding="utf-8") as f:
#     doc_data = json.load(f)
    
# for k, v in doc_data.items():
#     sentence = doc_data[k]["text"]
#     print("Original: {}".format(sentence))
#     segments = segmentize(sentence=sentence, stride=75, window=150)
#     for seg in segments:
#         print("Segment: {}".format(segments[seg]))


_WORD_SPLIT = re.compile("([.,!?\"/':;)(])")

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
        # return [w.lower() for w in words if w not in stop_words and w != '' and w != ' ']
    return [w.lower() for w in words if w != '' and w != ' ' and w not in string.punctuation and w not in chars]


def create_sliding_window(tokenized_sent: List[str], stride: int=128, window:int=256):
    derived_sent: Dict[int, str] = {}
    if len(tokenized_sent) <= window:
        derived_sent["pos{}".format(0)] = " ".join(tokenized_sent)
    else:
        for j in range(0, len(tokenized_sent), stride):
            seg = tokenized_sent[j:j+window]
            if len(seg) <= int(0.75 * window):
                continue
            derived_sent["pos{}".format(j)] = " ".join(seg)
            
    return derived_sent


def load_json(path, encoding):
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)
    
    
def load_bm25(bm25_path) -> BM25Plus:
    with open(bm25_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    return bm25


def load_encoded_legal_data(encoded_legal_path):
    print("Start loading embedding of legal data")
    with open(encoded_legal_path, "rb") as f:
        emb_legal_data = pickle.load(f)
    return emb_legal_data[0].astype(np.float32)


def load_encoded_question_data(question_encoded_path):
    print("Start loading embedding of question data")
    with open(question_encoded_path, "rb") as f:
        emb_question_data = pickle.load(f)

    return emb_question_data[0].astype(np.float32)

def compute_overall_emb_legal_data(emb_legal_data, passage_id_to_index_group):
    assert emb_legal_data.shape[0] == 115683
    
    emb_legal_data_new = []
    
    for concat_id in tqdm(passage_id_to_index_group):
        group_emb = emb_legal_data[passage_id_to_index_group[concat_id]]
        assert group_emb.shape[0] == len(passage_id_to_index_group[concat_id])
        emb_legal_data_new.append(np.mean(group_emb, axis=0))
        
    emb_legal_data = np.stack(emb_legal_data_new, axis=0)
    assert emb_legal_data.shape[0] == 61425
    
    return emb_legal_data

def compute_overall_score(scores, passage_id_to_index_group):
    assert scores.shape[0] == 115683
    
    scores_new = []
    
    for concat_id in passage_id_to_index_group:
        group_scores = scores[passage_id_to_index_group[concat_id]]
        assert group_scores.shape[0] == len(passage_id_to_index_group[concat_id])
        scores_new.append(np.max(group_scores))
        
    scores = np.stack(scores_new, axis=0)
    assert scores.shape[0] == 61425
    
    return scores


def load_model(model_dir: str, model_path: str) -> SentenceTransformer:
    model_path = os.path.join(model_dir, model_path)
    model = SentenceTransformer(model_name_or_path=model_path)

    return model

def load_legal_data(legal_dict_segments_json, passage_id_to_index_group, encoding_mode="utf-8") -> List[str]:
    doc_data = load_json(legal_dict_segments_json, encoding=encoding_mode)
    
    legal_list = []
    count = 0
    
    for concat_id in doc_data:
        segments = doc_data[concat_id]["text"]
        indices = []
        for segment in segments:
            indices.append(count)
            count += 1
            seg = doc_data[concat_id]["text"][segment]
            legal_list.append(doc_data[concat_id]["title"] + " " + seg)

        #print(passage_id_to_index_group[concat_id], indices)
        assert passage_id_to_index_group[concat_id] == indices

    print("Num laws: {}".format(len(legal_list)))
    
    return legal_list

def load_question_data(questions_path, encoding_mode="utf-8"):
    items = load_json(questions_path, encoding=encoding_mode)["items"]
    
    questions_list = []
    for item in tqdm(items):
        question = item["question"]
        preprocessed_query = " ".join(basic_tokenizer(question))
        questions_list.append(preprocessed_query)

    print("Num questions: {}".format(len(questions_list)))
    
    return questions_list


def encode_text_data(passage_list: List[str], model: SentenceTransformer, batch_size: int=32) -> np.ndarray:
    
    steps = int(np.ceil(len(passage_list)/batch_size))
    emb_legal_data = []
    for i in tqdm(range(steps)):
        text_batch = passage_list[i*batch_size:(i+1)*batch_size]
        emb = model.encode(text_batch)
        emb_legal_data.append(emb)
        
    emb_legal_data = np.stack(emb_legal_data, axis=0)
    
    print("Shape of embedding data: {}".format(emb_legal_data.shape))
    
    return emb_legal_data
# with open(os.path.join("generated_data", "legal_dict.json"), "r", encoding="utf-8") as f:
#     doc_data = json.load(f)
    
# for k, v in doc_data.items():
#     sentence = doc_data[k]["text"]
#     print("Original: {}".format(sentence))
#     tokenized_sent = basic_tokenizer(sentence=sentence)
#     segments = create_sliding_window(tokenized_sent=tokenized_sent, stride=75, window=150)
#     for seg in segments:
#         print("Segment {}: {}".format(seg, segments[seg]))