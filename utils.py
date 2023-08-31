import re
import argparse
import string
from underthesea import word_tokenize
import os
import json

from typing import Dict


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
            if len(seg) <= int(0.3 * window):
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
              