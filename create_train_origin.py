import os
import argparse
import json

from tqdm import tqdm

from utils import basic_tokenizer

def load_json(path, encoding):
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--legal_dict_segments", type=str, default=r"F:\Programming\PythonProjects\Natural_Language_Processing\Legal_Text_Retrieval\saved_model\legal_dict_segments.json")
    parser.add_argument("--data_dir", type=str, default=r"E:\ZALO2021\zac2021-ltr-data\zac2021-ltr-data")
    parser.add_argument("--save_dir", type=str, default="generated_data")
    parser.add_argument("--encoding_mode", default="utf-8", type=str)
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    args = get_args()
    
    legal_dict_segments = args.legal_dict_segments
    data_dir = args.data_dir
    save_dir = args.save_dir
    encoding_mode = args.encoding_mode
    
    train_path = os.path.join(data_dir, "for_train_question_answer.json")
    training_items = load_json(train_path, encoding=encoding_mode)["items"]
    
    doc_data = load_json(legal_dict_segments, encoding=encoding_mode)
    
    save_pairs = []
    train_data = []
    for idx, item in tqdm(enumerate(training_items)):
        query_data = {}
        
        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        preprocessed_query = " ".join(basic_tokenizer(question))
        
        query_data["query_id"] = question_id
        query_data["query"] = preprocessed_query
        
        query_data["positive_passages"] = []
        
        query_data["negative_passages"] = []
        
        train_data.append(query_data)
        
    if not os.path.exists(os.path.join(save_dir, "tevatron_data", "train_dir_origin")):
        os.makedirs(os.path.join(save_dir, "tevatron_data", "train_dir_origin"), exist_ok=True)
        
    with open(os.path.join(save_dir, "tevatron_data", "train_dir_origin", "train.jsonl"), "w", encoding=encoding_mode) as f:
        json.dump(train_data, f, indent=3)