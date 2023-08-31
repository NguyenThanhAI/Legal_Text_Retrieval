import os
import argparse
import json
import pickle
from tqdm import tqdm
import numpy as np
from rank_bm25 import BM25Plus, BM25Okapi, BM25L, BM25

from utils import bm25_tokenizer, calculate_f2, str2bool, segmentize, tokenizing


class Config:
    save_bm25 = "saved_model"
    num_eval = 500
    top_k_bm25 = 2
    bm25_k1 = 0.4
    bm25_b = 0.6

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"E:\ZALO2021\zac2021-ltr-data\zac2021-ltr-data", type=str, help="directory of training data")
    parser.add_argument("--save_dir", default="./generated_data", type=str, help="directory of training data")
    parser.add_argument("--encoding_mode", default="utf-8", type=str)
    parser.add_argument("--load_docs", type=str2bool, default=True)
    parser.add_argument("--top_n", type=int, default=25)
    parser.add_argument("--stride", type=int, default=75)
    parser.add_argument("--window", type=int, default=150)
    args = parser.parse_args()
    
    return args


def load_json(path, encoding):
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)
    
    
if __name__ == "__main__":
    
    args = get_args()
    
    data_dir = args.data_dir
    save_dir = args.save_dir
    encoding_mode = args.encoding_mode
    load_docs = args.load_docs
    top_n = args.top_n
    stride = args.stride
    window = args.window
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print("==========================Create corpus.txt and legal_dict.json==========================================")\
    
    cp = open(os.path.join(save_dir, "corpus.txt"), "w", encoding=encoding_mode)
    corpus_path = os.path.join(data_dir, "legal_corpus.json")
    
    data = load_json(path=corpus_path, encoding=encoding_mode)
    
    save_dict = {}
    
    count = 0
    
    for law_article in tqdm(data):
        law_id = law_article["law_id"]
        law_articles = law_article["articles"]
        
        for sub_article in law_articles:
            article_id = sub_article["article_id"]
            article_title = sub_article["title"]
            article_title = article_title.lower()
            article_text = sub_article["text"]
            article_text = article_text.lower()
            article_full = article_title + ". " + article_text
            article_full = article_full.replace("\n", " ")
            cp.write(article_full + "\n")
            
            concat_id = law_id + "_" + article_id
            if concat_id not in save_dict:
                count += 1
                save_dict[concat_id] = {"title": article_title, "text": article_text}
                
    print("Num law: {}".format(count))
    print("Create legal dict from raw data")
    with open(os.path.join(save_dir, "legal_dict.json"), "w", encoding=encoding_mode) as outfile:
        json.dump(save_dict, outfile, indent=3)
        
    corpus_path_train = os.path.join(data_dir, "train_question_answer.json")
    items = load_json(path=corpus_path_train, encoding=encoding_mode)["items"]
    
    for item in tqdm(items):
        question = item["question"]
        question = question.lower()
        cp.write(question + "\n")
        
    corpus_path_test = os.path.join(data_dir, "public_test_question.json")
    items = load_json(corpus_path_test, encoding=encoding_mode)["items"]
    
    for item in tqdm(items):
        question = item["question"]
        question = question.lower()
        cp.write(question + "\n")
        
    corpus_path_test = os.path.join(data_dir, "private_test_question.json")
    items = load_json(corpus_path_test, encoding=encoding_mode)["items"]
    
    for item in tqdm(items):
        question = item["question"]
        question = question.lower()
        cp.write(question + "\n")
        
    cp.close()
    
    
    print("===================================Train BM25=============================================")
    
    cfg = Config()
    
    save_path = cfg.save_bm25
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    corpus_path = os.path.join(data_dir, "legal_corpus.json")
    data = load_json(path=corpus_path, encoding=encoding_mode)
    
    
    print("Process documents")
    
    if load_docs:
        documents = []
        doc_refers = []
        doc_refers_segments = []
        save_dict_tokens = {}
        save_dict_segments = {}

        for law_article in tqdm(data):
            law_id = law_article["law_id"]
            law_articles = law_article["articles"]

            for sub_article in law_articles:
                article_id = sub_article["article_id"]
                article_title = sub_article["title"]
                article_title = article_title.lower()
                article_text = sub_article["text"]
                article_text = article_text.lower()
                article_full = article_title + ". " + article_text

                tokens = bm25_tokenizer(article_full)
                documents.append(tokens)
                doc_refers.append([law_id, article_id, article_full])
                tokenized_sent = tokenizing(article_text)
                segmented = segmentize(sentence=article_text, stride=stride, window=window)
                doc_refers_segments.append([law_id, article_id, article_title, segmented])
                concat_id = law_id + "_" + article_id
                if concat_id not in save_dict_segments:
                    save_dict_segments[concat_id] = {"title": article_title, "text": segmented}
                if concat_id not in save_dict_tokens:
                    save_dict_tokens[concat_id] = {"title": article_title, "text": tokenized_sent}

        with open(os.path.join(save_path, "documents_manual"), "wb") as documents_file:
            pickle.dump(documents, documents_file)
        with open(os.path.join(save_path,"doc_refers_saved"), "wb") as doc_refer_file:
            pickle.dump(doc_refers, doc_refer_file)
        with open(os.path.join(save_path, "doc_refers_segments"), "wb") as f:
            pickle.dump(doc_refers_segments, f)
        with open(os.path.join(save_path, "legal_dict_segments.json"), "w", encoding=encoding_mode) as f:
            json.dump(save_dict_segments, f, indent=3)
        with open(os.path.join(save_path, "legal_dict_tokens.json"), "w", encoding=encoding_mode) as f:
            json.dump(save_dict_tokens, f, indent=3)
    else:
        with open(os.path.join(save_path, "documents_manual"), "rb") as documents_file:
            documents = pickle.load(documents_file)
        with open(os.path.join(save_path,"doc_refers_saved"), "rb") as doc_refer_file:
            doc_refers = pickle.load(doc_refer_file)
        with open(os.path.join(save_path, "doc_refers_segments"), "rb") as f:
            doc_refers_segments = pickle.load(f)
        with open(os.path.join(save_path, "legal_dict_segments.json"), "r", encoding=encoding_mode) as f:
            save_dict_segments = json.load(f)
        with open(os.path.join(save_path, "legal_dict_tokens.json"), "r", encoding=encoding_mode) as f:
            save_dict_tokens = json.load(f)
        
    train_path = os.path.join(data_dir, "for_train_question_answer.json")
    data = load_json(path=train_path, encoding=encoding_mode)
    items = data["items"]
    
    print("Num train questions: {}".format(len(items)))
    
    bm25 = BM25Plus(documents, k1=cfg.bm25_k1, b=cfg.bm25_b)
    with open(os.path.join(save_path, "bm25_Plus_04_06_model_full_manual_stopword"), "wb") as bm_file:
        pickle.dump(bm25, bm_file)
        
    total_f2 = 0
    total_precision = 0
    total_recall = 0
    
    k = cfg.num_eval
    for idx, item in tqdm(enumerate(items)):
        if idx >= k:
            continue

        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Get top N
        # N large -> reduce precision, increase recall
        # N small -> increase precision, reduce recall
        predictions = np.argsort(doc_scores)[::-1][:cfg.top_k_bm25]
        
        # Trick to balance precision and recall
        if doc_scores[predictions[1]] - doc_scores[predictions[0]] >= 2.7:
            predictions = [predictions[1]]

        true_positive = 0
        false_positive = 0
        for idx, idx_pred in enumerate(predictions):
            pred = doc_refers[idx_pred]
            # print(pred, doc_scores[idx_pred])
            
            # Remove prediction with too low score: 20
            if doc_scores[idx_pred] >= 20:
                for article in relevant_articles:
                    if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                        true_positive += 1
                    else:
                        false_positive += 1
                    
        precision = true_positive/(true_positive + false_positive + 1e-20)
        recall = true_positive/actual_positive
        f2 = calculate_f2(precision, recall)
        total_precision += precision
        total_recall += recall
        total_f2 += f2
        
    print(f"Average F2: \t\t{total_f2/k}")
    print(f"Average Precision: {total_precision/k}")
    print(f"Average Recall: {total_recall/k}\n")
    

    print("================================================BM25 Negative Mining==================================================")
    
    
    train_path = os.path.join(data_dir, "for_train_question_answer.json")
    training_items = load_json(train_path, encoding=encoding_mode)["items"]
    
    doc_data = load_json(path=os.path.join(save_path, "legal_dict_segments.json"), encoding=encoding_mode)
    
    save_pairs = []
    train_data = []
    for idx, item in tqdm(enumerate(training_items)):
        query_data = {}
        
        question_id = item["question_id"]
        question = item["question"]
        relevant_articles = item["relevant_articles"]
        actual_positive = len(relevant_articles)
        
        
        tokenized_query = bm25_tokenizer(question)
        preprocessed_query = " ".join(tokenizing(question))
        doc_scores = bm25.get_scores(tokenized_query)
        
        query_data["query_id"] = question_id
        query_data["query"] = preprocessed_query
        
        query_data["positive_passages"] = []
        
        predictions = np.argsort(doc_scores)[::-1][:top_n]
        
        # Save positive pairs
        for article in relevant_articles:
            concat_id = article["law_id"] + "_" + article["article_id"]
            segments = doc_data[concat_id]["text"]
            #segments = segmentize(sentence=sentence, stride=stride, window=window)
            for seg in segments:
                save_dict = {}
                save_dict["question"] = preprocessed_query
                save_dict["document"] = doc_data[concat_id]["title"] + " " + segments[seg]
                save_dict["relevant"] = 1
                save_pairs.append(save_dict)
                
                save_dict_teva = {}
                pos_passage_id = concat_id + "_" + seg
                save_dict_teva["docid"] = pos_passage_id
                save_dict_teva["title"] = doc_data[concat_id]["title"]
                save_dict_teva["text"] = segments[seg]
                
                query_data["positive_passages"].append(save_dict_teva)
        
        query_data["negative_passages"] = []
            
        # Save negative pairs
        for idx, idx_pred in enumerate(predictions):
            pred = doc_refers_segments[idx_pred]
            
            check = 0
            for article in relevant_articles:
                if pred[0] == article["law_id"] and pred[1] == article["article_id"]:
                    check += 1
                    break
            
            if check == 0:
                concat_id = pred[0] + "_" + pred[1]
                segments = doc_data[concat_id]["text"]
                #segments = segmentize(sentence=sentence, stride=stride, window=window)
                for seg in segments:
                    save_dict = {}
                    save_dict["question"] = preprocessed_query
                    save_dict["document"] = doc_data[concat_id]["title"] + " " + segments[seg]
                    save_dict["relevant"] = 0
                    save_pairs.append(save_dict)
                    
                    save_dict_teva = {}
                    neg_passage_id = concat_id + "_" + seg
                    save_dict_teva["docid"] = neg_passage_id
                    save_dict_teva["title"] = doc_data[concat_id]["title"]
                    save_dict_teva["text"] = segments[seg]
                    
                    query_data["negative_passages"].append(save_dict_teva)
                    
        train_data.append(query_data)
                    
                    
    with open(os.path.join(save_dir, "bm25_pairs_top{}".format(top_n)), "wb") as pairs_file:
        pickle.dump(save_pairs, pairs_file)
        
    if not os.path.exists(os.path.join(save_dir, "tevatron_data", "train_dir")):
        os.makedirs(os.path.join(save_dir, "tevatron_data", "train_dir"), exist_ok=True)
        
    with open(os.path.join(save_dir, "tevatron_data", "train_dir", "train.jsonl"), "w", encoding=encoding_mode) as f:
        json.dump(train_data, f, indent=3)
        
        
    
    print("===============================================Create corpus passage for tevatron================================================")
    
    doc_data = load_json(path=os.path.join(save_path, "legal_dict_segments.json"), encoding=encoding_mode)
    
    data = []
    
    index = 0
    passage_id_to_index_group = {}
    
    for k, v in tqdm(doc_data.items()):
        
        passage_id_to_index_group[k] = []
        
        segments = doc_data[k]["text"]
        
        for seg in segments:
            save_dict = {}
            save_dict["docid"] = k + "_" + seg
            save_dict["title"] = doc_data[k]["title"]
            save_dict["text"] = segments[seg]
            
            data.append(save_dict)
            
            passage_id_to_index_group[k].append(index)
            index += 1
            
    if not os.path.exists(os.path.join(save_dir, "tevatron_data", "corpus")):
        os.makedirs(os.path.join(save_dir, "tevatron_data", "corpus"), exist_ok=True)
        
    with open(os.path.join(save_dir, "tevatron_data", "corpus", "corpus.jsonl"), "w", encoding=encoding_mode) as f:
        json.dump(data, f, indent=3)
        
    with open(os.path.join(save_dir, "tevatron_passage_id_to_index_group.json"), "w") as f:
        json.dump(passage_id_to_index_group, f, indent=3)
        
    print("================================================Create dev.jsonl====================================")
    
    dev_path = os.path.join(data_dir, "for_test_question_answer.json")
    dev_items = load_json(dev_path, encoding=encoding_mode)["items"]
    
    dev_data = []
    
    for idx, item in tqdm(enumerate(dev_items)):
        query_data = {}

        question_id = item["question_id"]
        question = item["question"]
        preprocessed_query = " ".join(tokenizing(question))

        query_data["query_id"] = question_id
        query_data["query"] = preprocessed_query

        query_data["positive_passages"] = []

        query_data["negative_passages"] = []
        
        dev_data.append(query_data)
        
    if not os.path.exists(os.path.join(save_dir, "tevatron_data", "dev_dir")):
        os.makedirs(os.path.join(save_dir, "tevatron_data", "dev_dir"), exist_ok=True)
        
    with open(os.path.join(save_dir, "tevatron_data", "dev_dir", "train.jsonl"), "w", encoding=encoding_mode) as f:
        json.dump(dev_data, f, indent=3)
