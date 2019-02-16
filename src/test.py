"""
test.py
"""
import numpy as np
from sklearn.metrics import f1_score
from main import vectorizeShortDoc, vectorizeLongDoc
import argparse
import match_helper
import pickle
from gensim.models import KeyedVectors


def readRes(res_fn="data/res.txt"):
    f = open(res_fn, "r")
    gold_labels = [int(line.strip()) for line in f.readlines()]
    f.close()
    return gold_labels

def readData(data_fn="data/dataset.txt"):
    f = open(data_fn, "r")
    lines = f.readlines()
    f.close()
    delimiter = "\t****\t"
    raw_short_docs = []
    raw_long_docs = []
    for line in lines:
        concept_name, concept, proj_title, proj_abstract, proj_intro, proj_web \
                      = line.strip().lower().split(delimiter)
        raw_short_docs.append(" ".join([concept_name, concept]))
        raw_long_docs.append(" ".join([proj_title, proj_abstract, proj_intro]))
    return raw_short_docs, raw_long_docs
        

def testMapping(embedding_path, raw_short_docs, raw_long_docs, topic_num=10, \
            is_binary=False, is_refine_short=False, is_refine_long=False, \
            short_word_limit=100, long_word_limit=1000):
    # load word embeddings
    word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=is_binary, limit=20000)

    # tokenize long docs
    short_docs_vecs = vectorizeShortDoc(raw_short_docs, word_vectors, \
                                        is_refine_short, short_word_limit)
    long_docs_vecs, topic_weights = vectorizeLongDoc(raw_long_docs, word_vectors, topic_num, \
                                      is_refine_long, long_word_limit)
    matching_scores = []
    counter = 0
    for (short_doc_vec, long_doc_vec, topic_weight) in zip(short_docs_vecs, long_docs_vecs, topic_weights):
        print("pair:", counter)
        counter += 1
        s = match_helper.weightedMatching([short_doc_vec], [long_doc_vec], [topic_weight])
        matching_scores.append(s[0])
    #with open("data/matching_scores.pkl", "wb") as handle:
    #    pickle.dump(matching_scores, handle)
    return matching_scores

        
def evalMatching(gold_scores, pred_scores):
    best_fscore = 0.0
    for t in np.arange(0, 1.0, 0.02):
        pred_labels = [int(s > t) for s in pred_scores]
        fscore = f1_score(gold_scores, pred_labels)
        best_fscore = max(fscore, best_fscore)
    print("best_fscore: {}".format(best_fscore))


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, default="../src/data/gold_d3_word2vec_model.txt")
    parser.add_argument('--binary_embedding', default=False, action='store_true')
    parser.add_argument('--is_refine_short', default=False, action='store_true')
    parser.add_argument('--is_refine_long', default=False, action='store_true')
    parser.add_argument('--short_word_limit', type=int, default=100)
    parser.add_argument('--long_word_limit', type=int, default=1000)
    parser.add_argument('--topic_num', type=int, default=10)
    args = parser.parse_args()
    embedding_path = args.embedding_path
    is_binary = args.binary_embedding
    is_refine_short = args.is_refine_short
    is_refine_long = args.is_refine_long
    short_word_limit = args.short_word_limit
    long_word_limit = args.long_word_limit
    topic_num = args.topic_num

    print("reading data...")
    raw_short_docs, raw_long_docs = readData()
    print("mapping...")
    pred_scores = testMapping(embedding_path, raw_short_docs, raw_long_docs, topic_num, \
            is_binary, is_refine_short, is_refine_long, short_word_limit, long_word_limit)
    gold_scores = readRes()
    print("evaluation...")
    evalMatching(gold_scores, pred_scores)
