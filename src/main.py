"""
main.py
long_docs: list of concateation of reviewers' works

short_docs: list of abstract in submissions

matching_scores: shape (submission_num, reviewers)
each entry (i,j) is a real value in range of (0,1),
indicating how well submission i matches reviewer j
"""

import preprocess
import tfidf_helper
import match_helper
from gensim.models import KeyedVectors
import argparse


def vectorizeShortDoc(raw_docs, word_vectors, is_refine=False, word_limit=100):
    """
    word vectors for each short doc
    """
    # tokenize
    print("vectorize short docs...")
    docs = []
    for raw_doc in raw_docs:
        docs.append(preprocess.tokenizeText(raw_doc))
    #docs = preprocess.tokenizeText(raw_docs)
    if (is_refine):
        docs = tfidf_helper.extract(docs, word_limit)
    docs_vecs = match_helper.findWordVectors(docs, word_vectors)
    return docs_vecs
    


def vectorizeLongDoc(raw_docs, word_vectors, topic_num=10, is_refine=False, word_limit=100):
    """
    raw_docs: a list of the concateation of reviewers' works
    vector space for each long doc
    """
    # tokenize
    print("vectorize long docs...")
    docs = []
    for raw_doc in raw_docs:
        docs.append(preprocess.tokenizeText(raw_doc))
    #docs = preprocess.tokenizeText(raw_docs)
    # if refine with tf-idf methods
    if (is_refine):
        docs = tfidf_helper.extract(docs, word_limit)
    docs_topics, topic_weights = match_helper.findHiddenTopics(docs, word_vectors, topic_num)
    return docs_topics, topic_weights
    


def mapping(embedding_path, raw_short_docs, raw_long_docs, topic_num=10, \
            is_binary=False, is_refine_short=False, is_refine_long=False, \
            short_word_limit=100, long_word_limit=1000):
    # load word embeddings
    word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=is_binary, limit=20000)
    
    # tokenize long docs
    short_docs_vecs = vectorizeShortDoc(raw_short_docs, word_vectors, \
                                        is_refine_short, short_word_limit)
    long_docs_vecs, topic_weights = vectorizeLongDoc(raw_long_docs, word_vectors, topic_num, \
                                      is_refine_long, long_word_limit)
    matching_scores = match_helper.weightedMatching(short_docs_vecs, long_docs_vecs, topic_weights)
    matching_scores = np.sqrt(matching_scores)
    return matching_scores


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

    # read short_docs and long docs
    # !! to be implemented !!
    
    mapping(embedding_path, raw_short_doc, raw_long_docs, topic_num, \
            is_binary, is_refine_short, is_refine_long, short_word_limit, long_word_limit)

    
    
    
    
