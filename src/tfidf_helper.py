"""
extract key words from articles
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import nltk
import pickle

dump_folder = "dump/"


def extract(docs, max_word=500):
    """
    docs: [[word1, word2], [word3, word4]]
    """
    str_docs = [" ".join(word_seq) for word_seq in docs]
    vectorizer = CountVectorizer(min_df=5)
    X = vectorizer.fit_transform(str_docs)
    word_list = vectorizer.get_feature_names()
    count_matrix = X.toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(count_matrix)
    tfidf_mat = tfidf.toarray()
    # sort words in each doc according to its weight
    sorted_tfidf_mat = np.argsort(tfidf_mat, axis=1)
    sorted_tfidf_mat = np.fliplr(sorted_tfidf_mat)
    
    # take most informative k words
    refined_docs = []
    doc_count, word_count = tfidf_mat.shape
    k = min(word_count, max_word)
    for doc_ind in range(doc_count):
        word_inds = [sorted_tfidf_mat[doc_ind][pos] for pos in range(k)]
        words = [word_list[ind] for ind in word_inds]
        refined_docs.append(words[:])
    return refined_docs

