"""
matching_helper.py
"""

from gensim.models import KeyedVectors
import numpy as np
from numpy.linalg import svd
import copy


def getTopicRelevance(topic, vecs):
    sim_list = [np.square(cosSim(topic, vec)) for vec in vecs]
    return np.mean(sim_list)
    

def cosSim(array1, array2):
        if (np.linalg.norm(array1) * np.linalg.norm(array2)) == 0:
                return 0
        return np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))


def normVector(seq):
    """
    normalize a vector: calc ratio of each component
    """
    var_seq = np.square(seq)
    norm = np.sum(var_seq)
    ratio_seq = 1.0 * var_seq / norm
    return ratio_seq


def findHiddenTopics(docs, word_vectors, topic_num):
    docs_vecs = findWordVectors(docs, word_vectors)
    docs_topics = []
    weights = []
    for vecs in docs_vecs:
        component_num = min(topic_num, len(vecs))
        U, s, V = svd(vecs, full_matrices=True)
        V_selected = V[:component_num, :]
        docs_topics.append(copy.deepcopy(V_selected))
        weights.append(normVector(s[:component_num]))
    return docs_topics, weights



def findWordVectors(docs, word_vectors):
    docs_vecs = []
    for word_seq in docs:
        vecs = []
        for word in word_seq:
            try:
                vecs.append(word_vectors[word])
            except:
                continue
        docs_vecs.append(vecs[:])
    return docs_vecs



def weightedMatching(docs_vecs, docs_topics, topics_weights):
    print("mapping long and short docs...")
    docs_topics_relevance = []
    #counter = 0
    for vecs in docs_vecs:
        #print("pair", counter)
        #counter += 1
        weighted_scores = []
        for (topics, weights) in zip(docs_topics, topics_weights):
            scores = [getTopicRelevance(topic, vecs) for topic in topics]
            weighted_scores.append(np.dot(scores, weights))
        docs_topics_relevance.append(weighted_scores[:])
    return np.array(docs_topics_relevance)
            
        






            
