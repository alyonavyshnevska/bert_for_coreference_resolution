from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import numpy as np


def get_parent_child_emb(clusters, span_emb, span_starts, span_ends):
    """
    Make [n,emb] tensor where n is number of samples and emb is dimension of embeddings
    """
    span_emb_list = []
    for coref_relation in clusters:
        if len(coref_relation) == 2:
            parent_idx = np.intersect1d(np.where(span_starts==coref_relation[0][0]), 
                                        np.where(span_ends==coref_relation[0][1]))
            child_idx = np.intersect1d(np.where(span_starts==coref_relation[1][0]),
                                       np.where(span_ends==coref_relation[1][1]))
            parent_child_span = tf.concat([span_emb[parent_idx], span_emb[child_idx]], 1)
            span_emb_list.append(parent_child_span)

    parent_child_emb = tf.concat(span_emb_list, 0)
    return parent_child_emb
