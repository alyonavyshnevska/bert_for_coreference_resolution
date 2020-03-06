from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import numpy as np


def get_parent_child_emb(clusters, span_emb, span_starts, span_ends, label_type):
    """
    Make [n,emb + 2 ] tensor where n is number of samples and emb is dimension of embeddings
    and last 2 values are distance between mentions and gold label, respectively.
    """
    span_emb_list = []
    dist_list = []
    for coref_relation, dist in clusters:
        assert len(coref_relation) == 2, 'Member of mentions are not equal to 2'
        parent_idx = np.intersect1d(np.where(span_starts==coref_relation[0][0]), 
                                    np.where(span_ends==coref_relation[0][1]))
        child_idx = np.intersect1d(np.where(span_starts==coref_relation[1][0]),
                                   np.where(span_ends==coref_relation[1][1]))
        if len(parent_idx) == 1 and len(child_idx) == 1:  # There are some mentions that exceeded max_span_width, this check skips such mentions.
            parent_child_span = tf.concat([span_emb[parent_idx], span_emb[child_idx]], 1)
            span_emb_list.append(parent_child_span)
            dist_list.append(dist)

    parent_child_emb = tf.concat(span_emb_list, 0)
    mention_dist = tf.dtypes.cast(tf.stack(dist_list, 0), tf.float32)
    mention_dist = tf.reshape(mention_dist, [-1,1])
    if label_type == "positive":
        gold_label = tf.ones([parent_child_emb.shape[0],1], tf.float32)
    elif label_type == "negative":
        gold_label = tf.zeros([parent_child_emb.shape[0],1], tf.float32)
    return tf.concat([parent_child_emb, mention_dist, gold_label], 1)
