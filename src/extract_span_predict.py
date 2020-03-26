from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os

import tensorflow as tf
import numpy as np
import span_util_predict
import util
import h5py
from custom_coref import CustomCorefIndependent

if __name__ == "__main__":
    config = util.initialize_from_env()
    log_dir = config["log_dir"]

    # Input file and output file in .jsonlines format.
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]

    # input_filename = '../data/test.english.128.probe_reduced.jsonlines'
    # output_filename = '../data/test.english.128.probe_reduced_output.jsonlines'

    model = CustomCorefIndependent(config)
    saver = tf.train.Saver()

    with tf.Session() as session:
        model.restore(session)

        with open(output_filename, 'w') as output_file:
            with open(input_filename) as input_file:
                parent_child_list = []
                num_lines = sum(1 for line in input_file.readlines())
                input_file.seek(0)  # return to first line
                for example_num, line in enumerate(input_file.readlines()):
                    example = json.loads(line)
                    tensorized_example = model.tensorize_example(example, is_training=False)
                    feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
                    candidate_span_emb, candidate_starts, candidate_ends = session.run(model.embeddings, feed_dict=feed_dict)
                    pos_clusters, neg_clusters = example["distances_positive"], example["distances_negative"]

                    # Parent child emb pos and neg are dicts with keys
                    #       {'parent_child_emb': parent_child_emb,
                    #         'mention_dist': mention_dist,
                    #         'men1_start': men1_start,
                    #         'men1_end': men1_end,
                    #         'men2_start': men2_start,
                    #         'men2_end': men2_end,
                    #         'doc_keys': doc_keys,
                    #         'gold_label': gold_label}

                    parent_child_emb_pos = span_util_predict.get_parent_child_emb(pos_clusters, candidate_span_emb, candidate_starts, candidate_ends, "positive")
                    parent_child_emb_neg = span_util_predict.get_parent_child_emb(neg_clusters, candidate_span_emb, candidate_starts, candidate_ends, "negative")
                    if parent_child_emb_neg:
                        parent_child_emb_neg['doc_key'] = example["doc_key"]
                        parent_child_emb_neg['sentence'] = example["sentences"]
                        output_file.write(json.dumps(parent_child_emb_neg))
                        output_file.write("\n")
                    if parent_child_emb_pos:
                        parent_child_emb_pos['doc_key'] = example["doc_key"]
                        parent_child_emb_pos['sentence'] = example["sentences"]
                        output_file.write(json.dumps(parent_child_emb_pos))
                        output_file.write("\n")
                    if example_num % 100 == 0:
                        print("Decoded {} examples.".format(example_num + 1))