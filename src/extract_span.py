from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import numpy as np
import span_util
import util
from custom_coref import CustomCorefIndependent

if __name__ == "__main__":
    config = util.initialize_from_env()
    log_dir = config["log_dir"]

    # Input file in .jsonlines format.
    input_filename = sys.argv[2]

    model = CustomCorefIndependent(config)
    saver = tf.train.Saver()

    with tf.Session() as session:
        model.restore(session)

        with open(input_filename) as input_file:
            for example_num, line in enumerate(input_file.readlines()):
                example = json.loads(line)
                tensorized_example = model.tensorize_example(example, is_training=False)
                feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
                candidate_span_emb, candidate_starts, candidate_ends = session.run(model.embeddings, feed_dict=feed_dict)
                positive_clusters = example["clusters"] # temporary, change to key for positive and negative sampels later
                parent_child_emb = span_util.get_parent_child_emb(positive_clusters, candidate_span_emb, candidate_starts, candidate_ends)
                print(parent_child_emb.shape)
