from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import numpy as np
import span_util
import util
import h5py
from custom_coref import CustomCorefIndependent

if __name__ == "__main__":
    config = util.initialize_from_env()
    log_dir = config["log_dir"]

    # Input file in .jsonlines format.
    input_filename = sys.argv[2]

    # Span embeddings will be written to this file in .h5 format.
    output_filename = sys.argv[3]

    model = CustomCorefIndependent(config)
    saver = tf.train.Saver()

    with tf.Session() as session:
        model.restore(session)

        with open(input_filename) as input_file:
            with h5py.File(output_filename, 'w') as hf:
                parent_child_list = []
                for example_num, line in enumerate(input_file.readlines()):
                    example = json.loads(line)
                    tensorized_example = model.tensorize_example(example, is_training=False)
                    feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
                    candidate_span_emb, candidate_starts, candidate_ends = session.run(model.embeddings, feed_dict=feed_dict)
                    pos_clusters, neg_clusters = example["distances_positive"], example["distances_negative"]
                    parent_child_emb_pos = span_util.get_parent_child_emb(pos_clusters, candidate_span_emb, candidate_starts, candidate_ends, "positive")
                    parent_child_emb_neg = span_util.get_parent_child_emb(neg_clusters, candidate_span_emb, candidate_starts, candidate_ends, "negative")
                    parent_child_list.extend([parent_child_emb_pos, parent_child_emb_neg])

                parent_child_reps = tf.concat(parent_child_list, 0).eval()
                hf.create_dataset("span_representations", data=parent_child_reps, compression="gzip", compression_opts=0, shuffle=True, chunks=True)
                print(parent_child_reps[:100,-2])
                print(parent_child_reps[:100,-1])
                print(parent_child_reps.shape)
