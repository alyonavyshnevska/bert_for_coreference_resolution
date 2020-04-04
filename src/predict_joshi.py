import h5py
import numpy as np
import argparse
import json

from keras.models import load_model
from keras_self_attention import SeqWeightedAttention

# python3 predict_joshi.py --model --test_data --exp_name

def get_args():
    parser = argparse.ArgumentParser(description='Run probing experiment for c2f-coref with BERT embeddings')

    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)  # without extension
    parser.add_argument('--exp_name', type=str, default=None)   # export name without extension
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # test data name comes without the extention. same name for two data files
    test_data_json = args.test_data + ".jsonlines"
    test_data_h5 = args.test_data + ".h5"
    exp_name = args.exp_name + ".jsonlines"
    model = load_model(str(args.model), custom_objects=SeqWeightedAttention.get_custom_objects())

    # span representations: [parent_child_emb, men1_start, men1_end, men2_start,
    # men2_end, doc_key_arr, mention_dist, gold_label]
    with h5py.File(args.test_data_h5, 'r') as f:
        test_data = f.get('span_representations').value
        x_test = test_data[:, :-7]
        y_test = test_data[:, -1].astype(int)

        mention_dist = test_data[:, -2].astype(int)
        doc_key_arr = test_data[:, -3].astype(float)
        men2_end = test_data[:, -4].astype(int)
        men2_start = test_data[:, -5].astype(int)
        men1_end = test_data[:, -6].astype(int)
        men1_start = test_data[:, -7].astype(int)

    test_predict = (np.asarray(model.predict(x_test))).round()

    with open(exp_name, 'w') as output_file:
        with open(test_data_json, 'r') as input_file:
            for line in input_file.readlines():
                example = json.loads(line)
                # get the dockey of this example
                doc_key = example['doc_key'].astype(float)
                # find the index of this dockey to get the prediction at the same index
                ind = doc_key_arr.index(doc_key)
                # get prediction for this doc key
                example['pred'] = test_predict[ind]
                example['men2_end'] = men2_end[ind]
                example['men2_start'] = men2_start[ind]
                example['men1_start'] = men1_start[ind]
                example['men1_end'] = men1_end[ind]

                output_file.write(json.dumps(example))
                output_file.write("\n")


#        info_dict = {'doc_key': doc_key,
#                'mention_dist': dist_list,
#                'gold_label': json_label}

