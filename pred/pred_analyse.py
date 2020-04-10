import json
import ast

def get_error_count(input_filename):
    '''
    :param input_filename: predicted jsonlines filepath
    :return:

    Jsonlines one line:
    dict_keys(['mention_dist', 'men1_end', 'sentences', 'men1_start', 'pred', 'men2_start', 'men2_end', 'gold_label', 'doc_key'])
    '''
    error_counter = 0
    num_examples = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    errors = 'errors' + input_filename

    # with open(errors, 'w'):
    with open(input_filename) as input_file:
        for example_num, line in enumerate(input_file.readlines()):
            example = json.loads(line)
            # for each gold label in cluster check it against pred label
            for idx, label in enumerate(example['gold_label']):
                pred_label = example['pred'][idx]
                # calculate number of all mention pairs
                num_examples += 1
                # if num_examples < 30:

                #     print(len(example['sentences'][0]))
                #     print(example['sentences'])


                # # calculate true positives...
                # if label == 1 and pred_label == 1:
                #     true_pos += 1
                # elif label == 0 and pred_label == 0:
                #     true_neg +=1
                # elif label == 1 and pred_label == 0:
                #     false_neg +=1
                # elif label == 0 and pred_label == 1:
                #     false_pos += 1
                #
                # # calculate number of errors
                # if label != pred_label:
                #     error_counter += 1

                # print errors
                if label != pred_label:
                    error_counter += 1
                    if error_counter < 100:
                        sentences = [item for sublist in example['sentences'] for item in sublist]
                        print(example['mention_dist'][idx])
                        before = max(0, example['men1_start'][idx] - 7)
                        after = min(len(sentences)-1, example['men2_end'][idx])
                        print(" ".join(sentences[example['men1_start'][idx]:example['men1_end'][idx]+3]), '   ==   ', " ".join(sentences[example['men2_start'][idx]:example['men2_end'][idx]+1]))
                        if example['mention_dist'][idx] < 20:
                            print(' in ', " ".join(sentences[before:after]), '\n' )
                        else:
                            print('in', " ".join(sentences[before:example['men1_end'][idx]+5]), '...', " ".join(sentences[example['men2_start'][idx]-5:example['men2_end'][idx]+5]), '\n' )





    return error_counter, num_examples, true_pos, true_neg, false_pos, false_neg


if __name__ == '__main__':
    filenames = ["pred_test.english.128.probe_reduce.joshi.jsonlines", "pred_test.english.384.probe_reduce.joshi.jsonlines"]
    # input_filename = "pred_test.english.128.probe_reduce.joshi.jsonlines"

    for input_filename in filenames:
        print(input_filename)
        error_count, num_examples, true_pos, true_neg, false_pos, false_neg  = get_error_count(input_filename)
        # print(f'error_count: {error_count} errors in {num_examples} samples. {100 - ((error_count/num_examples)*100):.3f} accuracy % \n')
        # print(f'true_pos, true_neg, false_pos, false_neg, {true_pos, true_neg, false_pos, false_neg}')
        # print(f'error_clusters: {error_clusters} errors in {num_clusters} clusters. {100 - ((error_clusters/num_clusters)*100):.3f} accuracy % \n')