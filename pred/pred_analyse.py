import json

def get_error_count(input_filename):
    '''
    :param input_filename: predicted jsonlines filepath
    :return:

    Jsonlines one line:
    dict_keys(['mention_dist', 'men1_end', 'sentences', 'men1_start', 'pred', 'men2_start', 'men2_end', 'gold_label', 'doc_key'])
    '''
    error_counter = 0
    num_examples = 0
    with open(input_filename) as input_file:
        for example_num, line in enumerate(input_file.readlines()):
            example = json.loads(line)

            # for each gold label in cluster check it against pred label
            for idx, label in enumerate(example['gold_label']):
                # calculate number of all mention pairs
                num_examples += 1
                # calculate number of errors
                if label != example['pred'][idx]:
                    error_counter += 1

    return error_counter, num_examples



if __name__ == '__main__':
    filenames = ["pred_test.english.128.probe_reduce.joshi.jsonlines", "pred_test.english.128.probe_reduced.baseline.jsonlines",
                 "pred_test.english.384.probe_reduce.joshi.jsonlines", "pred_test.english.384.probe_reduced.baseline.jsonlines"]
    # input_filename = "pred_test.english.128.probe_reduce.joshi.jsonlines"

    for input_filename in filenames:
        print(input_filename)
        error_count, num_examples = get_error_count(input_filename)
        print(f'error_count: {error_count} errors in {num_examples} samples. {100 - ((error_count/num_examples)*100):.3f} accuracy % \n')