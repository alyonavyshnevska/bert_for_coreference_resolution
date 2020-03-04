import os
import jsonlines
import random
import json
import sys
import itertools


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))



def create_negative_positive_samples(path_jsonlines_file, output_path, debug_mode=True):
    '''

    :param path_jsonlines_file: File with clusters of coreferent mentions
    Structure: key = 'clusters'. Inside clusters are lists of coreferences. Each coreference has len 2
    For e.g. clusters = [[[61, 63], [44, 46], [27, 29]],
                            [[21, 25], [18, 18]],
                            [[87, 87], [86, 86]]]

        Mention 61-63 is coreferent with 44-46 and 27-29

    :output_file: Where to save the jsonlines file with negative, positive examples and distances

    :return: nothing
    '''

    output_samples = list()

    # read jsonlines and save dictionaries(=samples) in a list
    with jsonlines.open(path_jsonlines_file) as reader:
        # for each dictionary in jsonlines code
        for sample in reader.iter(type=dict, skip_invalid=True):

            mentions = set()

            # create a set of spans to choose from randomly
            for cluster in sample['clusters']:
                for mention in cluster:
                    mentions.add(tuple(mention))

            # back to list of mentions
            mentions = [list(mention) for mention in mentions]

            list_of_negative_clusters = list()
            list_of_positive_clusters = list()


            for cluster in sample['clusters']:

                # create positive samples: combinations of a cluster of length 2
                # combinations because we want only [[87,87],[86,86]] as positive example out of cluster
                # [[87,87], [86,86]]. ermutations would give us [[87,87],[86,86]], [[86,86],[87,87]]
                permutations = list(itertools.combinations(cluster, 2))
                permutations = [sorted(list(i)) for i in permutations]
                list_of_positive_clusters.extend(permutations)

                # create random negative samples:
                # for each mention choose a mention from all mentions in this document,
                # that is not in the current cluster
                negative_cluster = list()

                for mention in cluster:
                    #remove mentions from current cluster as option for a random negative examples
                    options = [m for m in mentions if m not in cluster]

                    # choose a random mention from all mentions in this sample except from own cluster
                    random_mention = random.choice(options)
                    negative_cluster.append(sorted([mention, random_mention]))

                if debug_mode:
                    print('\navaliable options for negative: ', options)
                    print(cluster, '=========>', negative_cluster)

                list_of_negative_clusters.extend(negative_cluster)

            # assert that every mention received one negative mention
            assert len(mentions) == len(list_of_negative_clusters), 'Not every mention received a negative examples'

            #calculate distances
            distances_positive = [[cluster, abs(cluster[1][0] - cluster[0][1])] for cluster in list_of_positive_clusters]
            distances_negative = [[cluster, abs(cluster[1][0] - cluster[0][1])] for cluster in list_of_negative_clusters]


            # write negative_clusters to the sample
            sample['negative_clusters'] = list_of_negative_clusters
            sample['positive_clusters'] = list_of_positive_clusters
            sample['distances_positive'] = distances_positive
            sample['distances_negative'] = distances_negative

            output_samples.append(sample)

        dump_jsonl(output_samples, output_path, append=False)


if __name__ == '__main__':
    # create_negative_positive_samples('../test/dummy_test.jsonlines', '../test/dummy_test_output.jsonlines', debug_mode=True)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    debug_mode = False

    # insert verbose as sys.args[3] to print created negative clusters
    if '--verbose' in sys.argv:
        debug_mode = True
    create_negative_positive_samples(input_file, output_file, debug_mode=debug_mode)


    # if '--verbose' in sys.argv:
    #     with jsonlines.open(output_file) as reader:
    #         for sample in reader.iter(type=dict, skip_invalid=True):
    #             print("\nClusters:")
    #             for i in sample['clusters']:
    #                 print(i)
    #             print("Negative Clusters: ", sample['negative_clusters'])