import numpy as np
from collections import Counter
from functools import reduce
import logging


class Vsm:
    def __init__(self, method_file, query_file, feature_id_list):
        self.method_file = method_file
        self.query_file = query_file
        self.feature_id_list = feature_id_list
        self.term_documented_matrix = None
        self.terms = None
        self.normalized_term_documented_matrix = None
        self.method_line = None
        self.document_frequency = None
        self.inverse_document_frequency = None
        self.tf_idf_weighted_matrix = None
        self.query_vector = None
        self.cosine_similaries_matrix = None

    def generate_term_documented_matrix(self):
        with open(self.method_file, "r") as method_file_open:
            contents = [(line.rstrip()).split() for line in method_file_open]
            content_flat = (reduce(lambda x, y: x+y, contents))
        self.terms = list(set(content_flat))
        term_documented_matrix = np.zeros((len(contents), len(self.terms))).astype(int)
        for term_matrix_row, single_content in enumerate(contents):
            for single_term in single_content:
                term_matrix_column = self.terms.index(single_term)
                term_documented_matrix[term_matrix_row][term_matrix_column] += 1
        self.term_documented_matrix = term_documented_matrix
        self.method_line = contents.copy()

    def normalize_term_documented_matrix(self):
        self.normalized_term_documented_matrix = (self.term_documented_matrix.T/self.term_documented_matrix.max(axis=1)).T

    def compute_document_frequency(self):
        document_frequency = np.zeros(len(self.terms)).astype(int)
        for index, single_term in enumerate(self.terms):
            for single_document_line in self.method_line:
                if single_term in single_document_line:
                    document_frequency[index] += 1
        self.document_frequency = document_frequency

    def compute_inverse_document_frequency(self):
        inverse_document_frequency = np.full(len(self.terms), len(self.method_line)) / self.document_frequency
        self.inverse_document_frequency = np.log(inverse_document_frequency)

    def generate_tf_idf_weighted_matrix(self):
        self.tf_idf_weighted_matrix = self.normalized_term_documented_matrix * self.inverse_document_frequency

    def generate_vector_from_query(self):
        with open(self.query_file, "r") as query_file_open:
            query_content = [(line.rstrip()).split() for line in query_file_open]
        query_vector = np.zeros((len(query_content), len(self.terms))).astype(int)
        for matrix_row, single_content in enumerate(query_content):
            for single_term in single_content:
                try:
                    matrix_column = self.terms.index(single_term)
                    query_vector[matrix_row][matrix_column] += 1
                except Exception as e:
                    pass
        self.query_vector = query_vector

    def compute_cosine_similarity(self):
        cosine_similarities = np.zeros((len(self.query_vector), len(self.tf_idf_weighted_matrix)))
        for query_index, single_query_vector in enumerate(self.query_vector):
            for method_index, single_method_line in enumerate(self.tf_idf_weighted_matrix):
                cosine_similarities[query_index][method_index] = np.dot(single_method_line, single_query_vector) /\
                                                                 (np.math.sqrt(np.sum(single_method_line ** 2))
                                                                  * np.math.sqrt(np.sum(single_query_vector ** 2)))
        self.cosine_similaries_matrix = cosine_similarities

    def generate_ranked_list(self):
        pass

    def effectiveness_for_features(self):
        # opening the effectiveness csv file
        try:
            file = open("effectiveness_file.csv", "w")
            file.write("FeatureID\tGoldSet MethodID Position\tGoldSetMethodID\tVSM GoldSetMethodID Rank - All Ranks\t"
                       "VSM GoldSetMethodID Rank - Best Rank\n")
        except Exception as e:
            logging.error("Error opening the effectiveness file")
            exit(1)

        # read the mapping to generate the Goldset methodID position
        with open("../docs/Homework4AdditionalFiles/jEdit4.3/CorpusMethods-jEdit4.3.mapping", "r") as map:
            method_map = [single_map.rstrip() for single_map in map.readlines()]
        # Read the feature IDs
        with open("../docs/Homework4AdditionalFiles/jEdit4.3/jEdit4.3ListOfFeatureIDs.txt", "r") as feature_ids_file:
            feature_ids = [feature_id.rstrip() for feature_id in feature_ids_file.readlines()]
        # Calculate effectiveness for all feature ids
        for feature_index, feature_id in enumerate(feature_ids):
            #  by reading the goldset method get the methods associated with this feature id
            with open(f"../docs/Homework4AdditionalFiles/jEdit4.3/jEdit4.3GoldSets/GoldSet{feature_id}.txt", "r") as goldset:
                methods_from_goldset = [method.rstrip() for method in goldset.readlines()]

            # Check for each method
            for goldset_method_index, goldset_method_id in enumerate(methods_from_goldset):
                # Find the method Id position
                try:
                    method_id_position = method_map.index(goldset_method_id)
                    # Get the cosine similarity of this specific query
                    query_similarity = self.cosine_similaries_matrix[feature_index]
                    # argsort sort the array in ascendding order so need to make it reverse to get the descending order
                    query_similarity_sorted_index = np.argsort(query_similarity)[::-1]
                    # Get the ranking both all rank and best rank
                    all_rank = np.where(query_similarity_sorted_index == method_id_position)[0][0] + 1
                    best_rank = -1
                except Exception as e:  # Some of the method won't be on the map so we will put -1 for them
                    method_id_position = -2  # will add one later to match with the line number
                    all_rank = ""
                    best_rank = ""

                # String to write
                if goldset_method_index == 0:
                    string_to_write = feature_id + "\t" + str(method_id_position+1) + "\t" + goldset_method_id + "\t" +\
                                      str(all_rank) + "\t" + str(best_rank)
                else:
                    string_to_write = "" + "\t" + str(
                        method_id_position + 1) + "\t" + goldset_method_id + "\t" + \
                                      str(all_rank) + "\t" + str(best_rank)
                file.write(string_to_write+"\n")


if __name__ == '__main__':
    vsm = Vsm(method_file="../docs/Homework4AdditionalFiles/jEdit4.3/CorpusMethods-jEdit4.3-AfterSplitStopStem.txt",
              query_file="../docs/Homework4AdditionalFiles/jEdit4.3/CorpusQueries-jEdit4.3-AfterSplitStopStem.txt",
              feature_id_list="../docs/Homework4AdditionalFiles/jEdit4.3/jEdit4.3ListOfFeatureIDs.txt")
    vsm.generate_term_documented_matrix()
    vsm.normalize_term_documented_matrix()
    vsm.compute_document_frequency()
    vsm.compute_inverse_document_frequency()
    vsm.generate_tf_idf_weighted_matrix()
    vsm.generate_vector_from_query()
    vsm.compute_cosine_similarity()
    vsm.effectiveness_for_features()
    print("hi")

