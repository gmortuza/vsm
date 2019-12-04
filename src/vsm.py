import numpy as np
from functools import reduce
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Vsm:
    def __init__(self, input_file_dir):
        # Input files
        self.input_file_dir = input_file_dir
        self.method_file = input_file_dir + "/CorpusMethods-jEdit4.3-AfterSplitStopStem.txt"
        self.query_file = input_file_dir + "/CorpusQueries-jEdit4.3-AfterSplitStopStem.txt"
        self.feature_id_list = input_file_dir + "/jEdit4.3ListOfFeatureIDs.txt"
        # All of these will be calculated in their own method
        self.term_documented_matrix = None
        self.terms = None
        self.normalized_term_documented_matrix = None
        self.method_line = None
        self.document_frequency = None
        self.inverse_document_frequency = None
        self.tf_idf_weighted_matrix = None
        self.query_vector = None
        self.cosine_similarities_matrix = None
        # Call all the method. That will compute the cosine similarity
        self.generate_term_documented_matrix()
        self.normalize_term_documented_matrix()
        self.compute_document_frequency()
        self.compute_inverse_document_frequency()
        self.generate_tf_idf_weighted_matrix()
        self.generate_vector_from_query()
        self.compute_cosine_similarity()

    def generate_term_documented_matrix(self):
        logging.info("Calculating Term Documented Matrix")
        # Reading the method file to get all the terms
        with open(self.method_file, "r") as method_file_open:
            contents = [(line.rstrip()).split() for line in method_file_open]
            content_flat = (reduce(lambda x, y: x+y, contents))
        self.terms = list(set(content_flat))
        # Making the term documented matrix
        term_documented_matrix = np.zeros((len(contents), len(self.terms))).astype(int)
        for term_matrix_row, single_content in enumerate(contents):
            for single_term in single_content:
                term_matrix_column = self.terms.index(single_term)
                term_documented_matrix[term_matrix_row][term_matrix_column] += 1
        self.term_documented_matrix = term_documented_matrix
        self.method_line = contents.copy()

    def normalize_term_documented_matrix(self):
        logging.info("Normalizing the Term Documented Matrix")
        # Normalize the term documented matrix
        self.normalized_term_documented_matrix = (self.term_documented_matrix.T/self.term_documented_matrix.max(axis=1)).T

    def compute_document_frequency(self):
        logging.info("Computing document frequency")
        document_frequency = np.zeros(len(self.terms)).astype(int)
        # Look for the whole document for document frequency of a term.
        for index, single_term in enumerate(self.terms):
            for single_document_line in self.method_line:
                if single_term in single_document_line:
                    document_frequency[index] += 1
        self.document_frequency = document_frequency

    def compute_inverse_document_frequency(self):
        logging.info("Computing inverse document frequency")
        # Computing inverse
        inverse_document_frequency = np.full(len(self.terms), len(self.method_line)) / self.document_frequency
        self.inverse_document_frequency = np.log(inverse_document_frequency)

    def generate_tf_idf_weighted_matrix(self):
        logging.info("Generating TF-IDF weighted matrix")
        # GEnerating tf_idf matrix
        self.tf_idf_weighted_matrix = self.normalized_term_documented_matrix * self.inverse_document_frequency

    def generate_vector_from_query(self):
        logging.info("Creating vector from query")
        # REading the query file
        with open(self.query_file, "r") as query_file_open:
            query_content = [(line.rstrip()).split() for line in query_file_open]
        query_vector = np.zeros((len(query_content), len(self.terms))).astype(int)
        # Making query as vector
        for matrix_row, single_content in enumerate(query_content):
            for single_term in single_content:
                try:
                    matrix_column = self.terms.index(single_term)
                    query_vector[matrix_row][matrix_column] += 1
                except Exception as e:
                    pass
        self.query_vector = query_vector

    def compute_cosine_similarity(self):
        logging.info("Computing cosine similarities")
        cosine_similarities = np.zeros((len(self.query_vector), len(self.tf_idf_weighted_matrix)))
        # Cosine similarites of query and document
        for query_index, single_query_vector in enumerate(self.query_vector):
            for method_index, single_method_line in enumerate(self.tf_idf_weighted_matrix):
                cosine_similarities[query_index][method_index] = np.dot(single_method_line, single_query_vector) /\
                                                                 (np.math.sqrt(np.sum(single_method_line ** 2))
                                                                  * np.math.sqrt(np.sum(single_query_vector ** 2)))
        self.cosine_similarities_matrix = cosine_similarities

    def effectiveness_for_features(self):
        logging.info("Generating the CSV file containing the effectiveness data")
        # opening the effectiveness csv file
        try:
            file = open("VSM_Effectiveness.csv", "w")
            file.write("FeatureID\tGoldSet MethodID Position\tGoldSetMethodID\tVSM GoldSetMethodID Rank - All Ranks\t"
                       "VSM GoldSetMethodID Rank - Best Rank\n")
        except Exception as e:
            logging.error("Error opening the effectiveness file")
            exit(1)

        # read the mapping to generate the Goldset methodID position
        with open(self.input_file_dir + "/CorpusMethods-jEdit4.3.mapping", "r") as map:
            method_map = [single_map.rstrip() for single_map in map.readlines()]
        # Read the feature IDs
        with open(self.input_file_dir + "/jEdit4.3ListOfFeatureIDs.txt", "r") as feature_ids_file:
            feature_ids = [feature_id.rstrip() for feature_id in feature_ids_file.readlines()]
        # Calculate effectiveness for all feature ids
        for feature_index, feature_id in enumerate(feature_ids):
            #  by reading the goldset method get the methods associated with this feature id
            with open(f"{self.input_file_dir}/jEdit4.3GoldSets/GoldSet{feature_id}.txt", "r") as goldset:
                methods_from_goldset = [method.rstrip() for method in goldset.readlines()]

            feature_details = []  # Key value pair of the feauture id and it's details
            best_rank = sys.maxsize
            # Check for each method
            for goldset_method_index, goldset_method_id in enumerate(methods_from_goldset):
                # Find the method Id position
                try:
                    method_id_position = method_map.index(goldset_method_id)
                    # Get the cosine similarity of this specific query
                    query_similarity = self.cosine_similarities_matrix[feature_index]
                    # argsort sort the array in ascendding order so need to make it reverse to get the descending order
                    query_similarity_sorted_index = np.argsort(query_similarity)[::-1]
                    # Get the ranking both all rank and best rank
                    all_rank = np.where(query_similarity_sorted_index == method_id_position)[0][0] + 1
                    best_rank = all_rank if all_rank < best_rank else best_rank
                except Exception as e:  # Some of the method won't be on the map so we will put -1 for them
                    method_id_position = -2  # will add one later to match with the line number
                    all_rank = ""
                feature_details.append([str(method_id_position+1), str(goldset_method_id), str(all_rank), ""])
            # write in the file
            for index, single_method_id_details in enumerate(feature_details):
                if index == 0:
                    if single_method_id_details[0] == "-1":
                        best_rank = ""
                    string_to_write = str(feature_id) + "\t" + single_method_id_details[0] + "\t" + \
                                      single_method_id_details[1] + "\t" + \
                                      single_method_id_details[2] + "\t" + str(best_rank)
                else:
                    string_to_write = "" + "\t" + single_method_id_details[0] + "\t" + single_method_id_details[
                        1] + "\t" + single_method_id_details[2] + "\t" + ""
                file.write(string_to_write+"\n")
        file.close()
        logging.info("Done writing the CSV file")


if __name__ == '__main__':
    vsm = Vsm(input_file_dir="input_file/jEdit4.3")
    vsm.effectiveness_for_features()

