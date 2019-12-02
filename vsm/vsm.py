import numpy as np


class Vsm:
    def __init__(self, method_file, query_file):
        self.method_file = method_file
        self.query_file = query_file

    def generate_term_documented_matrix(self):
        pass

    def normalize_term_documented_matrix(self, term_documented_matrix):
        pass

    def compute_document_frequency(self):
        pass

    def compute_inverse_document_frequency(self):
        pass

    def generate_tf_idf_wegihted_matrix(self):
        pass

    def generate_vector_from_query(self):
        pass

    def compute_cosine_similarity(self):
        pass

    def generate_ranked_list(self):
        pass


if __name__ == '__main__':
    vsm = Vsm(method_file="docs/Homework4AdditionalFiles/jEdit4.3/CorpusMethods-jEdit4.3-AfterSplitStopStem.txt",
              query_file="docs/Homework4AdditionalFiles/jEdit4.3/CorpusQueries-jEdit4.3-AfterSplitStopStem.txt")
