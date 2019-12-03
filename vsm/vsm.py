import numpy as np
from collections import Counter
from functools import reduce


class Vsm:
    def __init__(self, method_file, query_file):
        self.method_file = method_file
        self.query_file = query_file
        self.term_documented_matrix = None
        self.terms = None
        self.normalized_term_documented_matrix = None
        self.method_line = None
        self.document_frequency = None
        self.inverse_document_frequency = None
        self.tf_idf_weighted_matrix = None

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
        query_vector = np.zeros(len(self.terms)).astype(int)
        with open(self.query_file, "r") as query_file_open:
            query_content = [(line.rstrip()).split() for line in query_file_open]
        for index, single_term in enumerate(self.terms):
            for single_query_line in query_content:
                if single_term in single_query_line:
                    query_vector[index] += 1

    def compute_cosine_similarity(self):
        pass

    def generate_ranked_list(self):
        pass


if __name__ == '__main__':
    vsm = Vsm(method_file="../docs/Homework4AdditionalFiles/jEdit4.3/CorpusMethods-jEdit4.3-AfterSplitStopStem.txt",
              query_file="../docs/Homework4AdditionalFiles/jEdit4.3/CorpusQueries-jEdit4.3-AfterSplitStopStem.txt")
    vsm.generate_term_documented_matrix()
    vsm.normalize_term_documented_matrix()
    vsm.compute_document_frequency()
    vsm.compute_inverse_document_frequency()
    vsm.generate_tf_idf_weighted_matrix()
    print("hi")

