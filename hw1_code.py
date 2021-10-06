from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from math import log2
from pydotplus import graph_from_dot_data
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import numpy as np
import random 

def load_files(filename: str) -> List[str]:
    """
    Helper function for Question 2 (a)
    Responsible for returning all the news headlines as a list
    """

    text_file = open(filename, 'r')
    file_data = text_file.readlines()
    return [x.strip('\n') for x in file_data]

def vectorize_data(data: List[str])-> Tuple[List]:
    """
    Helper function for Question 2 (a)
    Returns data in a vectorised form
    """

    vectorizer = CountVectorizer()
    corpus = vectorizer.fit_transform(data)
    return corpus.toarray(), vectorizer.get_feature_names_out(), vectorizer.vocabulary_

def split_data(data: List[List[int]], real_length: int) -> Tuple:

    """
    Helper function for Question 2 (a)

    Randomly splits data in the following proportions
    Training set: 70%
    Validation set: 15%
    Testing set: 15%

    """

    # 1 means real and 0 means fake
    labels = ([1]*real_length) + ([0]*(len(data) - real_length))

    X_train, X_test_val, Y_train, Y_test_val = train_test_split(
        data, labels, train_size=0.7, test_size=0.3)
    
    X_test, X_val, Y_test, Y_val = train_test_split(
        X_test_val, Y_test_val, train_size=0.5, test_size=0.5)

    return X_train, X_test, X_val, Y_train, Y_test, Y_val
    
def load_data() -> Tuple:
    """

    Final Solution for Question 2 (a) 

    Load, pre-process, vectorizes and splits data into training, validation
    and testing examples.
    """

    real_data = load_files("clean_real.txt")
    fake_data = load_files("clean_fake.txt")
    all_data = real_data + fake_data
    vectorized_data, feature_names, vocab = vectorize_data(all_data)
    return split_data(vectorized_data, len(real_data)), feature_names, vocab


def create_decision_tree(x_train, y_train, max_depth: Optional[int], 
                            criterion: str = "gini") -> DecisionTreeClassifier :
    """
    Helper funtion Question 2 (b)

    Responsible for creating decision tree with given max_depth and split criteria
    """

    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    clf.fit(x_train, y_train)

    return clf

def validate_decision_tree(clf: DecisionTreeClassifier, x_val: List, 
                            y_val: List) -> int:

    """
    Helper funtion Question 2 (b)

    Responsible for validation of decision trees to be able to select the 
    model hyperparams.
    """

    return clf.score(x_val, y_val)

def visualise_tree(clf: DecisionTreeClassifier, feature_names):
    """
    Solution to Question 2 (c)
    Visualise the tree and save it in a png
    """
    dot_data = export_graphviz(clf, out_file=None, 
                                rounded=True,
                                filled=True,  
                                class_names=['fake', 'real'], 
                                feature_names=feature_names, 
                                max_depth=3
                                )
    graph = graph_from_dot_data(dot_data)
    graph.write_png('tree.png')

def select_model(datasets: Tuple):
    """
    Final Solution for Q2 (b)
    
    Trains the decision tree classifier using 5 different values of max_depth, 
    as well as two different split criteria (information gain and Gini coefficient), 
    and evaluates the performance of each one on the
    validation set, and prints the resulting accuracies of each model

    """
    criterion_list = ['gini', 'entropy']
    max_depth_list = [10, 30, 50, 75, 120]
    X_train, X_test, X_val, Y_train, Y_test, Y_val = datasets
    for criterion in criterion_list:
        for max_depth in max_depth_list:
            tree = create_decision_tree(X_train, Y_train, max_depth=max_depth, 
                                                          criterion=criterion)
            accuracy = validate_decision_tree(tree, X_val, Y_val)
            print(f"Parameters: max_depth: {max_depth} and split criterion: {criterion} | Model Accuracy: {accuracy}")

def prepare_dataframe():

    real_data = load_files("clean_real.txt")
    fake_data = load_files("clean_fake.txt")
    all_data = real_data + fake_data
    real_length = len(real_data)
    labels = ([1]*real_length) + ([0]*(len(all_data) - real_length))

    df = pd.DataFrame(list(zip(all_data, labels)), columns=['headline', 'label'])

    return df

def compute_information_gain(split_keyword: str, X_train, Y_train, vocab):

    # Finding Entropy
    total = len(Y_train)
    num_true = len([x for x in Y_train if x == 1])
    p_true = (num_true/total)
    p_false = 1 - p_true

    entropy = -1 * ((p_true*log2(p_true) + p_false*log2(p_false)))

    # Finding the vocab index
    word_index = vocab[split_keyword]

    # Finding Condition Entropy
    keyword_list = [x for x in list(zip(X_train, Y_train)) if x[0][word_index] > 0]

    true_list = [x for x in keyword_list if x[1] == 1]

    total = len(keyword_list)
    num_true = len(true_list)
    p_true = (num_true/total)
    p_false = 1 - p_true

    condtional_entropy = -1 * ((p_true*log2(p_true) + p_false*log2(p_false)))

    ig = entropy - condtional_entropy

    return ig


if __name__ == "__main__":
    
    datasets, feature_names, vocab= load_data()

    X_train, X_test, X_val, Y_train, Y_test, Y_val = datasets

    print(X_train)
    # tree = create_decision_tree(X_train, Y_train, max_depth=75, 
    #                                                       criterion="gini")
    # select_model(datasets)
    # visualise_tree(tree, feature_names)
    print(compute_information_gain("turnbull", X_train, Y_train, vocab))


# Parameters: max_depth: 10 and split criterion: gini | Model Accuracy: 0.7040816326530612
# Parameters: max_depth: 30 and split criterion: gini | Model Accuracy: 0.7591836734693878
# Parameters: max_depth: 50 and split criterion: gini | Model Accuracy: 0.7612244897959184
# Parameters: max_depth: 75 and split criterion: gini | Model Accuracy: 0.7653061224489796
# Parameters: max_depth: 120 and split criterion: gini | Model Accuracy: 0.746938775510204
# Parameters: max_depth: 10 and split criterion: entropy | Model Accuracy: 0.6979591836734694
# Parameters: max_depth: 30 and split criterion: entropy | Model Accuracy: 0.7224489795918367
# Parameters: max_depth: 50 and split criterion: entropy | Model Accuracy: 0.753061224489796
# Parameters: max_depth: 75 and split criterion: entropy | Model Accuracy: 0.7428571428571429
# Parameters: max_depth: 120 and split criterion: entropy | Model Accuracy: 0.7428571428571429