"""
Blake Washburn
2024/03/04

IMDb movie review classifier

This code classifies the movie reviews in the TensorFlow dataset (TFDS) into 
two classes, positive and negative. 

Task: Binary Classification
Model: 
Dataset: IMDb Movie Review Dataset
    - 50,000 movie reviews from the Internet Movie Database (IMD)
    - 25,000 labeled 1 for positive and 25,000 labeled 1 for negative
    - reviews are in english, vary in length, and cover a wide range of movies


"""
import os
import tensorflow as tf
import tensorflow_datasets as tfds


def load_imdb_dataset(imdb_directory: str):
    """
    Extracts the movie review and labels from the imdb movie review dataset 
    downloaded from https://ai.stanford.edu/%7Eamaas/data/sentiment/

    Args: 
    imdb_directory: absolute path to either train or test directory

    Returns:
    reviews: list of strings that contains the movie review
    labels: list of labels, either 1 or 0
    """
    reviews = []
    labels = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(imdb_directory, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                with open(os.path.join(dir_name, fname)) as f:
                    reviews.append(f.read())
                if label_type == 'pos':
                    labels.append(1)
                else:
                    labels.append(0)
    return reviews, labels

# load IMDb dataset
train_dir = os.getcwd() + "/datasets/aclImdb/train"
test_dir = os.getcwd() + "/datasets/aclImdb/test"
train_review, train_labels = load_imdb_dataset(train_dir)
test_review, test_labels = load_imdb_dataset(test_dir)


# Quick exploration of the dataset
# print(train_review[0])
# print(train_labels[0])

