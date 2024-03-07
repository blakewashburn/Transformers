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


    
Notes:
Tokenization:
    - Tokenizing = convert text into tokens, which are units for analysis in NLP
    - Tokens can be words, characters, or subwords. Tokenizer breaks down text into
    these tokens and then converts each token into numerical data (token IDs)

"""
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertTokenizer
import numpy as np


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


# PREPROCESSING: Tokenize and Encode
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_reviews(tokenizer: any, reviews: list[str], max_length: int):
    """
    Process a list of text reviews using a tokenizer from the transformer library

    Args:
    tokenizer: An instance of a tokenizer from the transformer library
    reviews: A list of strings, where each string is a review
    max_length: the max length of the tokenized output

    Returns:
    a dictionary with tensors that are ready to be fed into a model. 
    """
    # 'padding=max_length' ensures that all tokenized sequences are padded to 
    # the max_length with zeros if the sequence is shorter
    # 'truncation=True' allows sequences longer than max_length to be truncated
    # 'max_length=max_length' specifies max sequence length the tokens can be 
    # 'return_tensors=tf' specifies the output in the dict should be tensors
    return tokenizer(reviews, padding='max_length', truncation=True, 
                     max_length=max_length, return_tensors='tf')


max_seq_length = 256        # maximum length based on model limits and dataset
print("Beginning Tokenizing...")
train_encodings = encode_reviews(tokenizer=tokenizer, reviews=train_review, 
                                 max_length=max_seq_length)
test_encodings = encode_reviews(tokenizer=tokenizer, reviews=test_review, 
                                 max_length=max_seq_length)
print("Completed Tokenizing...")
print(type(train_encodings))
print(train_encodings.keys())
print(train_encodings)

# exit()


# Prepare labels 
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


# Create a TensorFlow Dataset, as tf.data.Dataset objects are more effecient
# for training
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings), train_labels
))
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings), test_labels
))