"""
Blake Washburn

IMDb movie review classifier

This code classifies the movie reviews in the TensorFlow dataset (TFDS) into 
two classes, positive and negative. 

Notes: 
- The core concept behind transformers is the self-attention mechanism, which 
allows the model to weigh the importance of different parts of the input data 
differently. 
- This is a departure from RNNs and CNNs, which process data sequentially or 
through local receptive fields, respectively. 
- Transformers can handle long-range dependencies in data more effectively.

Key Components of Transformers
1. Embeddings
    - Both the input and output tokens are converted into vectors of a fixed 
    dimension
2. Positional Encoding: 
    - Since transformers do not process data sequentially, we need to provide
    some information about the position of tokens in the sequence. 
    - This is done through positional encodings added to the embeddings
3. Self Attention: 
    - Allows the model to focus on different parts of the input sequence when
    producing a specific part of the output sequence. 
4. Multi-Head Attension: 
    - An extension of self-attention, where the attention mechanism is run in
    multiple times to allow the model to jointly attend to information from 
    different representation subspaces. 
5. Feed-Forward Neural Networks: 
    - Applied to each position separately and identically
6. Normalization and Residual Connections
    - Used around each of the sublayers (self-attension and feed-forward
    networks)
"""


import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


def get_positional_encoding(max_seq_length, embed_dim):
    positions = np.arange(max_seq_length)[:, np.newaxis]
    div_terms = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
    pos_encoding = np.zeros((max_seq_length, embed_dim))
    pos_encoding[:, 0::2] = np.sin(positions * div_terms)
    pos_encoding[:, 1::2] = np.cos(positions * div_terms)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

