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
from keras import layers
import numpy as np


def get_positional_encoding(max_seq_length: int, embed_dim: int) -> tf.Tensor:
    """
    Povides information about the position of each token in the sequence. 
    Helps the transformer understand the order of the tokens

    Args: 
    max_sequence_length: maximum length of sequence in dataset
    embed_dim: dimensionality of the embeddings
    
    Returns:
    tf.Tensor of positional encodings of shape (1, max_seq_length, embed_dim)
    """
    # create an array of shape (max_seq_length) where each element is its index. 
    # assign a unique position number to each position in the sequence
    positions = np.arange(max_seq_length)[:, np.newaxis]

    # Calculate the denominator part of the positional encoding. 
    # creates varying wavelengths for sinusoidal functions, so each dimension
    # of encoding varies at a different rate
    div_terms = np.exp(np.arange(0, embed_dim, 2) * - (np.log(10000.0) / embed_dim))
    
    # Create a matrix to be filled with encoding values
    pos_encoding = np.zeros((max_seq_length, embed_dim))

    # Fills positional encoding matrix with sine values for even indicies. 
    # We use sine because it provides a smooth gradient that can be easily differentiated.
    # multiplication by div_terms ensure that dimension oscillates 
    pos_encoding[:, 0::2] = np.sin(positions * div_terms)

    # Cosine used for odd values
    pos_encoding[:, 1::2] = np.cos(positions * div_terms)

    # Add an extra dimension to the start of encoding matrix, 
    # Facilitates easy addition to the embeddings matrix which may have a batch
    # dimension
    pos_encoding = pos_encoding[np.newaxis, ...]

    # Cast the positional encoding matrix to a tensor with a float32 type
    # to make it more combatable with TensorFlow model and ready for adding to 
    # embeddings. 
    return tf.cast(pos_encoding, dtype=tf.float32)
    

def scaled_dot_product_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, 
                                 mask: tf.Tensor = None) -> tf.Tensor:
    """
    Calculate the attenton score by scaling the dot products of the queries
    and keys. which helps stabalize the gradients during training. Then, 
    softmax is applied to obtain the weights on the values. 
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimensions

    Args:
    q: queries input tensor
    k: keys input tensor
    v: values input tensor
    mask: Float tensor with shape (..., seq_len_q, seq_len_k)

    Returns:
    output: tf.tensor representing the 
    attention_weights:
    """

    # Perform dot product of queries and keys
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # Scale matmul_qk by sqrt of depth of key vectors (dk)
    # helps prevent the softmax function from having extremely small gradients
    # during training
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add the mask to the scaled tensor.
    # Typically used to ignore padding tokens or prevent model from attending
    # to future tokens 
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax across keys for each query to obtain attention weights
    # Weights add up to 1, so they can be seen as probabilities
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) 

    # Attention weights are used to create a weighted sum of values
    # producing the outputs of the attention mechanism
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    """
    Enables the model to simultaniously process information from different 
    subspaces at different positions by splitting the attention mechanism
    into multiple heads. So it focuses on different parts of the input sequence
    for a given output part. 

    Inherits from TensorFlow's layers.Layer object. 
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initialize the MultiHeadAttention object

        Args: 
        embed_dim: size of input embeddings
        num_heads: number of attention heads
        """

        # call constructor for the keras.layers.Layer object
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # ensure that embedding dimensions are divisible by number of heads
        # The model splits the embeddings into equal parts for each head
        assert embed_dim % num_heads == 0

        # calculate the depth of each attention head's output. 
        # Each head deals with this number of embedding dimensions.      
        self.depth = embed_dim // num_heads
        
        # Initialize dense layers that will be used to generate queries
        # keys, and values for the attention mechanisms. 
        self.Wq = layers.Dense(embed_dim)   # weights applied to queries
        self.Wk = layers.Dense(embed_dim)   # weights applied to keys
        self.Wv = layers.Dense(embed_dim)   # weights applied to values

        # Create a dense layer that will be used to transform the concatinated
        # outputs of the attention heads back into the original embedding dimensions
        self.dense = layers.Dense(embed_dim)
    
    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """
        Reshape the input matricies (Wq, Wk, Wv) so that the model can process the 
        data through multiple attention heads in parallel

        Args: 
        x: one of the Wq, Wk, Wv matricies in self
        batch_size: number of sequences processed in parallel
        
        Returns:
        tf.tensor that is reshaped to be ready for multi-head processing
        """
        # reshape input tensor to prepare for parallel processing by multiple 
        # attention heads. Original shape = (batch_size, seq_len, embed_dim). 
        # New shape is (batch_size, seq_len, num_heads, depth)
        # This reshaping enables each attention head to process a different part
        # of the embedding space independently
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        # reshapes the tensor dimensions to (batch_size, num_heads, seq_len, depth)
        # because tensorflow works best with batch_size coming first. 
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        """
        Applies the multi-head attention mechanism to the inputs.
        call() is automatically invoked when MultiHeadAttention is called on 
        some input. 

        Args:
        q: queries input tensor
        k: keys input tensor
        v: values input tensor

        Returns:
        output: output tensor after attention application and concatination
        """

        # Get batch_size from query tensor
        batch_size = tf.shape(q)[0]
        
        # apply sense layers to their respective inputs. 
        # Projects query, keys, values into spaces for attention mechanism
        q = self.Wq(q) 
        k = self.Wk(k)
        v = self.Wv(v)
        # resulting shape: (batch_size, seq_len, embed_dim)
        
        # Split tensors into multiple heads
        q = self.split_heads(q, batch_size) 
        k = self.split_heads(k, batch_size) 
        v = self.split_heads(v, batch_size)
        # resulting shape: (batch_size, num_heads, seq_len, depth)
        
        # calculate the attention scores by scaling the dot product of queries
        # and keys, applying a softmax to obtain weights, and then using these
        # weights to aggregate the values
        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask=None)
        # resulting shape: (batch_size, num_heads, seq_len_q, depth)

        # Concatenate heads and run through final linear layer

        # reshape to: (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatinate the attention outputs from all heads back into a single tensor
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embed_dim))  
        # Resulting shape: (batch_size, seq_len_q, embed_dim)

        # Apply final dense layer to concatinated attnetion outputs
        output = self.dense(concat_attention)
        return output



class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs, inputs)  # Self attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2



class Transformer(tf.keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Transformer, self).__init__()
        self.token_embedding = layers.Embedding(input_vocab_size, embed_dim)
        self.pos_encoding = get_positional_encoding(maximum_position_encoding, embed_dim)
        
        self.encoder_layers = [TransformerEncoderLayer(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
        
        self.dropout = layers.Dropout(rate)
        self.final_layer = layers.Dense(1)  # Assuming binary classification for simplicity
    
    def call(self, inputs, training):
        seq_len = tf.shape(inputs)[1]
        x = self.token_embedding(inputs)  # (batch_size, input_seq_len, embed_dim)
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training)
        
        x = self.final_layer(x[:, 0, :])  # Take the encoding of the first token for classification
        
        return x
