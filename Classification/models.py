import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.activations import elu, relu, sigmoid
#from attention_decoder import AttentionDecoder
from tensorflow import keras
# from bert import BertModelLayer
# import bert




def get_model(algo, INPUT_LEN, dim, CLASSES):
    if algo == 'evidential_oos':
        return get_evidential_oos(INPUT_LEN,  CLASSES)
    elif algo == 'evidential_turn_30':
        return get_evidential_turn_30(INPUT_LEN, CLASSES)
    else:
        print("model not implemented")
        return None


def get_evidential_oos(INPUT_LEN, CLASSES):
    hidden_size = 128
    model = tf.keras.Sequential([
        # Seq2Seq
        Dense(hidden_size, input_shape=(INPUT_LEN,), activation=relu),
        Dropout(0.2),
        Dense(CLASSES)
    ])
    return model


def get_evidential_turn_30(INPUT_LEN, CLASSES):
    hidden_size = 128
    model = tf.keras.Sequential([
        # Seq2Seq
        Dense(hidden_size, input_shape=(INPUT_LEN,), activation=relu),
        Dropout(0.2),
        # Dense(hidden_size, input_shape=(INPUT_LEN,), activation=relu),
        # Dropout(0.2),
        # Dense(hidden_size, input_shape=(INPUT_LEN,), activation=relu),
        # Dropout(0.2),
        Dense(CLASSES) # , activation=relu
    ])
    return model
