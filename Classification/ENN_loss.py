import tensorflow as tf
import numpy as np
np.random.seed(1234)

class ENN_lambda_update(tf.keras.callbacks.Callback):
    def __init__(self, lambda_t=0, max_t=1):
        self.lambda_t = tf.Variable(initial_value=lambda_t, dtype=tf.float64)
        self.max_t = tf.Variable(initial_value=max_t, dtype=tf.float64)

    def on_epoch_end(self, epoch, logs={}):
        self.lambda_t.assign(tf.reduce_min([self.max_t, tf.cast(epoch, tf.dtypes.float64) / 10.0]))


def dirichlet_kl_divergence(alpha_c_target, alpha_c_pred, eps=10e-10):
    tf.print(alpha_c_target)
    tf.print(tf.argmax(alpha_c_pred, axis=-1))
    alpha_0_target = tf.reduce_sum(alpha_c_target, axis=-1, keepdims=True)
    alpha_0_pred = tf.reduce_sum(alpha_c_pred, axis=-1, keepdims=True)

    tf.print(tf.math.lgamma(alpha_0_pred))
    tf.print(tf.math.lgamma(alpha_0_target))

    tf.print(tf.math.lgamma(alpha_c_pred + eps))
    tf.print(tf.math.lgamma(alpha_c_target + eps))

    tf.print(tf.math.digamma(alpha_c_pred + eps))
    tf.print(tf.math.digamma(alpha_c_target + eps))

    term1 = tf.math.lgamma(alpha_0_target) - tf.math.lgamma(alpha_0_pred)
    term2 = tf.math.lgamma(alpha_c_pred + eps) - tf.math.lgamma(alpha_c_target + eps)

    term3_tmp = tf.math.digamma(alpha_c_target + eps) - tf.math.digamma(alpha_0_target + eps)
    term3 = (alpha_c_target - alpha_c_pred) * term3_tmp

    result = tf.squeeze(term1 + tf.reduce_sum(term2 + term3, keepdims=True, axis=-1))

    return result

def build_evidential_type_2_MLL(lambda_callback : ENN_lambda_update):
    def evidential_evidential_type_nll(y_true, logits):
        """
        :param y_true: one-hot probability vector of true class
        :param logits: logit output of network
        :return:
        """
        y_true = tf.cast(y_true, tf.float64)
        logits = tf.cast(logits, tf.float64)

        alpha_c = tf.exp(logits) + 1
        alpha_c = tf.clip_by_value(alpha_c, clip_value_min=10e-25, clip_value_max=10e25)
        S = tf.reduce_sum(alpha_c, axis=-1, keepdims=True)

        A = tf.reduce_sum(y_true * (tf.math.log(S) - tf.math.log(alpha_c)), axis=-1)

        annealing_coef = lambda_callback.lambda_t

        E = alpha_c - 1
        alp = E * (1 - y_true) + 1
        beta = tf.ones_like(alp)
        C = annealing_coef * dirichlet_kl_divergence(alp, beta)

        return A + C

    return evidential_evidential_type_nll


def build_evidential_cross_entropy(lambda_callback: ENN_lambda_update):
    def evidential_cross_entropy(y_true, logits):
        y_true = tf.cast(y_true, tf.float64)
        logits = tf.cast(logits, tf.float64)

        alpha_c = tf.exp(logits) + 1
        alpha_c = tf.clip_by_value(alpha_c, clip_value_min=10e-25, clip_value_max=10e25)

        S = tf.reduce_sum(alpha_c, axis=1, keepdims=True)
        E = alpha_c - 1

        A = tf.reduce_sum(y_true * (tf.math.digamma(S) - tf.math.digamma(alpha_c)), axis=1)

        annealing_coef = lambda_callback.lambda_t

        alp = E * (1 - y_true) + 1
        beta = tf.ones_like(alp)

        B = annealing_coef * dirichlet_kl_divergence(alp, beta)

        return A + B

    return evidential_cross_entropy


def build_evidential_mse(lambda_callback: ENN_lambda_update):
    def evidential_evidential_mse(y_true, logits):
        """
        :param y_true: one-hot probability vector of true class
        :param logits: logit output of network
        :return:
        """
        # The computation of alpha for evidential neural networks differs from prior networks (+ 1)
        y_true = tf.cast(y_true, tf.float64)
        logits = tf.cast(logits, tf.float64)

        alpha_c = tf.exp(logits) + 1
        alpha_c = tf.clip_by_value(alpha_c, clip_value_min=10e-25, clip_value_max=10e25)

        S = tf.reduce_sum(alpha_c, axis=-1, keepdims=True)
        E = alpha_c - 1
        m = alpha_c / S

        A = tf.reduce_sum((y_true - m) ** 2, axis=-1)
        B = tf.reduce_sum(alpha_c * (S - alpha_c) / (S * S * (S + 1)), axis=-1)

        annealing_coef = lambda_callback.lambda_t

        alp = E * (1 - y_true) + 1

        beta = tf.ones_like(alp)

        C = annealing_coef * dirichlet_kl_divergence(alp, beta)

        return (A + B) + C

    return evidential_evidential_mse