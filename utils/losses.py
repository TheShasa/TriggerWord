import tensorflow as tf
import tensorflow.keras.backend as K


class CustomLosses:
    @staticmethod
    def f1_score(y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2 * p * r / (p + r + K.epsilon())
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        return K.mean(f1)

    @staticmethod
    def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """

        weights = K.variable(weights)

        def loss(y_true, y_pred):
            y_true = K.cast(y_true, 'float')
            # scale predictions so that the class probas of each sample sum to 1
            # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            in_loss = y_true * K.log(y_pred) * weights
            in_loss = -K.sum(in_loss, -1)
            return in_loss

        return loss

    @staticmethod
    def weighted_bincrossentropy(weight_zero=1.0, weight_one=10.0):
        """
        Calculates weighted binary cross entropy. The weights are fixed.

        This can be useful for unbalanced catagories.

        Adjust the weights here depending on what is required.

        For example if there are 10x as many positive classes as negative classes,
            if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
            will be penalize 10 times as much as false negatives.
        """
        def loss(true, pred):
            # calculate the binary cross entropy
            bin_crossentropy = K.binary_crossentropy(true, pred)

            # apply the weights
            weights = true * weight_one + (1. - true) * weight_zero
            weighted_bin_crossentropy = weights * bin_crossentropy

            return K.mean(weighted_bin_crossentropy)
        return loss
