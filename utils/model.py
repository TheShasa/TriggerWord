from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input,\
    TimeDistributed, Conv1D
from tensorflow.keras.layers import GRU, BatchNormalization


def create_model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape=input_shape)

    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, kernel_size=15, strides=4)(
        X_input)                                 # CONV1D
    # Batch normalization
    X = BatchNormalization()(X)
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    # GRU (use 128 units and return the sequences)
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    # Batch normalization
    X = BatchNormalization()(X)

    # Step 3: Second GRU Layer (≈4 lines)
    # GRU (use 128 units and return the sequences)
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    # Batch normalization
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)                                  # dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(
        X)  # time distributed  (sigmoid)

    model = Model(inputs=X_input, outputs=X)

    return model
