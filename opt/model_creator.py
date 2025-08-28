import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2

import custom_transform
from DFTlayer import DFTLayer


class ModelCreator:
    """
    A helper class to build and compile a Keras model with a custom DFT layer
    for exchange-correlation (XC) density calculations.

    Attributes
    ----------
    num_inputs : int
        Dimensionality of input data (number of features per sample).
    input_squeeze : str or None
        Name of the transform to apply to the inputs, e.g. "SignedLogTransform",
        "AbsLogTransform", or None.
    number_of_layers : int
        Total number of layers in the main neural network (excluding the final 
        DFT layer).
    hidden_units : int
        Number of hidden neurons in each hidden layer.
    activation_function_hidden : str
        Activation function for hidden layers (e.g. "relu").
    activation_function_output : str 
        Activation function for the output layer before passing to the DFT layer.
    l2_ratio : float
        L2 regularization factor applied to Dense layers.
    x_model : str
        Name/identifier for the exchange functional (e.g. "PBE").
    c_model : str
        Name/identifier for the correlation functional (e.g. "B95", "B97").
    nlx : float
        Mixing fraction (0 <= nlx <= 1) for exchange weighting.
    scal_opp : np.ndarray
        Initial scaling factor(s) for the opposite-spin correlation term.
    scal_ss : np.ndarray
        Initial scaling factor(s) for the same-spin correlation term.
    c_ss : float
        Parameter controlling same-spin correlation.
    c_opp : float
        Parameter controlling opposite-spin correlation.
    seed : int
        Random seed for weight initializers.
    corr_train : bool
        Whether the correlation parameters in DFTLayer should be trainable.
    rs_model : bool
        Whether to apply range-separation logic in DFTLayer.
    """

    def __init__(
        self,
        num_inputs: int,
        input_squeeze: str | None,
        number_of_layers: int,
        hidden_units: int,
        activation_function_hidden: str,
        activation_function_output: str,
        l2_ratio: float,
        x_model: str,
        c_model: str,
        nlx: float = 1.0,
        scal_opp: np.ndarray = 1.0,
        scal_ss: np.ndarray = 1.0,
        c_ss: float = 0.038,
        c_opp: float = 0.0031,
        seed: int = 42,
        corr_train: bool = False,
        rs_model: bool = False,
        linear_scaling: str = 'm0p1',
        restart = None,
        qac = None
    ):
        """
        Initialize the ModelCreator with parameters necessary 
        for building the model.
        """
        self.num_inputs = num_inputs
        self.input_squeeze = input_squeeze
        self.number_of_layers = number_of_layers
        self.hidden_units = hidden_units
        self.activation_function_hidden = activation_function_hidden
        self.activation_function_output = activation_function_output
        self.l2_ratio = l2_ratio
        self.nlx = nlx
        self.scal_opp = scal_opp
        self.scal_ss = scal_ss
        self.c_ss = c_ss
        self.c_opp = c_opp
        self.corr_train = corr_train
        self.x_model = x_model
        self.c_model = c_model
        self.seed = seed
        self.rs_model = rs_model
        self.restart = restart
        self.qac = qac
        self.linear_scaling = linear_scaling

    def create_model(self) -> tuple[Model,]:
        """
        Build and compile the Keras model, consisting of:
        1) A sequence of Dense layers (optionally applying input transforms).
        2) A final Dense layer with the chosen activation.
        4) A DFTLayer that uses the neural network output and additional
           features to compute XC density.

        Returns
        -------
        model : tensorflow.keras.Model
        """
        # Create the input placeholder
        inputs = tf.keras.Input(shape=(self.num_inputs,))
        features_input = tf.keras.Input(shape=(13,), name="features_input")

        # Optional input transforms
        x = inputs
        if self.input_squeeze is None:
            x = custom_transform.NoneTransform()(x)
        elif self.input_squeeze == "SignedLogTransform":
            x = custom_transform.SignedLogTransform()(x)
        elif self.input_squeeze == "AbsLogTransform":
            x = custom_transform.AbsLogTransform()(x)

        # Hidden Dense layers
        for _ in range(self.number_of_layers - 1):
            x = Dense(
                self.hidden_units,
                activation=self.activation_function_hidden,
                kernel_initializer=initializers.glorot_uniform(seed=self.seed),
                kernel_regularizer=l2(self.l2_ratio),
                dtype='float32'
            )(x)

        # Final Dense layer (before DFTLayer)
        x = Dense(
            1,
            activation=self.activation_function_output,
            kernel_regularizer=l2(self.l2_ratio),
            kernel_initializer=initializers.glorot_uniform(seed=self.seed),
            dtype='float32'
        )(x)

        if (self.linear_scaling=='m1p1'):
            x = 2.0 * x - 1.0 
        elif (self.linear_scaling=='m0p1'):
            pass

        # Pass through DFTLayer
        outputs = DFTLayer(
            name='dft_layer',
            scal_opp=self.scal_opp,
            scal_ss=self.scal_ss,
            c_ss=self.c_ss,
            c_opp=self.c_opp,
            nlx=self.nlx,
            train=self.corr_train,
            x_model=self.x_model,
            c_model=self.c_model,
            rs_model=self.rs_model,
            qac = self.qac
        )([x, features_input])

        # Build the model with multiple outputs: [xc_density, nLMF], 
        # nLMF is x here
        model = Model(inputs=[inputs, features_input], outputs=[outputs,x])

        # load old weights 
        if (self.restart is not None):
            model.load_weights(self.restart, skip_mismatch=True)

        return model

