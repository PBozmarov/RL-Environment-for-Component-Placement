"""
Policy model for a rectangle component placement task using a factorized action space.
"""

from agent.models.rectangle_model import RectangleModel
from ray.rllib.utils.framework import try_import_tf  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class RectangleFactorizedModel(RectangleModel):
    r"""A model to predict the action logits and value for a rectangle component
    placement task using a factorized action space.

    This model takes as inputs both a 2D grid and a component features array
    corresponding to the features of components to be placed on the grid. The
    model learns an encoding of the grid like in RectangleModel and component
    features using a convolutional neural network with:

        - Convolutional layers
        - Batch normalization
        - ReLU activations
        - Optional max pooling

    The encoding for both the grid and component features are flattened and
    concatenated. The dimensions of the encoding depends on the specified
    convolutional layers.

    The encoding is passed through a dense layer for value prediction. The encoding is also
    concatenated with the flattened action mask and outputted in the forward pass. The agent uses
    this output to sample actions from the factorized action space by first splitting the encoding
    from the action mask and using the encoding as inputs for the action models and using the action
    mask to mask invalid actions during sampling.

    Th action space is factorized into three separate components: x, y, and
    orientation, where each component is a categorical distribution

        - Orientation: The action space is factorized using
            :math:`p(\text{orientation}, x, y) = p(\text{orientation})p(x \mid \text{orientation}) p (y \mid \text{orientation}, x)`
        - Coordinates: The action space is factorized using
            :math:`p(x, y, \text{orientation}) = p(x)p(y)p(\text{orientation} \mid x, y)`

    To sample actions from the factorized action space, the model samples from
    action models for each component, conditioned on previous components. This sampling
    is done using the FactorizedActionDistribution classes.

    Attributes:
        component_feature_input (tf.Tensor): Input tensor for component features.
        encoding (tf.Tensor): Concatenated encoding of the grid and component features.
        encoding_feature (tf.Tensor): Encoded flattened component features.
        logits_model (tf.keras.Model): Model for predicting action logits.
        value_model (tf.keras.Model): Model for predicting value.

    Note:
        The encoding of the grid is done in the parent class RectangleModel.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Initialize the RectangleFactorizedModel object.

        Args:
            obs_space (gym.Space): Observation space.
            action_space (gym.Space): Action space.
            num_outputs (int): Number of outputs.
            model_config (Dict): Model configuration dictionary.
            name (str): Model name.
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.build_encoding_model_factorized()
        self.build_value_model_factorized(encoding_size=self.encoding_size)

        # Build action model depending on specified factorization
        if model_config["custom_model_config"]["factorization"] == "orientation":
            self.build_action_models_orientation(
                encoding_size=self.encoding_size, action_space=action_space
            )
        elif model_config["custom_model_config"]["factorization"] == "coordinates":
            self.build_action_models_coordinates(
                encoding_size=self.encoding_size, action_space=action_space
            )

        # Set the action space size and shape
        self.action_space_size = (
            self.action_space[0].n * self.action_space[1].n * self.action_space[2].n
        )
        self.action_space_shape = (
            action_space[0].n,
            action_space[1].n,
            action_space[2].n,
        )

    def build_encoding_model_factorized(self):
        """Build the encoding layer.

        This layer creates an encoding model to create encodings like for
        the RectangleModel. The encoding concatenates the encoded grid
        and encoded component features which are encoded like in the
        RectangleModel.

        Args:
            model_config (Dict): Model configuration dictionary.
        """
        # Get encoding size and shape
        self.encoding_size = self.encoding.shape[1]
        self.encoding_shape = self.encoding.shape

        # Create the encoding model which takes the grid and the component features
        # and outputs the encoding - this is the concatenation of the grid encoding
        # and the component feature encoding
        self.encoding_model = tf.keras.Model(
            [self.grid_input, self.component_feature_input],
            self.encoding,
            name="encoding_model",
        )

    def build_value_model_factorized(self, encoding_size: int):
        """Build the value model.

        This model calculates the value using a dense layer from the encoding layer,
        where the dense layer has 1 output neuron corresponding to the value of the state.

        Args:
            encoding_size (int): Size of the encoding.
        """
        # Create the input for the encoding
        encoding_input = tfkl.Input(
            shape=(encoding_size,), dtype=tf.float32, name="encoding_input_for_vf"
        )
        # Create the value model which takes the encoding and outputs a value
        value = tfkl.Dense(1)(encoding_input)
        self.value_model = tf.keras.Model(encoding_input, value, name="value_model")

    def build_action_models_orientation(self, encoding_size: int, action_space):
        r"""Build the action models for the orientatoin, x and y actions.

        The distribution of orientation, x and y actions are modelled using the following
        factorization:
            :math:`p(\text{orientation}, x, y) = p(\text{orientation})p(x \mid \text{orientation}) p (y \mid \text{orientation}, x)`

        The orientation model is created by taking the encoding as an input and using a
        dense layer to output logits of size equal to the number of orientations.

        The x action model is created by taking the encoding and the one hot encoded orientation
        as an input and using a dense layer to output logits of size equal to the number
        of x values.

        The y action model is created by taking the encoding, the one hot encoded orientation
        and the x input as an input and using a dense layer to output logits of size equal
        to the number of y values.

        Args:
            encoding_size (int): Size of the encoding.
            action_space (gym.spaces.Tuple): Action space.

        Note:
            - The x inputs to the y action model are processed to be in the range of
                [0, 1] by dividing by the maximum x value during sampling.
            - The sampling is done using the FactorizedActionDistribution classes.
        """
        # =======================================================================
        # Orientation model
        # =======================================================================
        # Encoding input which has size of the encoding
        encoding_input = tfkl.Input(
            shape=(encoding_size,),
            dtype=tf.float32,
            name="encoding_input_for_action_models",
        )

        # Orientation logits which takes the encoding input and outputs logits
        # of size equal to the number of orientations
        orientation_logits = tfkl.Dense(action_space[0].n)(encoding_input)

        # Create the orientation model which takes the encoding input and outputs
        # the orientation logits
        self.orientation_model = tf.keras.Model(
            encoding_input, orientation_logits, name="orientation_model"
        )
        # One hot encode the orientation
        one_hot_orientation = tfkl.Input(
            shape=(action_space[0].n,),
            dtype=tf.float32,
            name="orientation_input_for_action_models",
        )  # need this to be float to match the type for the encodings

        # =======================================================================
        # X model
        # =======================================================================
        # Create x input which is the concatenation of the encoding input and the
        # one hot encoded orientation
        x_model_input = tfkl.Concatenate()([encoding_input, one_hot_orientation])

        # X logits which takes the x input and outputs logits of size equal to the
        # number of x values
        x_logits = tfkl.Dense(action_space[1].n)(x_model_input)

        # Create the x model which takes the encoding input and the one hot encoded
        # orientation and outputs the x logits
        self.x_model = tf.keras.Model(
            [encoding_input, one_hot_orientation], x_logits, name="x_model"
        )
        # Get the x input
        x_input = tfkl.Input(
            shape=(1,), dtype=tf.float32, name="x_input_for_action_models"
        )

        # =======================================================================
        # Y model
        # =======================================================================
        # Create y input which is the concatenation of the encoding input, the
        # one hot encoded orientation and the x input
        y_model_input = tfkl.Concatenate()(
            [encoding_input, one_hot_orientation, x_input]
        )

        # Y logits which takes the y input and outputs logits of size equal to the
        # number of y values
        y_logits = tfkl.Dense(action_space[2].n)(y_model_input)
        self.y_model = tf.keras.Model(
            [encoding_input, one_hot_orientation, x_input], y_logits, name="y_model"
        )

    def build_action_models_coordinates(self, encoding_size: int, action_space):
        r"""Build the action models for the orientatoin, x and y actions.

        The distribution of orientation, x and y actions are modelled using the following
        factorization:
            :math:`p(\text{orientation}, x, y) = p(\text{x})p(y \mid x) p (\text{orientation} \mid x, y)`

        The x action model is created by taking the encoding as an input and using a
        dense layer to output logits of size equal to the number of x values.

        The y action model is created by taking the encoding and x input as an input and
        using a dense layer to output logits of size equal to the number of y values.

        The orientation model is created by taking the encoding, the x input and the y input
        as an input and using a dense layer to output logits of size equal to the number
        of orientations.

        Args:
            encoding_size (int): Size of the encoding.
            action_space (gym.spaces.Tuple): Action space.

        Note:
            - The x inputs and y inputs to the y action model and orientation model
                are processed to be in the range of[0, 1] by dividing by the maximum x
                and y values respectively during sampling.
            - The sampling is done using the FactorizedActionDistribution classes.
        """
        # =======================================================================
        # X model
        # =======================================================================
        # Encoding input which has size of the encoding
        encoding_input = tfkl.Input(
            shape=(encoding_size,),
            dtype=tf.float32,
            name="encoding_input_for_action_models",
        )

        # x logits which takes the encoding input and outputs logits
        # of size equal to the number of x coordinates
        x_logits = tfkl.Dense(action_space[1].n)(encoding_input)

        # Create the x model which takes the encoding input and outputs
        # the x logits
        self.x_model = tf.keras.Model(encoding_input, x_logits, name="x_model")

        # Get x input
        x_input = tfkl.Input(
            shape=(1,), dtype=tf.float32, name="x_input_for_action_models"
        )

        # =======================================================================
        # Y model
        # =======================================================================
        # Create y input which is the concatenation of the encoding input, the
        # x input
        y_model_input = tfkl.Concatenate()([encoding_input, x_input])

        # Y logits which takes the y input and outputs logits of size equal to the
        # number of y values
        y_logits = tfkl.Dense(action_space[2].n)(y_model_input)
        self.y_model = tf.keras.Model(
            [encoding_input, x_input], y_logits, name="y_model"
        )

        # Get y input
        y_input = tfkl.Input(
            shape=(1,), dtype=tf.float32, name="y_input_for_action_models"
        )

        # =======================================================================
        # Orientation model
        # =======================================================================
        # Create x input which is the concatenation of the encoding input and the
        # x input and the y input
        orientation_model_input = tfkl.Concatenate()([encoding_input, x_input, y_input])

        # X logits which takes the x, y input and outputs logits of size equal to the
        # number of orientation values
        orientation_logits = tfkl.Dense(action_space[0].n)(orientation_model_input)

        # Create the x model which takes the encoding input and the one hot encoded
        # orientation and outputs the x logits
        self.orientation_model = tf.keras.Model(
            [encoding_input, x_input, y_input],
            orientation_logits,
            name="orientation_model",
        )

    def forward(self, input_dict, state, seq_lens):
        """Forward pass of the model.

        Args:
            input_dict (Dict): Input dictionary.
            state (List): List of state tensors.
            seq_lens (Tensor): Lengths of the input sequences.

        Returns:
            Tuple[Tensor, List]: Model output tensor and list of state tensors.
                The model output is the concatenation of the encoding and the
                flattened action mask.

        Note:
            - The output and the actions models are used to sample actions in the
                FactorizedActionDistribution classes.
            - Concatenating the action mask to the encoding allows the
                FactorizedActionDistribution classes to sample actions that are
                valid.
        """
        masked_components_feature = self.preprocess(input_dict)

        # Get encodings
        encoding = self.encoding_model(
            [input_dict["obs"]["grid"], masked_components_feature]
        )

        # Get model outputs
        self._value_out = self.value_model(encoding)
        output = tfkl.Concatenate()(
            [encoding, tfkl.Flatten()(input_dict["obs"]["action_mask"])]
        )
        self.output_shape = output.shape

        return output, state
