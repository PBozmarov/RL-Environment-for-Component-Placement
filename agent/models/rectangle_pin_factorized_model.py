"""
Policy model for the rectangle component placement task where the components have pins and where
the action space is factorized.
"""

from agent.models.rectangle_pin_model import RectanglePinModel
from ray.rllib.utils.framework import try_import_tf  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class RectanglePinFactorizedModel(RectanglePinModel):
    """A model to predict the action logits and value for placing components
    with pins on a 2D grid.

    This model takes as inputs a 2D grid, a component features array,
    a component pin features array corresponding to the features of pins
    on components, and a placement mask array. The model learns an encoding of
    these features by using the encodings for:

        - The grid (same as RectanglePinModel)
        - The component features
        - The component pin features
        - The placement mask

    The grid encoding is created like in RectanglePinModel. The component features
    array is encoded by passing the features for each component through a
    dense layer. The component pin features array is encoded by:

        - For each component, passing the features for each pin through a dense layer
        - Summing the encoded pin features for each component to get a single
            encoded feature vector for each component.

    The placement mask is encoded by simply one-hot encoding it. The encodings
    component features, component pin features, and placement mask are
    concatenated along the feature axis to get an encoding of dimensions
    (max_num_components, component_encoding_dim + pin_encoding_dim + 4).
    This encoding is then flattened and concatenated with the flattened grid encoding to
    get a single encoding.

    This new encoding is passed through a dense layer for value prediction. The encoding is also
    concatenated with the flattened action mask and outputted in the forward pass. The agent uses
    this output to sample actions from the factorized action space by first splitting the encoding
    from the action mask and using the encoding as inputs for the action models and using the action
    mask to mask invalid actions during sampling.

    To sample actions from the factorized action space, the model samples from
    action models for each component, conditioned on previous components. This sampling
    is done using the FactorizedActionDistribution classes.

    Attributes:
        encoding (tf.Tensor): Encoding of the input features.
        encoding_model (tf.keras.Model): Model to create the encoding.
        value_model (tf.keras.Model): Model to predict the value.
        action_model (tf.keras.Model): Model to predict the action logits.
        action_space_size (int): Size of the action space.
        action_space_shape (Tuple): Shape of the action space.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Initialize the RectangleFactorizedModel object.

        Args:
            obs_space (gym.spaces.Dict):Oobservation space.
            action_space (gym.spaces.Tuple): Action space.
            num_outputs (int): Number of outputs.
            model_config (Dict): Model configuration dictionary.
            name (str): Name of the model.
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.build_encoding_model_factorized()
        self.build_value_model_factorized(encoding_size=self.encoding_size)

        # Set the action space size and shape
        self.action_space_size = (
            self.action_space[0].n * self.action_space[1].n * self.action_space[2].n
        )
        self.action_space_shape = (
            action_space[0].n,
            action_space[1].n,
            action_space[2].n,
        )

        # Build action model depending on specified factorization
        if model_config["custom_model_config"]["factorization"] == "orientation":
            self.build_action_models_orientation(
                encoding_size=self.encoding_size, action_space=action_space
            )
        elif model_config["custom_model_config"]["factorization"] == "coordinates":
            self.build_action_models_coordinates(
                encoding_size=self.encoding_size, action_space=action_space
            )

    def build_encoding_model_factorized(self):
        """Build the encoding layer.

        This layer creates an encoding model to create encodings like for
        the RectanglePinModel. The encoding concatenates the encoded grid
        and encoded component features which are encoded like in the
        RectanglePinModel.

        Args:
            model_config (Dict): Model configuration dictionary.
        """
        # Get encoding size
        self.encoding_size = self.encoding.shape[1]
        self.encoding_shape = self.encoding.shape

        # Create encoding model
        self.encoding_model = tf.keras.Model(
            [
                self.grid_input,
                self.component_feature_input,
                self.component_pin_feature_input,
                self.placement_mask_input,
            ],
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
        r"""Build the action models for the orientation, x and y actions.

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
        # Orientation model
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

        # X model
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

        # Y model
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
        r"""Build the action models for the orientation, x and y actions.

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
        # X model
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

        # Y model
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

        # Orientation model
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
            input_dict (dict): Input dictionary containing the observation.
            state (List): List of state tensors.
            seq_lens (Tensor): Tensor containing the sequence length.

        Returns:
            Tuple[Tensor, List]: Tuple of action logits and state.

        Note:
            The output and the actions models are used to sample actions in the
                FactorizedActionDistribution classes.
        """
        (
            placement_mask,
            all_components_feature,
            all_component_pins_feature,
        ) = self.preprocess(input_dict)

        # Get encodings
        encoding = self.encoding_model(
            [
                input_dict["obs"]["grid"],
                all_components_feature,
                all_component_pins_feature,
                placement_mask,
            ]
        )

        # Get value and output
        self._value_out = self.value_model(encoding)
        output = tfkl.Concatenate()(
            [encoding, tfkl.Flatten()(input_dict["obs"]["action_mask"])]
        )
        self.output_shape = output.shape

        return output, state
