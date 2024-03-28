"""
Policy model for rectangle component placement task.
"""

from agent.models.square_model import SquareModel
from ray.rllib.utils.framework import try_import_tf  # type: ignore
import gym  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class RectangleModel(SquareModel):
    """A model to predict the action logits and value for a rectangle component
    placement task.

    This model takes as inputs both a 2D grid and a component features array
    corresponding to the features of components to be placed

    on the grid. The
    model learns an encoding of the grid like in SquareModel and component
    features using a convolutional neural network with:

        - Convolutional layers
        - Batch normalization
        - ReLU activations
        - Optional max pooling

    The encoding for both the grid and component features are flattened and
    concatenated. The concatenated encoding is passed through two separate dense
    layers to generate action logits and a value prediction. The dimensions of
    the encoding depends on the specified convolutional layers.

    Attributes:
        component_feature_input (tf.Tensor): Input tensor for component features.
        encoding (tf.Tensor): Concatenated encoding of the grid and component features.
        encoding_feature (tf.Tensor): Encoded flattened component features.
        logits_model (tf.keras.Model): Model for predicting action logits.
        value_model (tf.keras.Model): Model for predicting value.

    Note:
        The encoding of the grid is done in the parent class SquareModel.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Initialize the RectangleModel object.

        Args:
            obs_space (gym.Space): Observation space.
            action_space (gym.Space): Action space.
            num_outputs (int): Number of outputs.
            model_config (Dict): Model configuration dictionary.
            name (str): Model name.
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.component_feature_input = tfkl.Input(
            shape=(
                model_config["custom_model_config"]["max_num_components"],
                model_config["custom_model_config"]["component_feature_vector_width"],
            ),
            dtype=tf.float32,
            name="component_feature_input",
        )
        self.build_encoding_rectangle(model_config)
        # if action_space is not a tuple, then calculate logits
        if not isinstance(self.action_space, gym.spaces.tuple.Tuple):
            self.build_logits_model_rectangle()
            self.build_value_model_rectangle()

    def build_encoding_rectangle(self, model_config):
        """Build the encoding layer.

        This layer concatenates the encoded grid and encoded component features.

        Args:
            model_config (Dict): Model configuration dictionary.
        """
        self.encode_flattened_component_feature(model_config)
        self.encoding = tfkl.Concatenate()([self.encoding_grid, self.encoding_feature])

    def build_logits_model_rectangle(self):
        """Build the logits model.

        This model calculates the logits using a dense layer from the encoding layer.
        The dense layer has the same number of output neurons as the size of the action space.
        """
        logits = tfkl.Dense(self.action_space.n)(self.encoding)
        self.logits_model = tf.keras.Model(
            [self.grid_input, self.component_feature_input], logits, name="logits_model"
        )

    def build_value_model_rectangle(self):
        """Build the value model.

        This model calculates the value using a dense layer where the input is
        the model encoding and the dense layer has 1 output neuron corresponding
        to the value of the state.
        """
        value = tfkl.Dense(1)(self.encoding)
        self.value_model = tf.keras.Model(
            [self.grid_input, self.component_feature_input], value, name="value_model"
        )

    def encode_flattened_component_feature(self, model_config):
        """Encode the component feature using a dense layer.

        The component feature is flattened and encoded using a dense layer which
        has the same number of output neurons as the component_feature_encoding_dimension.
        The output of the dense layer is then batch normalized and passed through a ReLU activation
        before being flattened again.

        The dimensions of the component features are (max_num_components, component_feature_vector_width)
        which are specified in the model configuration dictionary. The component feature vector width is 5.
        The size of the encoding feature is specified by the component_feature_encoding_dimension.

        Args:
            model_config (Dict): Model configuration dictionary.
        """
        feature_encoding_dimension = model_config["custom_model_config"][
            "component_feature_encoding_dimension"
        ]
        processed_feature_input = tfkl.Flatten()(self.component_feature_input)
        processed_feature_input = tfkl.Dense(feature_encoding_dimension)(
            processed_feature_input
        )
        processed_feature_input = tfkl.BatchNormalization()(processed_feature_input)
        processed_feature_input = tf.nn.relu(processed_feature_input)

        self.encoding_feature = tfkl.Flatten()(processed_feature_input)

    def preprocess(self, obs_dict):
        """Preprocess the obs_dict.

        This method gets the placement mask and all components feature from obs_dict, and
        applies the placement mask to the all components feature, i.e., set the
        features of all components that have been placed to 0.

        Args:
            obs_dict (Dict): Observation space dictionary.

        Returns:
            np.ndarray: All components feature where all features of all components
                that have been placed are set to 0.
        """
        # convert placement mask to boolean
        placement_mask = tf.cast(obs_dict["obs"]["placement_mask"], tf.bool)
        placement_mask = tf.math.logical_not(placement_mask)

        # make the placement mask the same shape as the all components feature
        placement_mask = tf.tile(
            tf.expand_dims(placement_mask, axis=-1),
            [1, 1, obs_dict["obs"]["all_components_feature"].shape[-1]],
        )

        # Mask all components that are not in the placement mask, i.e.,
        # set all the features to 0 for all components that have been placed.
        masked_components_feature = tf.where(
            placement_mask,
            obs_dict["obs"]["all_components_feature"],
            tf.zeros_like(obs_dict["obs"]["all_components_feature"]),
        )

        return masked_components_feature

    def forward(self, input_dict, state, seq_lens):
        """Forward pass of the model.

        Args:
            input_dict (Dict): Input dictionary.
            state (List): List of state tensors.
            seq_lens (Tensor): Lengths of the input sequences.

        Returns:
            Tuple[Tensor, List]: Tuple of action logits and state.

        Note:
            An action mask is applied to the logits to prevent the agent from
                selecting invalid actions.
        """
        masked_components_feature = self.preprocess(input_dict)

        logits = self.logits_model(
            [input_dict["obs"]["grid"], masked_components_feature]
        )

        self._value_out = self.value_model(
            [input_dict["obs"]["grid"], masked_components_feature]
        )

        logits += tf.maximum(
            tf.math.log(input_dict["obs"]["action_mask"]), tf.float32.min
        )

        return logits, state
