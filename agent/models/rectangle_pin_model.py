"""
Policy model for the rectangle component placement task.
"""

from agent.models.rectangle_model import RectangleModel
from ray.rllib.utils.framework import try_import_tf  # type: ignore
import gym  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class RectanglePinModel(RectangleModel):
    """A model to predict the action logits and value for placing components
    with pins on a 2D grid.

    This model takes as inputs a 2D grid, a component features array,
    a component pin features array corresponding to the features of pins
    on components, and a placement mask array. The model learns an encoding of
    these features by using the encodings for:

        - The grid (same as RectangleModel)
        - The component features
        - The component pin features
        - The placement mask

    The grid encoding is created like in RectangleModel. The component features
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

    The concatenated encoding is passed through two separate dense layers to
    generate action logits and a value prediction.

    Attributes:
        component_feature_input (tf.Tensor): Input tensor for component features.
        component_pin_feature_input (tf.Tensor): Input tensor for component pin features.
        placement_mask_input (tf.Tensor): Input tensor for placement mask.
        encoding (tf.Tensor): Concatenated encoding of the grid, component features,
            component pin features, and placement mask.
        encoding_feature (tf.Tensor): Encoded flattened component features.
        encoding_pin (tf.Tensor): Encoded flattened component pin features.
        encoding_mask (tf.Tensor): Encoded flattened placement mask.
        logits_model (tf.keras.Model): Model for predicting action logits.
        value_model (tf.keras.Model): Model for predicting value.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Initialize the RectanglePinModel object.

        Args:
            obs_space (gym.spaces.Box): Observation space.
            action_space (gym.spaces.Discrete): Action space.
            num_outputs (int): Number of outputs.
            model_config (dict): Model config dict.
            name (str): Name of the model.
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.max_num_nets = model_config["custom_model_config"]["max_num_nets"]

        self.component_pin_feature_input = tfkl.Input(
            shape=(
                model_config["custom_model_config"]["max_num_components"],
                model_config["custom_model_config"]["max_num_pins_per_component"],
                model_config["custom_model_config"]["pin_feature_vector_width"],
            ),
            dtype=tf.float32,
            name="pin_feature_input",
        )

        self.placement_mask_input = tfkl.Input(
            shape=(model_config["custom_model_config"]["max_num_components"], 1),
            dtype=tf.float32,
            name="placement_mask_input",
        )

        self.build_encoding_rectangle_pin(model_config)

        # if action_space is not a tuple, then calculate logits
        if not isinstance(self.action_space, gym.spaces.tuple.Tuple):
            self.build_logits_model_rectangle_pin()
            self.build_value_model_rectangle_pin()

    def build_logits_model_rectangle_pin(self):
        """Build the logits model.

        This model calculates the action logits using a dense layer where the
        input is the model encoding and the dense layer has n output neurons
        corresponding to the number of actions.
        """
        logits = tfkl.Dense(self.action_space.n)(self.encoding)
        self.logits_model = tf.keras.Model(
            [
                self.grid_input,
                self.component_feature_input,
                self.component_pin_feature_input,
                self.placement_mask_input,
            ],
            logits,
            name="logits_model",
        )

    def build_value_model_rectangle_pin(self):
        """Build the value model.

        This model calculates the value using a dense layer where the input is
        the model encoding and the dense layer has 1 output neuron corresponding
        to the value of the state.
        """
        value = tfkl.Dense(1)(self.encoding)
        self.value_model = tf.keras.Model(
            [
                self.grid_input,
                self.component_feature_input,
                self.component_pin_feature_input,
                self.placement_mask_input,
            ],
            value,
            name="value_model",
        )

    def build_encoding_rectangle_pin(self, model_config):
        """Build the encoding for the RectanglePinModel.

        Build the encoding for the RectanglePinModel by encoding the component
        features, component pin features, and placement mask, concatenating
        them with the grid encoding, and flattening the result.

        The dimensions of the encoding before concatenating to the grid
        are (max_num_components * component_encoding_dim + pin_encoding_dim + 4).

        Args:
            model_config (dict): Model config dict.
        """
        self.encode_component_feature(model_config)
        self.encode_pin_feature(model_config)
        self.encode_placement_mask(model_config)

        # concatenate the component and pin encodings with the placement mask
        self.processed_component_pin_encodings = tf.concat(
            [
                self.component_encoding,
                self.component_pin_encodings,
                self.placement_mask_one_hot,
            ],
            axis=2,
        )

        self.encoding_component_pins = tfkl.Flatten()(
            self.processed_component_pin_encodings
        )
        self.encoding = tfkl.Concatenate()(
            [self.encoding_grid, self.encoding_component_pins]
        )

    def encode_component_feature(self, model_config):
        """Encode the component feature array.

        Encode the component feature array by passing the
        features for each component through a dense layer with
        output dimension component_encoding_dim.

        The encoding is of size (max_num_components, component_encoding_dim).

        Args:
            model_config (dict): Model config dict.
        """
        self.component_encoding_dim = model_config["custom_model_config"][
            "component_feature_encoding_dimension"
        ]

        self.component_encoding = tfkl.Dense(self.component_encoding_dim)(
            self.component_feature_input
        )

    def encode_pin_feature(self, model_config):
        """Encode the component pin feature array.

        Encode the component pin feature array by going through each component,
        and for each pin in the component, passing the pin features through
        a dense layer with output dimension pin_encoding_dim. Then to find the
        encoding of the pins for the component, sum up the encodings of
        the pins for the component.

        The component pin feature array is of size (max_num_components,
        max_num_pins_per_component, pin_feature_vector_width) and the
        encoding is of size (max_num_components, pin_encoding_dim).

        Args:
            model_config (dict): Model config dict.
        """
        self.pin_encoding_dim = model_config["custom_model_config"][
            "pin_feature_encoding_dimension"
        ]
        # encode all the pins for a component into a single vector
        # initialise pin encoding layer
        self.pin_encoding_layer = tfkl.Dense(self.pin_encoding_dim)
        self.component_pin_encodings = []
        for i in range(model_config["custom_model_config"]["max_num_components"]):
            pins_encoding = self.pin_encoding_layer(
                self.component_pin_feature_input[:, i]
            )
            # sum up all the pins for a component
            pins_encoding = tf.reduce_sum(pins_encoding, axis=1)
            # append to list
            self.component_pin_encodings.append(pins_encoding)
        self.component_pin_encodings = tf.stack(self.component_pin_encodings, axis=1)

    def encode_placement_mask(self, model_config):
        """Encode the placement mask by one hot encoding it.

        The placement mask is of size (max_num_components, 1) where
        each element is an integer between 0 and 3 inclusive. The
        encoding is of size (max_num_components, 4).

        Args:
            model_config (dict): Model config dict.
        """
        # one-hot encode placement_mask_input
        self.placement_mask_one_hot = tf.one_hot(
            tf.cast(self.placement_mask_input[:, :, 0], tf.int32), 4
        )

    def preprocess(self, obs_dict):
        """Preprocess the obs_dict.

        This method preprocesses the obs_dict by:

            - Getting the the numerical feature and categorical feature arrays
                for all pins from the observation space dictionary. The numerical
                feature array is of size (max_num_components, max_num_pins_per_component,
                4) and the categorical feature array is of size (max_num_components,
                max_num_pins_per_component, 1).
            - One-hot encoding the categorical feature array for the pins
                which contains the net_id of a pin. The one-hot encoded array is of
                size (max_num_components, max_num_pins_per_component, 4).
            - Concatenating the one-hot encoded categorical feature array with the
                numerical feature array to form the all pins feature tensor. The
                all pins feature tensor is of size (max_num_components,
                max_num_pins_per_component, num_nets + 4 + 1).

        Args:
            obs_dict (Dict): Observation space dictionary.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Tuple of placement mask,
                placement mask, all pins numerical feature tensor, and all pins
                feature tensor.
        """
        placement_mask = tf.cast(obs_dict["obs"]["placement_mask"], tf.float32)
        all_components_feature = obs_dict["obs"]["all_components_feature"]

        all_pins_num_feature = obs_dict["obs"]["all_pins_num_feature"]
        all_pins_cat_feature = tf.cast(
            obs_dict["obs"]["all_pins_cat_feature"], tf.int32
        )

        # One-hot encode pin categorical feature net_id
        all_pins_cat_one_hot = tf.one_hot(
            all_pins_cat_feature[:, :, :, 0], self.max_num_nets + 1
        )
        all_pins_cat_one_hot = tf.ensure_shape(
            all_pins_cat_one_hot,
            [
                None,
                all_pins_cat_feature.shape[1],
                all_pins_cat_feature.shape[2],
                self.max_num_nets + 1,
            ],
        )

        # concat the one-hot encoded categorical features with the numerical features
        all_component_pins_feature = tf.concat(
            [all_pins_num_feature, all_pins_cat_one_hot], axis=-1
        )

        return placement_mask, all_components_feature, all_component_pins_feature

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
        (
            placement_mask,
            all_components_feature,
            all_component_pins_feature,
        ) = self.preprocess(input_dict)

        logits = self.logits_model(
            [
                input_dict["obs"]["grid"],
                all_components_feature,
                all_component_pins_feature,
                placement_mask,
            ]
        )

        self._value_out = self.value_model(
            [
                input_dict["obs"]["grid"],
                all_components_feature,
                all_component_pins_feature,
                placement_mask,
            ]
        )

        logits += tf.maximum(
            tf.math.log(input_dict["obs"]["action_mask"]), tf.float32.min
        )

        return logits, state
