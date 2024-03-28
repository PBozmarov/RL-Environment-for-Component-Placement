"""
Policy model for the rectangle component placement task where the components have pins and where
attention is used to create the component encodings.
"""

from agent.models.rectangle_pin_model import RectanglePinModel
from agent.models.model_building_blocks import Attention

from ray.rllib.utils.framework import try_import_tf  # type: ignore
import gym  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class RectanglePinAttnCompModel(RectanglePinModel):
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

    The grid encoding is created like in RectangleModel. The component features,
    component pin features, and placement mask are encoded like in RectanglePinModel
    and then concatenated along the feature axis to get an encoding of dimensions
    (max_num_components, component_encoding_dim + pin_encoding_dim + 4).

    Before flattening this encoding and concatenating it with the flattened grid
    encoding, it is passed through a self-attention layer. The output of the
    is of dimensions (max_num_components, attn_hidden_size), where attn_hidden_size
    is the size of the hidden layer in the self-attention layer. This output is
    then flattened and concatenated with the flattened grid encoding to get a
    single encoding.

    The concatenated encoding is passed through two separate dense layers to
    generate action logits and a value prediction.

    Attributes:
        encoding (tf.Tensor): Encoding of the input features.
        processed_component_pin_encodings (tf.Tensor): Processed component pin
            encodings.

    Note:
        Attention is performed on the encodings for the component and pin features together.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Initialise the RectanglePinAttnCompModel.

        Args:
            obs_space (gym.spaces.Dict): Observation space.
            action_space (gym.spaces.Tuple): Action space.
            num_outputs (int): Number of outputs.
            model_config (Dict): Model config dictionary.
            name (str): Name of the model.
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.build_encoding_rectangle_pin_attn(model_config)
        # if action_space is not a tuple, then calculate logits
        if not isinstance(self.action_space, gym.spaces.tuple.Tuple):
            self.build_logits_model_rectangle_pin()
            self.build_value_model_rectangle_pin()

    def build_encoding_rectangle_pin_attn(self, model_config):
        """Build the encoding for the RectanglePinAttnCompModel.

        This method builds an encoding using component encodings, pin encodings,
        and the one-hot encoded placement mask. The encoding is built by performing
        self-attention on the processed_component_pin_encodings attribute from
        RectanglePinModel.

        The output of the self-attention layer is of dimensions (max_num_components,
        attn_hidden_size), where attn_hidden_size is the size of the hidden layer
        in the self-attention layer. This output is then flattened and concatenated
        with the flattened grid encoding to get a single encoding of dimensions
        (max_num_components * attn_hidden_size + grid_encoding_dim).

        Args:
            model_config (Dict): Model config dictionary.
        """
        # create the attention layer
        attn_hidden_size = model_config["custom_model_config"]["attn_hidden_size"]
        attention_output = Attention(attn_hidden_size)(
            self.processed_component_pin_encodings
        )

        encoding_component_pins = tfkl.Flatten()(attention_output)
        self.encoding = tfkl.Concatenate()(
            [self.encoding_grid, encoding_component_pins]
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
