"""
Policy model for the rectangle component placement task where the components have pins and where
attention is used to encode both the component features and the component pin features. The grid
is not used in the encoding.
"""

from agent.models.rectangle_pin_attn_component_pin_model import (
    RectanglePinAttnCompPinModel,
)
from ray.rllib.utils.framework import try_import_tf  # type: ignore
import gym  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class RectanglePinAttnAllNoGridModel(RectanglePinAttnCompPinModel):
    """A model to predict the action logits and value for placing components
    with pins on a 2D grid.

    This model takes as inputs a component features array, a component pin
    features array corresponding to the features of pins on components,
    and a placement mask array. The model learns an encoding of these
    features by using the encodings for:

        - The component features
        - The component pin features
        - The placement mask

    The component features, component pin features, and placement mask are
    encoded like in RectanglePinAttnCompPinModel. These encodings are
    concatenated along the feature axis to get an encoding of dimensions
    (max_num_components, component_encoding_dim + pin_encoding_dim + 4).

    Before flattening this encoding it is passed through a self-attention layer.
    The output of the is of dimensions (max_num_components, attn_hidden_size),
    where attn_hidden_size is the size of the hidden layer in the self-attention layer.
    This output is then flattened and used as the encoding for the model.

    The encoding is passed through two separate dense layers to generate action
    logits and a value prediction.

    Attributes:
        encoding (tf.Tensor): Encoding of the input features.
        processed_component_pin_encodings (tf.Tensor): Processed component pin
            encodings.

    Note:
        Attention is performed on the encodings the component and pin features together,
        as well as for getting the pin encodings.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Initialise the RectanglePinAttnCompPinModel.

        Args:
            obs_space (gym.spaces.Dict): Observation space.
            action_space (gym.spaces.Tuple): Action space.
            num_outputs (int): Number of outputs.
            model_config (Dict): Model configuration.
            name (str): Name of the model.
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.encoding = self.encoding_component_pins

        # if action_space is not a tuple, then calculate logits
        if not isinstance(self.action_space, gym.spaces.tuple.Tuple):
            self.build_logits_model_rectangle_pin()
            self.build_value_model_rectangle_pin()

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
