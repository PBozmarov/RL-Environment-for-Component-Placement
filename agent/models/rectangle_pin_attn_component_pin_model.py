"""
Policy model for the rectangle component placement task where the components have pins and where
attention is used to encode both the component features and the component pin features.
"""

from agent.models.rectangle_pin_attn_component_model import RectanglePinAttnCompModel
from agent.models.model_building_blocks import Attention
from ray.rllib.utils.framework import try_import_tf  # type: ignore
import gym  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class RectanglePinAttnCompPinModel(RectanglePinAttnCompModel):
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
    component pin features, and placement mask are encoded like in RectanglePinModel,
    except that the component pin features are passed through a self-attention
    layer to obtain the encodings of all the pins for each component. These encodings
    are then concatenated along the feature axis to get an encoding of dimensions
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
        Attention is performed on the encodings the component and pin features together,
        as well as for getting the pin encodings.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """Initialise the RectanglePinAttnCompPinModel.

        Args:
            obs_space (gym.spaces): Observation space.
            action_space (gym.spaces): Action space.
            num_outputs (int): Number of outputs.
            model_config (Dict): Model config dictionary.
            name (str): Name of the model.
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # build the encoding
        self.build_encoding_rectangle_comp_pin(model_config)

        # if action_space is not a tuple, then calculate logits
        if not isinstance(self.action_space, gym.spaces.tuple.Tuple):
            self.build_logits_model_rectangle_pin()
            self.build_value_model_rectangle_pin()

    def build_encoding_rectangle_comp_pin(self, model_config):
        """Build the encoding for the RectanglePinAttnCompModel.

        This method builds an encoding using component encodings, pin encodings,
        and the one-hot encoded placement mask. The encodings for the placement mask
        and component encodings are built like in RectanglePinAttnCompModel. To get
        the pin encodings we now use a self-attention layer.

        These encodings are then concatenated along the feature axis to get an
        encoding of dimensions (max_num_components, component_encoding_dim +
        attn_hidden_size_pin + 4). Self-attention is again performed on
        this encoding to get an encoding of dimensions (max_num_components,
        attn_hidden_size). This output is then flattened and concatenated with
        the flattened grid encoding to get a single encoding of dimensions
        (max_num_components * attn_hidden_size + grid_encoding_dim).

        Args:
            model_config (Dict): Model config dictionary.
        """
        # encode all the pins for a component into a single vector
        # initialise pin encoding layer
        self.encode_pin_feature_attn(model_config)

        # concatenate the component and pin encodings with the placement mask
        self.processed_component_pin_encodings = tf.concat(
            [
                self.component_encoding,
                self.component_pin_encodings,
                self.placement_mask_one_hot,
            ],
            axis=2,
        )

        # create the attention layer
        attn_hidden_size = model_config["custom_model_config"]["attn_hidden_size"]
        attention_output = Attention(attn_hidden_size)(
            self.processed_component_pin_encodings
        )

        self.encoding_component_pins = tfkl.Flatten()(attention_output)
        self.encoding = tfkl.Concatenate()(
            [self.encoding_grid, self.encoding_component_pins]
        )

    def encode_pin_feature_attn(self, model_config):
        """Encode the component pin feature array.

        Encode the component pin feature array by going through each component,
        and for each pin in the component, passing the pin features through
        a dense layer with output dimension pin_encoding_dim. Then to find the
        encoding of the pins for the component, self-attention is performed on
        these pin encodings. The output of the self-attention layer is then
        flattened to get a single encoding for the pins of the component.

        The component pin feature array is of size (max_num_components,
        max_num_pins_per_component, pin_feature_vector_width) and the
        encoding before flattening is of size (max_num_components,
        attn_hidden_size_pin), where  attn_hidden_size_pin is the size of the
        hidden layer in the self-attention layer. The encoding after flattening is of size
        max_num_components * attn_hidden_size_pin.

        Args:
            model_config (dict): Model config dict.
        """
        pin_encoding_layer = tfkl.Dense(self.pin_encoding_dim)

        # create the attention layers for pin
        attn_hidden_size_pin = model_config["custom_model_config"][
            "attn_hidden_size_pin"
        ]
        query_pin = tfkl.Dense(attn_hidden_size_pin)
        key_pin = tfkl.Dense(attn_hidden_size_pin)
        value_pin = tfkl.Dense(attn_hidden_size_pin)

        component_pin_encodings = []
        for i in range(model_config["custom_model_config"]["max_num_components"]):
            pins_encoding = pin_encoding_layer(self.component_pin_feature_input[:, i])
            # get pin query, key and value matrices
            query_pin_mat = query_pin(pins_encoding)
            key_pin_mat = key_pin(pins_encoding)
            value_pin_mat = value_pin(pins_encoding)

            attention_weights_pin = tf.matmul(
                query_pin_mat, key_pin_mat, transpose_b=True
            )
            attention_weights_pin = tf.nn.softmax(attention_weights_pin, axis=-1)

            attention_output_pin = tf.matmul(attention_weights_pin, value_pin_mat)
            dense_outputs_pin = tf.nn.relu(attention_output_pin)

            flatten_pin_attention = tfkl.Flatten()(dense_outputs_pin)

            # append to list
            component_pin_encodings.append(flatten_pin_attention)

        self.component_pin_encodings = tf.stack(component_pin_encodings, axis=1)

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
