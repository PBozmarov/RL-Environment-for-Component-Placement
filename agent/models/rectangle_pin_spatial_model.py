"""
Policy model for the rectangle component placement task where the components have pins and where
the pins, components and nets are represented spatially.
"""

from agent.models.square_model import SquareModel
from agent.models.model_building_blocks import ConvBlocks, Attention
from ray.rllib.utils.framework import try_import_tf  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class RectanglePinSpatialModel(SquareModel):
    """A model to predict the action logits and value for placing components
    with pins on a 2D grid.

    This model takes as inputs a 2D grid, a pin grid, a component grid, and a
    placement mask array. The model learns an encoding of these features by
    using the encodings for:

        - The grid (same as SquareModel)
        - The pin grid
        - The component grid
        - The placement mask

    The grid encoding is created like in SquareModel. The pin grid is a 3d grid
    with the same height and width as the 2d grid, but with a depth of
    max_num_nets + 1. The first channel is a mask for the pins, the remaining
    channels are a one-hot encoding of the net ids. The component grid is a 4d
    spatial representation of components with pins on them.

    The placement mask is encoded by simply one-hot encoding it. The encodings grid,
    pin grid, component and placement mask are concatenated along the feature axis
    to get an encoding of dimensions. This encoding is then flattened and concatenated
    with the flattened grid encoding to get a single encoding.

    The concatenated encoding is passed through two separate dense layers to
    generate action logits and a value prediction.

    Attributes:
        pin_grid_input (tf.Tensor): Input tensor for pin grid.
        component_grid_input (tf.Tensor): Input tensor for component grid.
        placement_mask_input (tf.Tensor): Input tensor for placement mask.
        encoding_pin_grid (tf.Tensor): Encoded flattened pin grid features.
        encoding_components (tf.Tensor): Encoded flattened component grid features.
        encoding (tf.Tensor): Encoded flattened features.
        logits_model (tf.keras.Model): Model to predict action logits.
        value_model (tf.keras.Model): Model to predict value.
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

        self.pin_grid_input = tfkl.Input(
            shape=(
                model_config["custom_model_config"]["height"],
                model_config["custom_model_config"]["width"],
                model_config["custom_model_config"]["max_num_nets"] + 1,
            ),
            dtype=tf.float32,
            name="pin_grid_input",
        )

        self.component_grid_input = tfkl.Input(
            shape=(
                model_config["custom_model_config"]["max_num_components"],
                model_config["custom_model_config"]["max_component_h"],
                model_config["custom_model_config"]["max_component_w"],
                model_config["custom_model_config"]["max_num_nets"] + 1,
            ),
            dtype=tf.float32,
            name="component_grid_input",
        )

        self.placement_mask_input = tfkl.Input(
            shape=(model_config["custom_model_config"]["max_num_components"], 1),
            dtype=tf.float32,
            name="placement_mask_input",
        )

        self.build_encoding_spatial(model_config)
        self.build_logits_model_spatial()
        self.build_value_model_spatial()

    def build_encoding_spatial(self, model_config):
        """Build the encoding for the model by computing pin encoding,
        component encoding and appending them to the grid encoding.

        Args:
            model_config (dict): Model config dict.
        """
        self.encode_pin_grid(model_config)
        self.encode_component_grid(model_config)

        self.encoding = tfkl.Concatenate()(
            [self.encoding_grid, self.encoding_pin_grid, self.encoding_components]
        )

    def build_logits_model_spatial(self):
        """Build the logits model.

        The logits model takes as input the grid, pin grid, component grid and
        placement mask and outputs action logits.
        """
        logits = tfkl.Dense(self.action_space.n)(self.encoding)
        self.logits_model = tf.keras.Model(
            [
                self.grid_input,
                self.pin_grid_input,
                self.component_grid_input,
                self.placement_mask_input,
            ],
            logits,
            name="logits_model",
        )

    def build_value_model_spatial(self):
        """Build the value model.

        The value model takes as input the grid, pin grid, component grid and
        placement mask and outputs a value prediction.
        """
        value = tfkl.Dense(1)(self.encoding)
        self.value_model = tf.keras.Model(
            [
                self.grid_input,
                self.pin_grid_input,
                self.component_grid_input,
                self.placement_mask_input,
            ],
            value,
            name="value_model",
        )

    def encode_pin_grid(self, model_config):
        """Encode the pin grid.

        The pin grid is processed by passing it through a series of convolutional
        blocks. The output of the last convolutional block is flattened to get
        a single encoding.

        Args:
            model_config (dict): Model config dict.
        """
        processed_pin_grid = ConvBlocks(
            num_conv_blocks=model_config["custom_model_config"]["num_conv_blocks"],
            num_conv_filters=model_config["custom_model_config"]["num_conv_filters"],
            conv_kernel_size=model_config["custom_model_config"]["conv_kernel_size"],
            activation=model_config["custom_model_config"]["activation"],
            max_pool=model_config["custom_model_config"]["max_pool"],
            max_pool_kernel_size=model_config["custom_model_config"][
                "max_pool_kernel_size"
            ],
        )(self.pin_grid_input)

        self.encoding_pin_grid = tfkl.Flatten()(processed_pin_grid)

    def encode_component_grid(self, model_config):
        """Encode the component grid.

        The component grid is processed by passing it through a series of
        convolutional blocks. The output of the last convolutional block is
        flattened to get a single encoding and then appended to the placement
        mask one hot encoding.

        Args:
            model_config (dict): Model config dict.
        """
        components_encodings = []
        for i in range(model_config["custom_model_config"]["max_num_components"]):
            # do convolution with padding on the component grid input
            processed_component_grid = ConvBlocks(
                num_conv_blocks=model_config["custom_model_config"][
                    "num_conv_blocks_component_grid"
                ],
                num_conv_filters=model_config["custom_model_config"][
                    "num_conv_filters_component_grid"
                ],
                conv_kernel_size=model_config["custom_model_config"][
                    "conv_kernel_size_component_grid"
                ],
                activation=model_config["custom_model_config"][
                    "activation_component_grid"
                ],
                max_pool=model_config["custom_model_config"]["max_pool_component_grid"],
                max_pool_kernel_size=model_config["custom_model_config"][
                    "max_pool_kernel_size_component_grid"
                ],
                conv_padding=model_config["custom_model_config"][
                    "conv_padding_component_grid"
                ],
            )(self.component_grid_input[:, i])

            # do 1x1 convolution to reduce the dimension of the component grid input
            # flatten the component grid input
            processed_component_grid = tfkl.Flatten()(processed_component_grid)
            components_encodings.append(processed_component_grid)

        # convert list of encodings to tensor
        components_encodings = tf.stack(components_encodings, axis=0)
        components_encodings = tf.transpose(components_encodings, perm=[1, 0, 2])

        # one-hot encode placement_mask_input
        placement_mask_one_hot = tf.one_hot(
            tf.cast(self.placement_mask_input[:, :, 0], tf.int32), 4
        )

        self.components_encodings = tf.concat(
            [components_encodings, placement_mask_one_hot], axis=2
        )
        # attention on component encoding
        component_attn_hidden_size = model_config["custom_model_config"][
            "component_attn_hidden_size"
        ]
        component_attn_output = Attention(component_attn_hidden_size)(
            self.components_encodings
        )

        # flatten the component grid input
        self.encoding_components = tfkl.Flatten()(component_attn_output)

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

        placement_mask = tf.cast(input_dict["obs"]["placement_mask"], tf.float32)

        logits = self.logits_model(
            [
                input_dict["obs"]["grid"],
                input_dict["obs"]["pin_grid"],
                input_dict["obs"]["component_grid"],
                placement_mask,
            ]
        )

        self._value_out = self.value_model(
            [
                input_dict["obs"]["grid"],
                input_dict["obs"]["pin_grid"],
                input_dict["obs"]["component_grid"],
                placement_mask,
            ]
        )

        logits += tf.maximum(
            tf.math.log(input_dict["obs"]["action_mask"]), tf.float32.min
        )

        return logits, state
