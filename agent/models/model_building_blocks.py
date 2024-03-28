"""
Building blocks for different policy network architectures.
"""

from ray.rllib.utils.framework import try_import_tf  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class ConvBlock:
    """Convolutional block that consists of convolutional layer, batch normalization layer, and activation function.

    Attributes:
        num_conv_filters (int): Number of filters in the convolutional layer.
        conv_kernel_size (tuple): Height and width of the 2D convolution window.
        conv_padding (str): Padding method. "valid" by default.
        activation (function): Activation function. tf.nn.relu by default.
        max_pool (bool): Whether to use max pooling layer. False by default.
        max_pool_kernel_size (int): Size of the max pooling window. 4 by default.
    """

    def __init__(
        self,
        num_conv_filters,
        conv_kernel_size,
        conv_padding="valid",
        activation=tf.nn.relu,
        max_pool=False,
        max_pool_kernel_size=4,
    ):
        """Initializes an instance of the ConvBlock class.

        Args:
            num_conv_filters (int): Number of filters in the convolutional layer.
            conv_kernel_size (tuple): Height and width of the 2D convolution window.
            conv_padding (str): Padding method. "valid" by default.
            activation (function): Activation function. tf.nn.relu by default.
            max_pool (bool): Whether to use max pooling layer. False by default.
            max_pool_kernel_size (int): Size of the max pooling window. 4 by default.
        """
        self.num_conv_filters = num_conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_padding
        self.activation = activation
        self.max_pool = max_pool
        self.max_pool_kernel_size = max_pool_kernel_size

    def __call__(self, x):
        """Applies convolution, batch normalization, activation function, and optionally max pooling to the input.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor.
        """
        # If x is two dimensional, add a dimension
        if len(x.shape) == 3:
            x = tf.expand_dims(x, axis=-1)

        x = tfkl.Conv2D(
            filters=self.num_conv_filters,
            kernel_size=self.conv_kernel_size,
            padding=self.conv_padding,
        )(x)
        x = tfkl.BatchNormalization()(x)
        x = self.activation(x)

        if self.max_pool:
            x = tf.nn.max_pool2d(
                x,
                ksize=self.max_pool_kernel_size,
                strides=self.max_pool_kernel_size,
                padding="VALID",
            )
        return x


class ConvBlocks:
    """Class for creating multiple convolution blocks.

    Attributes:
        num_conv_blocks (int): The number of convolution blocks to create.
        num_conv_filters (int): The number of filters in each convolution block.
        conv_kernel_size (int or tuple): The size of the convolution kernel.
        activation (function): The activation function to use after each convolutional layer. Default is tf.nn.relu.
        max_pool (bool): Whether or not to use max pooling after each convolutional layer. Default is False.
        max_pool_kernel_size (int or tuple): The size of the max pooling kernel. Default is 4.
        conv_padding (str): The padding to use for convolutional layers. Default is "valid".
    """

    def __init__(
        self,
        num_conv_blocks,
        num_conv_filters,
        conv_kernel_size,
        activation=tf.nn.relu,
        max_pool=False,
        max_pool_kernel_size=4,
        conv_padding="valid",
    ):
        """Initializes an instance of the ConvBlocks class.

        Args:
            num_conv_blocks (int): The number of convolution blocks to create.
            num_conv_filters (int): The number of filters in each convolution block.
            conv_kernel_size (int or tuple): The size of the convolution kernel.
            activation (function): The activation function to use after each convolutional layer. Default is tf.nn.relu.
            max_pool (bool): Whether or not to use max pooling after each convolutional layer. Default is False.
            max_pool_kernel_size (int or tuple): The size of the max pooling kernel. Default is 4.
            conv_padding (str): The padding to use for convolutional layers. Default is "valid".
        """
        self.num_conv_blocks = num_conv_blocks
        self.num_conv_filters = num_conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_padding
        self.activation = activation
        self.max_pool = max_pool
        self.max_pool_kernel_size = max_pool_kernel_size

    def __call__(self, x):
        """Creates multiple multiple convolution blocks.

        Args:
            x (tf.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            List[tf.Tensor]: A list of tensors, where each tensor is the output of a convolution block.
        """
        conv_blocks = []
        for _ in range(self.num_conv_blocks):
            x = ConvBlock(
                self.num_conv_filters,
                self.conv_kernel_size,
                self.conv_padding,
                self.activation,
                self.max_pool,
                self.max_pool_kernel_size,
            )(x)
            conv_blocks.append(x)
        return conv_blocks[-1]


class Attention:
    """Class for self-attention mechanism.

    Attributes:
        attn_hidden_size (int): The size of the hidden layer used to calculate the attention scores.
    """

    def __init__(self, attn_hidden_size):
        """Initializes an instance of the Attention class.

        Args:
            attn_hidden_size (int): The size of the hidden layer used to calculate the attention scores.
        """
        self.attn_hidden_size = attn_hidden_size

    def __call__(self, x):
        """Computes the self-attention mechanism.

        Args:
            x (tf.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            tf.Tensor: The output tensor of shape (batch_size, sequence_length, attn_hidden_size).
        """
        query = tfkl.Dense(self.attn_hidden_size)(x)
        key = tfkl.Dense(self.attn_hidden_size)(x)
        value = tfkl.Dense(self.attn_hidden_size)(x)

        attention_weights = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        attention_output = tf.matmul(attention_weights, value)
        attention_output = tf.nn.relu(attention_output)

        return attention_output
