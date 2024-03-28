"""
Factorised action distributions for the dummy environment.
"""

from ray.rllib.models.tf.tf_action_dist import TFActionDistribution  # type: ignore
from ray.rllib.models.action_dist import ActionDistribution  # type: ignore
from ray.rllib.utils.typing import TensorType, Union, Tuple  # type: ignore
from ray.rllib.utils.annotations import override  # type: ignore

import tensorflow_probability as tfp  # type: ignore
from ray.rllib.utils.framework import try_import_tf  # type: ignore

import gym  # type: ignore
import numpy as np


_, tf, _ = try_import_tf()
tfkl = tf.keras.layers


class Categorical(TFActionDistribution):
    """Categorical distribution for discrete action spaces."""

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        """Sample an action from the distribution by
            taking the argmax of the logits.

        Returns:
            TensorType: Action.
        """
        return tf.math.argmax(self.inputs, axis=-1)

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        """Calculate the log probability of the given action.

        Args:
            x (TensorType): Action.

        Returns:
            TensorType: Log probability of the action
        """
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.inputs, labels=tf.cast(x, tf.int32)
        )

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        """Calculate the entropy of the distribution.

        Returns:
            TensorType: Entropy of the distribution.
        """
        a0 = self.inputs - tf.reduce_max(self.inputs, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution) -> TensorType:
        """Calculate the KL divergence between this distribution and the other.

        Args:
            other (ActionDistribution): Other distribution.

        Returns:
            TensorType: KL divergence.
        """
        a0 = self.inputs - tf.reduce_max(self.inputs, axis=-1, keepdims=True)
        a1 = other.inputs - tf.reduce_max(other.inputs, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(
            p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1
        )

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        """Sample an action from the distribution.

        Returns:
            TensorType: Action.
        """
        dist = tfp.distributions.Categorical(self.inputs, dtype=tf.int64)
        return dist.sample()

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space, model_config):
        """Calculate the required model output shape for the given action space.

        Args:
            action_space (gym.Space): Action space.
            model_config (dict): Model configuration.

        Returns:
            int: Required model output shape.
        """
        return action_space.n


class FactorisedActionDistributionOrientation(TFActionDistribution):
    r"""Factorised action distribution for discrete action spaces.

    The factorization of the action space used is:
        :math:`p(\text{orientation}, x, y) = p(\text{orientation})p(x \mid \text{orientation}) p (y \mid \text{orientation}, x)`

    The distribution takes as input the concatenation of the encoding and the action mask as
    well as the model used to produce the inputs. For sampling, the encoding is used as input
    to the orientation model which produces logits for the orientation. These logits are masked
    using the action mask and then used to sample an orientation, where the distribution being
    sampled from is a categorical distribution over the orientations.

    The sampled orientation is then one-hot encoded and concatenated with the encoding
    as input to the x model. The x model produces logits for the x coordinate. These logits
    are then masked using the given orientation and used to sample an x coordinate, where the
    distribution being sampled from is a categorical distribution over the x coordinates.

    The sampled x coordinate is then normalized and concatenated with the encoding
    and the sampled orientation as input to the y model. The y model produces logits for the
    y coordinate. These logits are then masked using the given orientation and x coordinate
    and used to sample a y coordinate, where the distribution being sampled from is a
    categorical distribution over the y coordinates.

    Note:
        - The sampled x coordinate is normalized by the number of columns in the grid
            to be in the range [0, 1] wen being used as input to the y model.
    """
    # Define class variable for model output shape
    model_output_shape = None

    def __init__(self, inputs, model):
        """Initialise the distribution.

        Args:
            inputs (TensorType): Input vector to sample from.
                The input vector is the concatenation of the encoding and the action mask.
            model (ModelV2): Reference to model producing the inputs.
        """
        # Get the encoding and action mask sizes from model
        self.model = model
        self.action_space_shape = model.action_space_shape
        self.action_space_size = model.action_space_size
        self.encoding_shape = model.encoding_shape
        self.encoding_size = int(model.encoding_size)
        self.grid_size = self.action_space_shape[1] * self.action_space_shape[2]
        FactorisedActionDistributionOrientation.model_output_shape = model.output_shape

        # Split the inputs into the encoding and the action mask
        self.encoding, self.action_mask = tf.split(
            inputs,
            [
                self.encoding_size,
                self.action_space_size,
            ],
            axis=1,
        )

        # Reshape the action mask to be of size (orientations, height, width)
        desired_action_mask_shape = self.action_space_shape
        self.action_mask = tfkl.Reshape((desired_action_mask_shape))(self.action_mask)

        # Get the distribution for orientation
        self.orientation_distribution = self._orientation_distribution()
        TFActionDistribution.__init__(self, self.encoding, model)

    @staticmethod
    def required_model_output_shape(
        action_space: gym.Space, model_config: dict
    ) -> Union[int, np.ndarray]:
        """Calculate the required model output shape for the given action space.

        Args:
            action_space (gym.Space): Action space.
            model_config (dict): Model configuration.

        Returns:
            int: required model output shape
        """
        output_shape = FactorisedActionDistributionOrientation.model_output_shape
        return output_shape

    @override(TFActionDistribution)
    def _build_sample_op(self) -> tuple:
        """Sample an action from the distribution.

        Sample using the orientation, x and y models.

        Returns:
            TensorType: Action.
        """
        # Sample orientation from the distribution of orientations
        orientation_distribution = self.orientation_distribution
        orientation = orientation_distribution.sample()

        # Sample x from the distribution of x given orientation
        x_distribution = self._x_distribution(orientation)
        x = x_distribution.sample()

        # Sample y from the distribution of y given orientation and x
        y_distribution = self._y_distribution(orientation, x)
        y = y_distribution.sample()

        return orientation, x, y

    def deterministic_sample(self) -> tuple:
        """Sample an action from the distribution by taking
        the argmax of the logits.

        Sample using the orientation, x and y models.

        Returns:
            TensorType: Action.
        """
        # Sample orientation from the distribution of orientations
        orientation_distribution = self.orientation_distribution
        orientation = orientation_distribution.deterministic_sample()

        # Sample x from the distribution of x given orientation
        x_distribution = self._x_distribution(orientation)
        x = x_distribution.deterministic_sample()

        # Sample y from the distribution of y given orientation and x
        y_distribution = self._y_distribution(orientation, x)
        y = y_distribution.deterministic_sample()
        return orientation, x, y

    @override(ActionDistribution)
    def entropy(self):
        """Calculate the entropy of the distribution.

        Returns:
            TensorType: Entropy of the distribution.
        """
        # Calculate the entropy of the orientation distribution
        # by sampling from it and then calculating the entropy
        orientation_distribution = self.orientation_distribution
        orientation = orientation_distribution.sample()
        orientation_entropy = orientation_distribution.entropy()

        # Calculate the entropy of the x distribution given the orientation
        x_distribution = self._x_distribution(orientation)
        x = x_distribution.sample()
        x_entropy = x_distribution.entropy()

        # Calculate the entropy of the y distribution given the orientation and x
        y_distribution = self._y_distribution(orientation, x)
        y_entropy = y_distribution.entropy()
        return orientation_entropy + x_entropy + y_entropy

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution):
        """Calculate the KL divergence between this distribution and another.

        Args:
            other (ActionDistribution): Other distribution.

        Returns:
            TensorType: KL divergence between this distribution and the other.
        """
        # Calculate the KL divergence between the orientation distributions
        orientation_distribution = self.orientation_distribution
        other_orientation_distribution = other.orientation_distribution
        orientation = orientation_distribution.sample()
        orientation_kl = orientation_distribution.kl(other_orientation_distribution)

        # Calculate the KL divergence between the x distributions given the orientation
        x_distribution = self._x_distribution(orientation)
        other_x_distribution = other._x_distribution(orientation)
        x = x_distribution.sample()
        x_kl = x_distribution.kl(other_x_distribution)

        # Calculate the KL divergence between the y distributions given the orientation and x
        y_distribution = self._y_distribution(orientation, x)
        other_y_distribution = other._y_distribution(orientation, x)
        y_kl = y_distribution.kl(other_y_distribution)

        return orientation_kl + x_kl + y_kl

    @override(ActionDistribution)
    def logp(
        self, actions: Union[TensorType, Tuple[TensorType, TensorType]]
    ) -> TensorType:
        """Calculate the log probability of the given actions.

        Args:
            actions (Union[TensorType, Tuple[TensorType, TensorType]]): Actions.

        Returns:
            TensorType: Log probability of the given actions.
        """
        # Calculate the log probability of the orientation
        orientation, x, y = self.unpack_actions(actions)
        orientation_logp = self.orientation_distribution.logp(orientation)

        # Calculate the log probability of x given the orientation
        x_distribution = self._x_distribution(orientation)
        x_logp = x_distribution.logp(x)

        # Calculate the log probability of y given the orientation and x
        y_distribution = self._y_distribution(orientation, x)
        y_logp = y_distribution.logp(y)
        return orientation_logp + x_logp + y_logp

    def unpack_actions(
        self, actions: Union[TensorType, Tuple[TensorType, TensorType]]
    ) -> Tuple[TensorType, TensorType]:
        """Unpack the given actions into orientation, x, and y.

        Args:
            actions (Union[TensorType, Tuple[TensorType, TensorType]]): Actions.

        Returns:
            Tuple[TensorType, TensorType]: Orientation, x, and y.
        """
        # Unpack the actions into orientation, x, and y
        if isinstance(actions, tuple):
            orientation, x, y = list(zip(actions))
            orientation, x, y = orientation[0], x[0], y[0]

        # Unpack the actions into orientation, x, and y
        else:
            # Reshape the actions into a 2D tensor
            actions = tf.reshape(actions, (-1, 3))

            # Split the actions into orientation, x, and y
            orientation, x, y = tf.split(actions, num_or_size_splits=3, axis=-1)
            orientation = tf.squeeze(orientation, axis=-1)

            # Reshape the x and y into a 1D tensor
            x = tf.squeeze(x, axis=-1)
            y = tf.squeeze(y, axis=-1)
        return orientation, x, y

    def _orientation_distribution(self):
        """Create the distribution of orientations.

        Using the model encoding, calculate the logits of the orientations and
        use them to create a categorical distribution.

        Before using the logits, the logits are masked. The mask for the
        orientation is calculated from the full action mask by using checking if
        there is at least one valid action for each orientation.

        Returns:
            ActionDistribution: Distribution of orientations.
        """
        # Calculate the logits of the orientations
        orientation_logits = self.model.orientation_model(self.encoding)

        # Find the ratio of the sum of the action mask to the grid size
        # for the orientations
        action_mask_for_orientation = tf.reduce_max(self.action_mask, axis=(2, 3))

        # Mask the logits of the orientations
        orientation_logits_masked = orientation_logits + tf.maximum(
            tf.math.log(action_mask_for_orientation), tf.float32.min
        )
        return Categorical(
            inputs=orientation_logits_masked,
            model=self.model,
        )

    def _x_distribution(self, orientation: TensorType) -> ActionDistribution:
        """Create the distribution of x given the orientation.

        The sampled orientation is one-hot encoded and concatenated with the
        model encoding. Using the concatenated encoding, logits for the x coordinate are
        calculated and used to create a categorical distribution.

        Before using the logits, the logits are masked. The action mask for x given the
        orientation is calcualted from the full action mask in the following way:

            - Subset the action mask for the given orientation
            - Check if there is at least one valid action for each x coordiante

        Args:
            orientation (TensorType): Orientation (as type int32).

        Returns:
            ActionDistribution: Distribution of x given the orientation.
        """
        # Calculate the logits of the x given the orientation
        orientation_int = tf.cast(orientation, tf.int32)
        num_orientations = self.action_space_shape[0]
        one_hot_orientation = tf.one_hot(orientation, num_orientations, axis=-1)
        one_hot_orientation = tf.cast(one_hot_orientation, tf.float32)

        # Calculate the logits of the x given the orientation
        x_logits = self.model.x_model([self.encoding, one_hot_orientation])

        # Get action mask corresponding to given orientation
        action_mask_for_x = tf.gather(
            self.action_mask, orientation_int, axis=1, batch_dims=1
        )
        action_mask_for_x = tf.reduce_max(action_mask_for_x, axis=2)

        # Mask the logits of the x given the orientation
        x_logits_masked = x_logits + tf.maximum(
            tf.math.log(action_mask_for_x), tf.float32.min
        )
        return Categorical(
            inputs=x_logits_masked,
            model=self.model,
        )

    def _y_distribution(self, orientation, x):
        """Create the distribution of y given the orientation and x.

        The sampled orientation is one-hot encoded and the sampled x coordinate
        is normalized. These are then concatenated with the model encoding. Using the
        concatenated encoding, logits for the y coordinate are calculated and used to
        create a categorical distribution.

        Before using the logits, the logits are masked. The action mask for y
        given the orientation and x is calcualted from the full action
        mask by subsetting the action mask for the given orientation and x.

        Args:
            orientation (TensorType): Orientation (as type int32).
            x (TensorType): x coordinate (as type int32).

        Returns:
            ActionDistribution: Distribution of y given the orientation and x.
        """
        # Create inputs for the y action model
        orientation = tf.cast(orientation, tf.int32)
        x = tf.cast(x, tf.int32)
        num_orientations = self.action_space_shape[0]
        num_x = self.action_space_shape[1]
        one_hot_orientation = tf.one_hot(orientation, num_orientations, axis=-1)
        one_hot_orientation = tf.cast(one_hot_orientation, tf.float32)
        x_input = x / num_x
        x_input = tf.cast(x_input, tf.float32)

        # Calculate the logits of the y given the orientation and x
        y_logits = self.model.y_model([self.encoding, one_hot_orientation, x_input])

        # Find the action mask for the given orientation and x
        action_mask_for_y = tf.gather(
            self.action_mask, orientation, axis=1, batch_dims=1
        )
        action_mask_for_y = tf.gather(action_mask_for_y, x, axis=1, batch_dims=1)

        # Mask the logits of the y given the orientation and x
        y_logits_masked = y_logits + tf.maximum(
            tf.math.log(action_mask_for_y), tf.float32.min
        )

        return Categorical(
            inputs=y_logits_masked,
            model=self.model,
        )


class FactorisedActionDistributionCoordinates(TFActionDistribution):
    r"""Factorised action distribution for discrete action spaces.

    The factorization of the action space used is:
        :math:`p(\text{orientation}, x, y) = p(x)p(y \mid x) p (\text{orientation} \mid x, y)`

    The distribution takes as input the concatenation of the encoding and the action mask as
    well as the model used to produce the inputs. For sampling, the encoding is used as input
    to the x model which produces logits for the x coordiante. These logits are masked
    using the action mask and then used to sample an x coordinate, where the distribution being
    sampled from is a categorical distribution over the x coordinates.

    The sampled x coordinate is normalized and concatenated with the encoding
    as input to the y model. The y model produces logits for the y coordinate. These logits
    are then masked using the given x coordinate and used to sample a y coordinate, where the
    distribution being sampled from is a categorical distribution over the y coordinates.

    The sampled y coordinate is then normalized and concatenated with the encoding
    and the normalied sampled x coodinate as input to the y model. The orienation model produces
    logits for the orienation coordinate. These logits are then masked using the given
    y and x coordinate and used to sample an orientation, where the distribution being
    sampled from is a categorical distribution over the orientations coordinates.

    Note:
        - The sampled x and y coordinates are normalized to be in
            the range [0, 1] when being used as input to the y and orientation models.
    """
    # Define class variable for model output shape
    model_output_shape = None

    def __init__(self, inputs, model):
        """Initialise the distribution.

        Args:
            inputs (TensorType): Input vector to sample from.
                The input vector is the concatenation of the encoding and the action mask.
            model (ModelV2): Reference to model producing the inputs.
        """
        # Get the encoding and action mask sizes from model
        self.model = model
        self.action_space_shape = model.action_space_shape
        self.action_space_size = model.action_space_size
        self.encoding_shape = model.encoding_shape
        self.encoding_size = int(model.encoding_size)
        self.grid_size = self.action_space_shape[1] * self.action_space_shape[2]
        FactorisedActionDistributionCoordinates.model_output_shape = model.output_shape

        # Split the inputs into the encoding and the action mask
        self.encoding, self.action_mask = tf.split(
            inputs,
            [
                self.encoding_size,
                self.action_space_size,
            ],
            axis=1,
        )

        # Reshape the action mask to be of size (orientations, height, width)
        desired_action_mask_shape = self.action_space_shape
        self.action_mask = tfkl.Reshape((desired_action_mask_shape))(self.action_mask)

        # Get the distribution for the x coordinate
        self.x_distribution = self._x_distribution()
        TFActionDistribution.__init__(self, self.encoding, model)

    @staticmethod
    def required_model_output_shape(
        action_space: gym.Space, model_config: dict
    ) -> Union[int, np.ndarray]:
        """Calculate the required model output shape for the given action space.

        Args:
            action_space (gym.Space): Action space.
            model_config (dict): Model configuration.

        Returns:
            int: Required model output shape.
        """
        output_shape = FactorisedActionDistributionCoordinates.model_output_shape
        return output_shape

    @override(TFActionDistribution)
    def _build_sample_op(self) -> tuple:
        """Sample an action from the distribution.

        Sample using the orientation, x and y models.

        Returns:
            TensorType: Action.
        """
        # Sample x from the distribution of x
        x_distribution = self.x_distribution
        x = x_distribution.sample()

        # Sample y from the distribution of y given x
        y_distribution = self._y_distribution(x)
        y = y_distribution.sample()

        # Sample orientation from the distribution of orientations given x and y
        orientation_distribution = self._orientation_distribution(x, y)
        orientation = orientation_distribution.sample()

        return orientation, x, y

    def deterministic_sample(self) -> tuple:
        """Sample an action from the distribution by taking
        the argmax of the logits.

        Sample using the orientation, x and y models.

        Returns:
            TensorType: Action.
        """
        # Sample x from the distribution of x
        x_distribution = self.x_distribution
        x = x_distribution.deterministic_sample()

        # Sample y from the distribution of y given x
        y_distribution = self._y_distribution(x)
        y = y_distribution.deterministic_sample()

        # Sample orientation from the distribution of orientations given x and y
        orientation_distribution = self._orientation_distribution(x, y)
        orientation = orientation_distribution.deterministic_sample()

        return orientation, x, y

    @override(ActionDistribution)
    def entropy(self):
        """Calculate the entropy of the distribution.

        Returns:
            TensorType: Entropy of the distribution.
        """
        # Calculate the entropy of the x distribution
        # by sampling from it and then calculating the entropy
        x_distribution = self.x_distribution
        x = x_distribution.sample()
        x_entropy = x_distribution.entropy()

        # Calculate the entropy of the y distribution given x
        # by sampling from it and then calculating the entropy
        y_distribution = self._y_distribution(x)
        y = y_distribution.sample()
        y_entropy = y_distribution.entropy()

        # Calculate the entropy of the orientation distribution given x and y
        # by sampling from it and then calculating the entropy
        orientation_distribution = self._orientation_distribution(x, y)
        orientation_entropy = orientation_distribution.entropy()

        return orientation_entropy + x_entropy + y_entropy

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution):
        """Calculate the KL divergence between this distribution and another.

        Args:
            other (ActionDistribution): Other distribution.

        Returns:
            TensorType: KL divergence between this distribution and the other
                distribution.
        """
        # Calculate the KL divergence between the x distributions
        x_distribution = self.x_distribution
        other_x_distribution = other.x_distribution
        x = x_distribution.sample()
        x_kl = x_distribution.kl(other_x_distribution)

        # Calculate the KL divergence between the y distributions given x
        y_distribution = self._y_distribution(x)
        other_y_distribution = other._y_distribution(x)
        y = y_distribution.sample()
        y_kl = y_distribution.kl(other_y_distribution)

        # Calculate the KL divergence between the orientation distributions given x and y
        orientation_distribution = self._orientation_distribution(x, y)
        other_orientation_distribution = other._orientation_distribution(x, y)
        orientation_kl = orientation_distribution.kl(other_orientation_distribution)

        return orientation_kl + x_kl + y_kl

    @override(ActionDistribution)
    def logp(
        self, actions: Union[TensorType, Tuple[TensorType, TensorType]]
    ) -> TensorType:
        """Calculate the log probability of the given actions.

        Args:
            actions (Union[TensorType, Tuple[TensorType, TensorType]]): Actions.

        Returns:
            TensorType: Log probability of the given actions.
        """
        # Calculate the log probability of the x coordinate
        orientation, x, y = self.unpack_actions(actions)
        x_logp = self.x_distribution.logp(x)

        # Calculate the log probability of the y coordinate given the x coordinate
        y_distribution = self._y_distribution(x)
        y_logp = y_distribution.logp(y)

        # Calculate the log probability of the orientation given the x and y coordinates
        orientation_distribution = self._orientation_distribution(x, y)
        orientation_logp = orientation_distribution.logp(orientation)

        return orientation_logp + x_logp + y_logp

    def unpack_actions(
        self, actions: Union[TensorType, Tuple[TensorType, TensorType]]
    ) -> Tuple[TensorType, TensorType]:
        """Unpack the given actions into orientation, x, and y.

        Args:
            actions (Union[TensorType, Tuple[TensorType, TensorType]]): Actions.

        Returns:
            Tuple[TensorType, TensorType]: Orientation, x, and y.
        """
        # Unpack the actions into orientation, x, and y
        if isinstance(actions, tuple):
            orientation, x, y = list(zip(actions))
            orientation, x, y = orientation[0], x[0], y[0]

        # Unpack the actions into orientation, x, and y
        else:
            # Reshape the actions into a 2D tensor
            actions = tf.reshape(actions, (-1, 3))

            # Split the actions into orientation, x, and y
            orientation, x, y = tf.split(actions, num_or_size_splits=3, axis=-1)
            orientation = tf.squeeze(orientation, axis=-1)

            # Reshape the x and y into a 1D tensor
            x = tf.squeeze(x, axis=-1)
            y = tf.squeeze(y, axis=-1)
        return orientation, x, y

    def _x_distribution(self):
        """Create the distribution of coordinates.

        Using the model encoding, calculate the logits of the x coordinates and
        use them to create a categorical distribution.

        Before using the logits, the logits are masked. The mask for the
        x coordinates is calculated from the full action mask by using checking if
        there is at least one valid action for each x coordinate.

        Returns:
            ActionDistribution: Distribution of x coordinates.
        """
        # Calculate the logits of the x coordinates
        x_logits = self.model.x_model(self.encoding)

        # Find the action mask for the x
        action_mask_for_x = tf.reduce_max(self.action_mask, axis=(1, 3))

        # Mask the logits of the x coordinates
        x_logits_masked = x_logits + tf.maximum(
            tf.math.log(action_mask_for_x), tf.float32.min
        )

        return Categorical(
            inputs=x_logits_masked,
            model=self.model,
        )

    def _y_distribution(self, x: TensorType) -> ActionDistribution:
        """Create the distribution of y given the x coordinate.

        The sampled x coordinate is normalized and concatenated with the
        model encoding. Using the concatenated encoding, logits for the y coordinate are
        calculated and used to create a categorical distribution.

        Before using the logits, the logits are masked. The action mask for y given the
        x coordinate is calculated from the full action mask in the following way:

            - Subset the action mask for the given x coordinate
            - Check if there is at least one valid action for each y coordiante

        Args:
            x (TensorType): x coordinate.

        Returns:
            ActionDistribution: Distribution of y given the x coordinate.
        """
        # Calculate the logits of the y given the x
        x_int = tf.cast(x, tf.int32)
        num_x = self.action_space_shape[1]
        x_input = tf.cast(x / num_x, tf.float32)

        # Get the y logits given the x
        y_logits = self.model.y_model([self.encoding, x_input])

        # Get the action mask for the y given the x
        action_mask_for_y = tf.gather(self.action_mask, x_int, axis=2, batch_dims=1)
        action_mask_for_y = tf.reduce_max(action_mask_for_y, axis=1)

        # Mask the logits of the y given the x
        y_logits_masked = y_logits + tf.maximum(
            tf.math.log(action_mask_for_y), tf.float32.min
        )

        return Categorical(
            inputs=y_logits_masked,
            model=self.model,
        )

    def _orientation_distribution(self, x, y):
        """Create the distribution of orienation given the y and x.

        The sampled y and x coordinates are normalized. These are then
        concatenated with the model encoding. Using the concatenated encoding,
        logits for the y coordinate are calculated and used to create a
        categorical distribution.

        Before using the logits, the logits are masked. The action mask for orientation
        given the x and y is calculated from the full action mask by subsetting
        the action mask for the given y and x.

        Args:
            x (TensorType): x coordinate.
            y (TensorType): y coordinate.

        Returns:
            ActionDistribution: Distribution of orientation given the x and y.
        """
        # Create the inputs for the orientation model
        x = tf.cast(x, tf.int32)
        y = tf.cast(y, tf.int32)
        num_x = self.action_space_shape[1]
        num_y = self.action_space_shape[2]
        x_input = tf.cast(x / num_x, tf.float32)
        y_input = tf.cast(y / num_y, tf.float32)

        # Calculate the logits of the orientation given the x and y
        orientation_logits = self.model.orientation_model(
            [self.encoding, x_input, y_input]
        )

        # Find the action mask for the given x and y
        action_mask_for_orientation = tf.gather(
            self.action_mask, x, axis=2, batch_dims=1
        )
        action_mask_for_orientation = tf.gather(
            action_mask_for_orientation, y, axis=2, batch_dims=1
        )

        # Mask the logits of the orientation given the x and y
        orientation_logits_masked = orientation_logits + tf.maximum(
            tf.math.log(action_mask_for_orientation), tf.float32.min
        )

        return Categorical(
            inputs=orientation_logits_masked,
            model=self.model,
        )
