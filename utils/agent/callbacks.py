"""
Custom callbacks for wirelength and number of intersections for the rectangular environment with pins,
"""

from ray.rllib.algorithms.callbacks import DefaultCallbacks  # type: ignore


class CustomCallbackClass(DefaultCallbacks):
    """Custom callback that logs wirelength and the
    number of intersections at the end of each episode."""

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
                In case of environment failures, episode may also be an Exception
                that gets thrown from the environment before the episode finishes.
                Users of this callback may then handle these error cases properly
                with their custom logics.
            env_index: The index of the sub-environment that ended the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forw
        """
        # Get last info dict from the episode
        info = episode.last_info_for("agent0")
        wirelength = info["wirelength"]
        num_ints = info["num_intersections"]

        # Store the wirelength and num_ints in the episode's custom metrics
        episode.custom_metrics["normalized_wirelengths"] = wirelength
        episode.custom_metrics["num_intersections"] = num_ints
