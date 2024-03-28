from utils.agent.utils import get_config, generate_rollouts
from utils.visualization.csv_utils import save_config_to_csv

from streamlit_tensorboard import st_tensorboard
from pathlib import Path
from ray import tune  # type: ignore
from ray.tune import Callback  # type: ignore
import ray  # type: ignore

import pandas as pd
import streamlit as st  # type: ignore
import plotly.express as px  # type: ignore
import visualization_grid as vis_grid
import os
import glob
import time
from ray.rllib.utils.framework import try_import_tf  # type: ignore

_, tf, _ = try_import_tf()
tfkl = tf.keras.layers

# This page is used to train agents, and visualize the training process.
# The training process is visualized using Tensorboard, which is embedded in the web app.
# The user can select the hyperparameters of the training process, and the training will start.

# Global Variables
theme_plotly = None  # None or streamlit

# Config
st.set_page_config(page_title="Train Agents", page_icon=":bar_chart:", layout="wide")

# Streamlit preparation
st.image("web_app/images/instadeep_cropped_logo.png", width=600)

# Title
st.title("🤖 Train Agents 🏋🏼🦾")

# Style
with open("web_app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sessions tate initialise
# Check if 'key' already exists in session_state
# If not, then initialize it
if "x1" not in st.session_state:
    st.session_state["x1"] = 10  # Height of the environment

if "x2" not in st.session_state:
    st.session_state["x2"] = 10  # Width of the environment

if "x3" not in st.session_state:
    st.session_state["x3"] = 5  # Min number of components

if "x4" not in st.session_state:
    st.session_state["x4"] = 5  # Max number of components

if "x5" not in st.session_state:
    st.session_state["x5"] = 2  # Min component height

if "x6" not in st.session_state:
    st.session_state["x6"] = 2  # Max component height

if "x7" not in st.session_state:
    st.session_state["x7"] = 2  # Min component width

if "x8" not in st.session_state:
    st.session_state["x8"] = 2  # Max component width

if "x9" not in st.session_state:
    st.session_state["x9"] = 2 * 2  # Maximum number of pins per component

if "x10" not in st.session_state:
    st.session_state["x10"] = 5  # Component feature vector width

if "x11" not in st.session_state:
    st.session_state["x11"] = 4 + 3 + 1  # Pin feature vector width

if "x12" not in st.session_state:
    st.session_state["x12"] = 9  # Net distribution

if "x13" not in st.session_state:
    st.session_state["x13"] = 9  # Pin spread

if "x14" not in st.session_state:
    st.session_state["x14"] = 3  # Minimum number of nets

if "x15" not in st.session_state:
    st.session_state["x15"] = 3  # Maximum number of nets

if "x16" not in st.session_state:
    st.session_state["x16"] = 6  # Minimum number of pins per nets

if "x17" not in st.session_state:
    st.session_state["x17"] = 6  # Maximum number of pins per nets

if "x18" not in st.session_state:
    st.session_state["x18"] = 2  # Reward beam width

if "x19" not in st.session_state:
    st.session_state["x19"] = 1.0  # Wirelength weight

if "reward_type" not in st.session_state:
    st.session_state.reward_type = "centroid"  # Reward type

if "x20" not in st.session_state:
    st.session_state["x20"] = int(20)  # Number of epochs

if "x21" not in st.session_state:
    st.session_state["x21"] = int(2)  # Number of conv blocks

if "x22" not in st.session_state:
    st.session_state["x22"] = int(3)  # Number of conv filters

if "x23" not in st.session_state:
    st.session_state["x23"] = int(3)  # Conv kernel size

if "x24" not in st.session_state:
    st.session_state["x24"] = int(2)  # Max pool kernel size

if "x25" not in st.session_state:
    st.session_state["x25"] = int(16)  # Component feature encoding dimension

if "x26" not in st.session_state:
    st.session_state["x26"] = int(16)  # Pin feature encoding dimension

if "x27" not in st.session_state:
    st.session_state["x27"] = True  # Max pool

if "x28" not in st.session_state:
    st.session_state["x28"] = "relu"  # Activation

if "x50" not in st.session_state:
    st.session_state["x50"] = 0.0  # Learning rate

# The sidebar is used to set the environment parameters and hyperparameters of the training process
with st.sidebar.form(key="my_form"):
    st.subheader("Set Environment Parameters")
    name = st.selectbox("List of experiments", ["rectangle_pin"], key="dropdown_1")

    # Height and width of the environment
    col1, col2 = st.columns(2)

    # Add a slider to each column
    with col1:
        height = st.slider(
            "Environment height (m)", min_value=3, max_value=100, key="x1"
        )

    with col2:
        width = st.slider("Environment width (m)", min_value=3, max_value=100, key="x2")

    # Min and Max number of components in environment
    col3, col4 = st.columns(2)

    with col3:
        min_num_components = st.slider(
            "Minimum number of components", min_value=2, max_value=40, key="x3"
        )

    with col4:
        max_num_components = st.slider(
            "Maximum number of components", min_value=2, max_value=40, key="x4"
        )

    # Min and Max component height
    col5, col6 = st.columns(2)

    with col5:
        min_component_h = st.slider(
            "Minimum Component height (m)", min_value=1, max_value=10, key="x5"
        )

    with col6:
        max_component_h = st.slider(
            "Maximum Component height (m)", min_value=2, max_value=10, key="x6"
        )

    # Min and Max component width
    col7, col8 = st.columns(2)

    with col7:
        min_component_w = st.slider(
            "Minimum Component width (m)", min_value=1, max_value=10, key="x7"
        )

    with col8:
        max_component_w = st.slider(
            "Maximum Component width (m)", min_value=1, max_value=10, key="x8"
        )

    max_num_pins_per_component = st.slider(
        "Maximum number of pins per component", min_value=1, max_value=20, key="x9"
    )

    col9, col10 = st.columns(2)

    with col9:
        component_feature_vector_width = st.slider(
            "Component feature vector width (m)", min_value=1, max_value=20, key="x10"
        )

    with col10:
        pin_feature_vector_width = st.slider(
            "Pin feature vector width (m)", min_value=1, max_value=20, key="x11"
        )

    # Net distribution and Pin spread
    col11, col12 = st.columns(2)

    with col11:
        net_distribution = st.slider(
            "Net distribution", min_value=1, max_value=20, key="x12"
        )

    with col12:
        pin_spread = st.slider("Pin spread", min_value=1, max_value=20, key="x13")

    # Min and Max number of nets
    col13, col14 = st.columns(2)

    with col13:
        min_num_nets = st.slider(
            "Minimum number of nets", min_value=1, max_value=20, key="x14"
        )

    with col14:
        max_num_nets = st.slider(
            "Maximum number of nets", min_value=1, max_value=20, key="x15"
        )

    # Min and Max number of pins per nets
    col15, col16 = st.columns(2)

    with col15:
        min_num_pins_per_net = st.slider(
            "Minimum number of pins per nets", min_value=1, max_value=20, key="x16"
        )

    with col16:
        max_num_pins_per_net = st.slider(
            "Maximum number of pins per nets", min_value=1, max_value=20, key="x17"
        )

    # Reward beam width and weight of wirelength
    col17, col18 = st.columns(2)

    with col17:
        reward_beam_width = st.slider(
            "Reward beam width", min_value=1, max_value=10, key="x18"
        )

    with col18:
        weight_wirelength = st.slider(
            "Wirelength weight",
            min_value=float(0.0),
            max_value=float(1.0),
            step=float(0.1),
            key="x19",
        )
    weight_num_intersections = st.slider(
        "Number of intersections weight",
        min_value=float(0.0),
        max_value=float(1.0),
        step=float(0.1),
        key="x50",
    )

    st.subheader("Set Model Parameters")

    # Pin feature encoding dimension
    col20, col21 = st.columns(2)

    with col20:
        num_conv_blocks = st.slider(
            "Number of convolutional blocks", min_value=1, max_value=10, key="x21"
        )

    with col21:
        num_conv_filters = st.slider(
            "Number of convolutional filters", min_value=1, max_value=10, key="x22"
        )

    # Convolutional kernel size and max pooling kernel size
    col22, col23 = st.columns(2)

    with col22:
        conv_kernel_size = st.slider(
            "Convolutional kernel size", min_value=1, max_value=10, key="x23"
        )

    with col23:
        max_pool_kernel_size = st.slider(
            "Max pooling kernel size", min_value=1, max_value=10, key="x24"
        )

    # Component and pin feature encoding dimension
    col24, col25 = st.columns(2)

    with col24:
        component_feature_encoding_dimension = st.slider(
            "Component feature encoding", min_value=1, max_value=10, key="x25"
        )

    with col25:
        pin_feature_encoding_dimension = st.slider(
            "Pin feature encoding", min_value=1, max_value=10, key="x26"
        )

    # Max pool and activation
    col26, col27 = st.columns(2)

    with col26:
        max_pool = st.radio("Max pool", ("True", "False"))

    st.session_state.max_pool = max_pool

    with col27:
        activation = st.radio("Activation", ("relu", "sigmoid"))

    st.session_state.activation = activation
    reward_type = st.radio("Choose the reward type", ("centroid", "beam", "both"))

    st.session_state.reward_type = reward_type

    # Training iterations
    training_iterations = st.slider(
        "Training iterations", min_value=1, max_value=200, key="x20"
    )

    submit_button = st.form_submit_button(label="Run model!")
    st.write(submit_button)

# Draws a divider to separate the form and the rest of the app
st.divider()
col19, col20 = st.columns(2)
with col19:
    Traning_header = st.header("2. Train a new agent")

    max_table_height = 3

    table = st.markdown(
        f'<div style="height: {max_table_height}px; overflow-y: auto;">',
        unsafe_allow_html=True,
    )

    Training_iteration = []
    Mean_reward = []
    episodes_this_iter = []
    episodes_time = []

    # Creates a dataframe to store the training statistics
    loggings_df = pd.DataFrame(
        {
            "Iteration": Training_iteration,
            "Mean Reward": Mean_reward,
            "Episodes": episodes_this_iter,
            "Cumulative time": episodes_time,
        }
    )

    st.markdown("**:blue[2.1. Training summary statistics]**")
    table = st.dataframe(loggings_df)

st.markdown("**:blue[2.2. Reward curve]**")
chart = st.plotly_chart(px.line(loggings_df, x="Iteration", y="Mean Reward"))


class CallbackLogs(Callback):
    def on_trial_result(self, **info):
        """
        This function is called at the end of each training iteration.
        It is used to update the dataframe with the training statistics.
        """
        result = info["result"]

        with col19:
            loggings_df.loc[len(loggings_df)] = [
                result["training_iteration"],
                round(result["episode_reward_mean"], 2),
                result["episodes_this_iter"],
                round(result["time_total_s"] / 60, 2),
            ]

        with col20:
            progress_percentage.text(
                f'Percentage completed: {int((result["training_iteration"]/training_iterations)*100)}%'
            )
            prg.progress(
                int((result["training_iteration"] / training_iterations) * 100)
            )
        table.dataframe(loggings_df.iloc[:, :])

        # Plots only the 'Mean Reward'column and the 'training_iteration' column
        fig = px.line(loggings_df, x="Iteration", y="Mean Reward")
        chart.plotly_chart(fig)


logdir = ""
Grid = st.header("Grid")

if submit_button:
    with col20:
        prg = st.progress(0)
        progress_percentage = st.empty()
        progress_percentage.text("Percentage completed: 0%")

        with st.spinner(
            text="Agent is training...",
        ):
            spinner_style = """
                <style>
                .st-spinner div {{
                    width: 300px;
                    height: 300px;
                }}
                </style>
            """
            st.write(spinner_style, unsafe_allow_html=True)

            config = get_config(name)

            # Stores the model parameters in the session state
            data_points = {
                "Variable": [
                    "env height",
                    "env width",
                    "min # components",
                    "max # components",
                    "min component height",
                    "max component height",
                    "min component width",
                    "max component width",
                    "max pins/component",
                    "Component vector width",
                    "Pin vector width",
                    "Net distribution",
                    "Pin spread",
                    "Min # nets",
                    "Max # nets",
                    "Min # pins/net",
                    "Max # pins/nets",
                    "Reward beam width",
                    "Wirelength weight",
                    "Reward type",
                    "Training iterations",
                ],
                "Value": [
                    height,
                    width,
                    min_num_components,
                    max_num_components,
                    min_component_h,
                    max_component_h,
                    min_component_w,
                    max_component_w,
                    max_num_pins_per_component,
                    component_feature_vector_width,
                    pin_feature_vector_width,
                    net_distribution,
                    pin_spread,
                    min_num_nets,
                    max_num_nets,
                    min_num_pins_per_net,
                    max_num_pins_per_net,
                    reward_beam_width,
                    weight_wirelength,
                    reward_type,
                    training_iterations,
                ],
            }

            # Creates a dataframe to store the model parameters
            df_training = pd.DataFrame(data_points)
            st.dataframe(df_training)

            config["model"]["custom_model_config"]["height"] = height
            config["model"]["custom_model_config"]["width"] = width
            config["model"]["custom_model_config"][
                "max_num_components"
            ] = max_num_components
            config["model"]["custom_model_config"][
                "min_num_components"
            ] = min_num_components
            config["model"]["custom_model_config"]["max_component_w"] = max_component_w
            config["model"]["custom_model_config"]["max_component_h"] = max_component_h
            config["model"]["custom_model_config"][
                "max_num_pins_per_component"
            ] = max_num_pins_per_component
            config["model"]["custom_model_config"][
                "component_feature_vector_width"
            ] = component_feature_vector_width
            config["model"]["custom_model_config"][
                "pin_feature_vector_width"
            ] = pin_feature_vector_width
            config["model"]["custom_model_config"]["num_conv_blocks"] = num_conv_blocks
            config["model"]["custom_model_config"][
                "num_conv_filters"
            ] = num_conv_filters
            config["model"]["custom_model_config"][
                "conv_kernel_size"
            ] = conv_kernel_size
            if activation == "relu":
                config["model"]["custom_model_config"]["activation"] = tf.nn.relu
            elif activation == "tanh":
                config["model"]["custom_model_config"]["activation"] = tf.nn.tanh
            elif activation == "sigmoid":
                config["model"]["custom_model_config"]["activation"] = tf.nn.sigmoid
            config["model"]["custom_model_config"]["max_pool"] = max_pool
            config["model"]["custom_model_config"][
                "max_pool_kernel_size"
            ] = max_pool_kernel_size
            config["model"]["custom_model_config"][
                "component_feature_encoding_dimension"
            ] = component_feature_encoding_dimension
            config["model"]["custom_model_config"][
                "pin_feature_encoding_dimension"
            ] = pin_feature_encoding_dimension

            config["env_config"]["height"] = height
            config["env_config"]["width"] = width
            config["env_config"]["net_distribution"] = net_distribution
            config["env_config"]["pin_spread"] = pin_spread
            config["env_config"]["min_component_w"] = min_component_w
            config["env_config"]["max_component_w"] = max_component_w
            config["env_config"]["min_component_h"] = min_component_h
            config["env_config"]["max_component_h"] = max_component_h
            config["env_config"]["max_num_components"] = max_num_components
            config["env_config"]["min_num_components"] = min_num_components
            config["env_config"]["min_num_nets"] = min_num_nets
            config["env_config"]["max_num_nets"] = max_num_nets
            config["env_config"]["max_num_pins_per_net"] = max_num_pins_per_net
            config["env_config"]["min_num_pins_per_net"] = min_num_pins_per_net
            config["env_config"]["reward_type"] = reward_type
            config["env_config"]["reward_beam_width"] = reward_beam_width
            config["env_config"]["weight_wirelength"] = weight_wirelength
            config["env_config"]["weight_num_intersections"] = weight_num_intersections

            ray.init(local_mode=True)
            run_model = tune.run(
                "PPO",
                config=config,
                stop={"training_iteration": training_iterations},
                checkpoint_freq=1,
                checkpoint_at_end=True,
                keep_checkpoints_num=5,
                restore=None,
                callbacks=[CallbackLogs()],
            )

            # The generate rollouts function is called to generate rollouts from the
            # trained model and displays a visual representation of the policy
            generate_rollouts(config, name)

            dir_path = str(Path.home()) + "/ray_results/PPO/"
            all_files = [
                f
                for f in glob.glob(os.path.join(dir_path, "*"))
                if not f.endswith(".json")
            ]
            all_files.sort(key=os.path.getmtime, reverse=True)
            most_recent_file = all_files[0]
            logdir = most_recent_file

            df = save_config_to_csv(config)
            df.to_csv(f"{logdir}/my_model.csv", index=False)

            st.success("Training complete!")

    st.balloons()

    configuration_train, actions_train, components_train = vis_grid.load_pickle(
        logdir + "/"
    )

    height_train = height
    width_train = width

    policy_num_train = 0

    sample_num = int(policy_num_train)
    plot_placeholder = st.empty()

    # The for loop below displays the policy for the last iteration of training
    for i in range(1, len(actions_train) + 1):
        fig = vis_grid.render(
            height_train,
            width_train,
            components_train[sample_num][:i],
            actions_train[sample_num][:i],
        )
        plot_placeholder.pyplot(fig)
        time.sleep(2)

    # The code below displays the tensorboard for the training run
    Tensorboard_display = st.header("Tensorboard")
    st_tensorboard(logdir=logdir, port=8530, width=1000)
