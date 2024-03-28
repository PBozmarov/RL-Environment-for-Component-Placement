from streamlit_tensorboard import st_tensorboard
from pathlib import Path
import streamlit as st  # type: ignore
import pandas as pd
import plotly.express as px  # type: ignore
import visualization_grid as vis_grid
import numpy as np
import os
import time

# This page is used to visualize the sample rollouts and reward plots of the trained agents.
# User can select the agent from the drop down menu and the plots will be updated accordingly.

theme_plotly = None  # None or streamlit
st.set_page_config(page_title="Trained Agents", page_icon=":bar_chart:", layout="wide")
st.image("web_app/images/instadeep_cropped_logo.png", width=600)
st.title("🏋🏼 Trained Agents 🏋🏼‍♂️")

with open("web_app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_folder_names(path):
    """Get the names of all the folders in a directory."""
    folder_names = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folder_names.append(folder)

    return folder_names


# Gets the names of all the folders in the directory
dir_path = str(Path.home()) + "/ray_results/PPO/"
file_names = get_folder_names(dir_path)
file_names = np.array(file_names)

# Creates a dictionary with the time stamp as the key and the file name as the value
time_stamp_dict = {}
for file in file_names:
    time_stamp = file[-19:]
    time_stamp_dict[time_stamp] = file

selection_box_placeholder = st.empty()

st.divider()
heading_1 = st.header("1. Explore the results of previous experiments")

# Creates a dropdown menu to select the experiment which the user wants to run.
option_timestamp = st.selectbox(
    "List of previous experiments", time_stamp_dict.keys(), key="dropdown_1"
)
option = time_stamp_dict[option_timestamp]

st.write("You selected:", option)
col21, col22 = st.columns(2)

# Displays the input parameters of the chosen experiment
with col21:
    st.markdown("**:blue[1.1. Input parameters ]**")

if "dropdown_1" not in st.session_state:
    pass
else:
    # Loads the pickle file from the chosen experiment
    logdir_vis = dir_path + option
    configuration, actions, components = vis_grid.load_pickle(logdir_vis + "/")

    try:
        excel_file_name = configuration["env_config"]["type"]
        dir_path_excel_name = dir_path + option + "/" + excel_file_name + ".csv"

    except TypeError:
        print(
            f"ERROR: The file params.pkl in filepath='{logdir_vis}' could not be found. Please check the file path and try again."
        )
        st.write(
            f"ERROR: The file params.pkl in filepath='{logdir_vis}' could not be found. Please check the file path and try again."
        )
        excel_file_name = ""
        dir_path_excel_name = None

    if dir_path_excel_name is not None:
        try:
            input_params_df = pd.read_csv(dir_path_excel_name)

        except FileNotFoundError:
            print(
                f"ERROR: The file could not be opened. Please check the filepath  '{logdir_vis}' to ensure that the '{excel_file_name+'.csv'}' file is present. If not, please rerun the experiment."
            )
            st.write(
                f"ERROR: The file could not be opened. Please check the filepath  '{logdir_vis}' to ensure that the '{excel_file_name+'.csv'}' file is present. If not, please rerun the experiment."
            )
            input_params_df = None

        except Exception:
            print(
                f"ERROR: The file coould not be opened. Please check the filepath  '{dir_path_excel_name}' to ensure that the '{excel_file_name+'.csv'}' file is present. If not, please rerun the experiment."
            )
            st.write(
                f"ERROR: The file could not be opened. Please check the filepath  '{dir_path_excel_name}' to ensure that the '{excel_file_name+'.csv'}' file is present. If not, please rerun the experiment."
            )
            input_params_df = None

        if input_params_df is not None:
            with col21:
                # Transposes the dataframe
                input_params_transposed_df = input_params_df.T.copy()

                # Renames the columns
                input_params_transposed_df = input_params_transposed_df.rename(
                    columns={0: "Input parameters"}
                )
                input_params_transposed_df = input_params_transposed_df.astype("str")

                # Displays the input params_df in the streamlit app as a table
                st.write(input_params_transposed_df)

        # Displays the training stats in the streamlit app as a table
        dir_path_training_stats = dir_path + option + "/progress.csv"
        try:
            training_stats_df = pd.read_csv(dir_path_training_stats)
        except FileNotFoundError:
            print(
                f"ERROR: The file could not be opened. Please check the filepath  '{dir_path_training_stats}' to ensure that the progress.csv file is present. If not, please rerun the experiment."
            )
            st.write(
                f"ERROR: The file could not be opened. Please check the filepath  '{dir_path_training_stats}' to ensure that the progress.csv file is present. If not, please rerun the experiment."
            )
            training_stats_df = None

        if training_stats_df is not None:

            training_stats_df = training_stats_df[
                [
                    "training_iteration",
                    "episode_reward_mean",
                    "episodes_this_iter",
                    "time_total_s",
                ]
            ]
            training_stats_df = training_stats_df.rename(
                columns={
                    "training_iteration": "Iteration",
                    "episode_reward_mean": "Mean Reward",
                    "episodes_this_iter": "Episodes",
                    "time_total_s": "Cumulative time",
                }
            )

            training_stats_df["Cumulative time"] = round(
                training_stats_df["Cumulative time"] / 60, 2
            )

            with col22:
                st.markdown("**:blue[1.2. Training summary ]**")
                st.write(training_stats_df)

            st.markdown("**:blue[1.3. Reward curve ]**")
            # Plots the reward curve vs the number of iterations
            fig = px.line(
                training_stats_df, x="Iteration", y="Mean Reward", title="Reward curve"
            )
            st.plotly_chart(fig)

    st.markdown("**:blue[1.4. Grid Visualization (Policy rollout) ]**")

    # When the button is clicked, the policy rollout will be displayed
    if st.button("Display Policy rollout"):

        height = configuration["env_config"]["height"]
        width = configuration["env_config"]["width"]

        sample_num = int(0)
        plot_placeholder = st.empty()

        try:
            for i in range(1, len(actions) + 1):
                fig = vis_grid.render(
                    height, width, components[sample_num][:i], actions[sample_num][:i]
                )
                plot_placeholder.pyplot(fig)
                time.sleep(2)
        except Exception:
            print(
                f"Error in displaying the policy rollout. Please check the filepath '{logdir_vis}' to ensure the actions.pkl, params.pkl and components.pkl are present."
            )
            st.write(
                f"Error in displaying the policy rollout. Please check the filepath '{logdir_vis}' to ensure the actions.pkl, params.pkl and components.pkl are present. If they are not, please rerun the experiment."
            )

st.divider()
heading_2 = st.header("2. Tensorboard")

# When the button is clicked, the tensorboard will be displayed
if st.button("Display Tensorboard"):
    logdir = dir_path + option
    st_tensorboard(logdir=logdir, port=8530, width=1000)
