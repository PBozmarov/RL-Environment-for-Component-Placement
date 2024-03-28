from pathlib import Path
import streamlit as st  # type: ignore
import pandas as pd
import plotly.express as px  # type: ignore
import numpy as np
import os
import visualization_grid as vis_grid

# This page is used to compare the performance of different agents.
# User can select the agents from the drop down menu and the plots will be updated accordingly.

st.set_page_config(
    page_title="Comparison analysis", page_icon=":bar_chart:", layout="wide"
)
theme_plotly = None
# streamlit preparation
st.image("web_app/images/instadeep_cropped_logo.png", width=600)
st.title("📉 Comparison analysis 📈 ")
st.divider()


def get_folder_names(path):
    """Get the names of all the folders in a directory"""
    folder_names = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folder_names.append(folder)
    return folder_names


dir_path = str(Path.home()) + "/ray_results/PPO/"
file_names = get_folder_names(dir_path)
# Convert file_names from a list to a numpy array
file_names = np.array(file_names)

# Display the drop down menu of the files
selection_box_placeholder = st.empty()

heading_1 = st.header("Database of previous experiments")

col21, col22 = st.columns(2)
with col21:
    st.markdown("**:blue[1.1. Input parameters ]**")

# Calculates the number of files
num_files = len(file_names)

# Initializes an empty list to store all dataframes
all_dataframes = []

dir_path = str(Path.home()) + "/ray_results/PPO/"

# Initializes an empty list to store the names of the files which do not have the .csv file
no_rollouts = []
# Loops through all files and read in the dataframes
for file in file_names:
    logdir_vis = dir_path + file
    configuration, actions, components = vis_grid.load_pickle(logdir_vis + "/")

    if configuration is not None:
        excel_file_name = configuration["env_config"]["type"]

        try:
            dir_path_excel_name = dir_path + file + "/" + excel_file_name + ".csv"

        except FileNotFoundError:
            print(
                f"The .csv file with the input parameters are not present in the following filepath: {logdir_vis}. Please rerun the experiment."
            )
            st.write(
                f"The .csv file with the input parameters are not present in the following filepath: {logdir_vis}. Please rerun the experiment."
            )
            dir_path_excel_name = None

        if dir_path_excel_name is not None:

            try:
                input_params_df = pd.read_csv(dir_path_excel_name)

            except FileNotFoundError:
                no_rollouts.append(file)
                input_params_df = None

            if input_params_df is not None:
                input_params_df["name"] = file
                input_params_df["Time"] = input_params_df["name"].str[-19:]

                input_params_df = input_params_df.set_index("Time")

                all_dataframes.append(input_params_df)
    else:
        no_rollouts.append(file)

concatenated_df = pd.concat(all_dataframes, axis=0)
concatenated_df_transposed = concatenated_df.T.copy().astype("str")
st.write(concatenated_df_transposed)

if len(no_rollouts) > 0:
    st.write("No rollouts.pkl file for the following files: ", no_rollouts)

options = concatenated_df_transposed.columns
# Creates a multiselect box with all the options
model_options = st.multiselect(
    "**Select your desired models to compare:**", options=options, key="macro_options"
)

# Creates a dataframe for the episode_reward_mean
loggings_df_reward = pd.DataFrame(
    columns=["model", "training_iteration", "episode_reward_mean"]
)

# Creates a dataframe for the custom_metrics/normalized_wirelengths_mean
loggings_df_wire_length = pd.DataFrame(
    columns=[
        "model",
        "training_iteration",
        "custom_metrics/normalized_wirelengths_mean",
    ]
)

# Creates a dataframe for the custom_metrics/num_intersections_mean
loggings_df_intersections = pd.DataFrame(
    columns=["model", "training_iteration", "custom_metrics/num_intersections_mean"]
)

for model in model_options:
    filename = concatenated_df_transposed[model]["name"]
    dir_path_csv = dir_path + filename + "/progress.csv"
    progress_df = pd.read_csv(dir_path_csv)
    progress_df["model"] = model
    loggings_df_reward = pd.concat(
        [
            loggings_df_reward,
            progress_df[["model", "training_iteration", "episode_reward_mean"]],
        ]
    )
    loggings_df_wire_length = pd.concat(
        [
            loggings_df_wire_length,
            progress_df[
                [
                    "model",
                    "training_iteration",
                    "custom_metrics/normalized_wirelengths_mean",
                ]
            ],
        ]
    )
    rolling_df = (
        progress_df[["training_iteration", "custom_metrics/num_intersections_mean"]]
        .rolling(window=20)
        .mean()
    )
    rolling_df = rolling_df.join(progress_df["model"])

    loggings_df_intersections = pd.concat(
        [loggings_df_intersections, rolling_df]
    ).reset_index(drop=True)

# The below px.line plots, plot the reward, wire length and number of intersections curves for the selected models
fig_reward = px.line(
    loggings_df_reward, x="training_iteration", y="episode_reward_mean", color="model"
)
fig_wire = px.line(
    loggings_df_wire_length,
    x="training_iteration",
    y="custom_metrics/normalized_wirelengths_mean",
    color="model",
)
fig_intersect = px.line(
    loggings_df_intersections,
    x="training_iteration",
    y="custom_metrics/num_intersections_mean",
    color="model",
)

st.markdown("**:blue[Reward curve]**")
chart_reward = st.plotly_chart(fig_reward, use_container_width=True, theme=theme_plotly)

st.markdown("**:blue[Wire length curve]**")
chart_wire = st.plotly_chart(fig_wire, use_container_width=True, theme=theme_plotly)

st.markdown("**:blue[Number of intersections curve]**")
chart_intersect = st.plotly_chart(
    fig_intersect, use_container_width=True, theme=theme_plotly
)
