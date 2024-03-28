from environment.dummy_env_rectangular_pin import Component
from matplotlib.figure import Figure  # type: ignore
from matplotlib.colors import to_rgba
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from typing import List, Tuple

import numpy as np
import random
import pickle


def load_pickle(file_path: str):
    """Loads the pickle files from the directory specified by file_path.

    Args:
        file_path (str): The path to the pickle files.

    Returns:
        config (PPOConfig): The config dictionary.
        actions (List[Tuple[int]]): The actions.
        components (List[Components]): The components.
    """

    try:
        with open(file_path + "params.pkl", "rb") as f:
            config = pickle.load(f)
    except FileNotFoundError:
        print(
            f"ERROR: The file params.pkl in filepath='{file_path}' could not be found. Please check the file path and try again."
        )
        config = None
    except Exception:
        print(
            f"ERROR: There was an error opening the file params.pkl in file path = '{file_path}'. Please check the file and try again."
        )
        config = None

    try:
        with open(file_path + "actions.pkl", "rb") as f:
            actions = pickle.load(f)
    except FileNotFoundError:
        print(
            f"ERROR: The file actions.pkl in filepath='{file_path}' could not be found. Please check the file path and try again."
        )
        actions = None

    except Exception:
        print(
            f"ERROR: There was an error opening the file actions.pkl in file path = '{file_path}'. Please check the file and try again."
        )
        actions = None

    try:
        with open(file_path + "components.pkl", "rb") as f:
            components = pickle.load(f)
    except FileNotFoundError:
        print(
            f"ERROR: The file components.pkl in filepath='{file_path}' could not be found. Please check the file path and try again."
        )
        components = None

    except Exception:
        print(
            f"ERROR: There was an error opening the file components.pkl in file path = '{file_path}'. Please check the file and try again."
        )
        components = None

    return config, actions, components


def render(
    height: int, width: int, components: List[Component], actions: List[Tuple[int]]
) -> Figure:
    """Plots the given components on a grid of height x width, given the actions.

    Args:
        height (int): The height of the grid.
        width (int): The width of the grid.
        components (List[Component]): The components to place on the grid.
        actions (List[Tuple[int]]): The actions, one action corresponding to each component.

    Returns:
        fig (Figure): The figure.
    """

    # Load the colors for the components
    all_colors = list(matplotlib.colors.CSS4_COLORS.keys())
    # Colors for the pins
    pin_colors = list(matplotlib.colors.CSS4_COLORS.keys())

    random.seed(69)
    random.shuffle(pin_colors)

    # Colors for the net
    net_colors = []

    colors = [
        c
        for c in all_colors
        if "dark" not in c.lower()
        and "light" not in c.lower()
        and "white" not in c.lower()
        and "black" not in c.lower()
    ]
    colors.reverse()  # the reverse colors look nicer in the plot

    # create the figure
    fig, ax = plt.subplots()

    # plot the grid
    for i in range(width):
        for j in range(height):
            ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, zorder=0))

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")

    # plot the components
    for i, (component, action) in enumerate(zip(components, actions)):
        # get the component
        orientation = action[0]
        if orientation in [0, 2]:
            placement_height = component.h
            placement_width = component.w
        elif orientation in [1, 3]:
            placement_height = component.w
            placement_width = component.h

        x, y = placement_height, placement_width
        rgba_color = to_rgba(colors[i % len(colors)], alpha=0.85)
        # add the component
        rect = plt.Rectangle(
            (action[2], height - action[1] - x),
            y,
            x,
            color=rgba_color,
            lw=2,
            zorder=1,
        )
        ax.add_patch(rect)

        # add index label
        ax.text(
            action[2] + y / 2,
            height - action[1] - x / 2,
            f"{i}",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=12,
            color="black",
            zorder=2,
        )
        pin_colors = pin_colors[:5]
        # plot the pins
        for pin in component.pins:

            pin_color = pin_colors[pin.net_id]
            net_colors.append((pin_color, pin.net_id))

            pin_marker = plt.Circle(
                (pin.absolute_y + 0.5, height - pin.absolute_x - 0.5),
                radius=0.15,
                color=pin_color,
                zorder=4,
            )
            # Draw the black border circle
            pin_border = plt.Circle(
                (pin.absolute_y + 0.5, height - pin.absolute_x - 0.5),
                radius=0.17,  # Slightly larger radius for the border
                color="black",
                zorder=3,  # Lower zorder to place it behind the colored circle
            )
            ax.add_patch(pin_marker)
            ax.add_patch(pin_border)

    # remove the axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # sort the net_colors to make them unique
    net_colors = list(set(net_colors))
    net_colors = sorted(net_colors, key=lambda x: x[1])
    net_colors = [x for (x, y) in net_colors]

    # Create a map
    cmap = matplotlib.colors.ListedColormap(net_colors)

    # Plot the colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(net_colors) - 1)
    cb = plt.colorbar(
        matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        orientation="vertical",
    )

    # Set the colorbar ticks and labels
    cb.set_ticks(np.arange(len(net_colors)))
    cb.set_ticklabels(np.arange(len(net_colors)))

    return fig
