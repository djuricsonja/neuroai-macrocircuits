"""Learning-curve plots and network-architecture diagrams."""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

NCAP_PAPER_FIGURE_URL = (
    'https://github.com/neuromatch/NeuroAI_Course/blob/main/projects/'
    'project-notebooks/static/NCAPPaper.png?raw=true'
)


def show_paper_figure():
    """Displays the NCAP paper illustration."""
    display(Image(url=NCAP_PAPER_FIGURE_URL))


# Running totals each x-axis can be read from, in priority order, with its axis label.
_X_AXES = {
    'steps': (('train/steps', 'total_env_steps'), 'Environment Steps'),
    'episodes': (('train/episodes',), 'Episodes'),
}


def _x_axis(df, x, path):
    """Running total to plot a run against.

    Tonic logs both counters directly (`train/steps`, `train/episodes`); ES logs steps as
    `total_env_steps` (see `es._save_es_run`) and has no episode counter at all, so an
    episode axis is refused rather than faked. Only when no step counter is present do we
    fall back to accumulating `test/episode_length/mean`, which is a true per-row step
    delta for ES but NOT for tonic -- there it is the fixed time-limit length (1000), so
    the fallback would silently rescale a 500k-step run onto a 25k-step axis.
    """
    for column in _X_AXES[x][0]:
        if column in df.columns:
            return df[column]
    if x == 'episodes':
        raise ValueError(f"{path} logs no episode counter; plot it with x='steps'")
    return np.cumsum(df['test/episode_length/mean'])


def plot_performance(paths, ax=None, title='Model Performance', x='steps', band=True):
    """
    Plots the performance of multiple models on the same axes using Seaborn for styling.

    Reads CSV log files from specified paths and plots the mean test episode score
    against training progress, for each model. Each line's legend is set to the name of
    the last folder in the path, representing the model's name.

    Test scores are averaged over only a handful of episodes per evaluation, so a bare
    mean line is mostly sampling noise; `band` shades mean +/- std around it (where the
    log records one) to keep that spread visible rather than reading it as learning.

    Parameters:
    - paths (list of str): Paths to the experiment directories.
    - ax (matplotlib.axes.Axes, optional): A matplotlib axis object to plot on. If None,
      a new figure and axis are created.
    - x (str): Training progress measure for the x-axis -- 'steps' (environment steps) or
      'episodes' (episodes trained on). ES runs log no episode counter and so support
      only 'steps'.
    - band (bool): Shade mean +/- std of the test episode score. Ignored for logs
      without a `test/episode_score/std` column (ES runs log a running best, not a mean).
    """
    if x not in _X_AXES:
        raise ValueError(f'x must be one of {sorted(_X_AXES)}, got {x!r}')

    # Set the Seaborn style
    sns.set(style="whitegrid")
    colors = sns.color_palette("colorblind")  # Colorblind-friendly palette

    if ax is None:
        fig, ax = plt.subplots()

    for index, path in enumerate(paths):
        # Extract the model name from the path
        model_name = os.path.basename(path.rstrip('/'))
        color = colors[index % len(colors)]

        # Load data
        df = pd.read_csv(os.path.join(path, 'log.csv'))
        scores = df['test/episode_score/mean']
        progress = _x_axis(df, x, path)
        ax.plot(progress, scores, label=model_name, color=color)
        if band and 'test/episode_score/std' in df.columns:
            spread = df['test/episode_score/std']
            ax.fill_between(progress, scores - spread, scores + spread, color=color, alpha=0.2)

    ax.set_xlabel(_X_AXES[x][1])
    # Each point totals the rewards of one episode ('score' in tonic's terms, see
    # utils/trainer.py); the averaging is across the handful of test episodes per
    # evaluation, NOT within an episode. 'Mean Episode Score' read as reward-per-step.
    ax.set_ylabel('Total Episode Reward (mean of test episodes)')
    ax.legend()
    ax.set_title(title)


def runs_by_reward(paths, key='best', min_reward=None, top=None):
    """Experiment run directories ranked (and optionally filtered) by achieved reward.

    Reads each path's log.csv -- the same file plot_performance() plots -- and scores
    it from the test/episode_score/mean column:
    - 'best': the highest value logged over the whole run.
    - 'last': the value at the run's most recent logged epoch.

    Paths with no log.csv (not yet trained past the first epoch) are skipped rather
    than raising, so a run directory tree with in-progress runs can be scanned as-is.

    Parameters:
    - paths (list of str): Paths to the experiment directories (e.g. from run_path()).
    - key (str): 'best' or 'last'; see above.
    - min_reward (float, optional): Drop runs scoring below this reward.
    - top (int, optional): Keep only the top-scoring `top` runs.

    Returns a list of (path, reward) tuples sorted by reward, descending.
    """
    if key not in ('best', 'last'):
        raise ValueError(f"key must be 'best' or 'last', got {key!r}")

    scored = []
    for path in paths:
        log_path = os.path.join(path, 'log.csv')
        if not os.path.isfile(log_path):
            continue
        scores = pd.read_csv(log_path)['test/episode_score/mean']
        reward = float(scores.max() if key == 'best' else scores.iloc[-1])
        if min_reward is not None and reward < min_reward:
            continue
        scored.append((path, reward))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top] if top is not None else scored


def draw_network(mode='NCAP', N=2, include_speed_control=False, include_turn_control=False):
    """
    Draws a network graph for a swimmer model based on either NCAP or MLP architecture.

    Parameters:
    - mode (str): Determines the architecture type ('NCAP' or 'MLP'). Defaults to 'NCAP'.
    - N (int): Number of joints in the swimmer model. Defaults to 2.
    - include_speed_control (bool): If True, includes nodes for speed control in the graph.
    - include_turn_control (bool): If True, includes nodes for turn control in the graph.
    """

    G = nx.DiGraph()

    n = 2 + N * 4

    nodes = dict()

    if include_speed_control:
        nodes['1-s'] = n + 7
    if include_turn_control:
        nodes['r'] = n + 5
        nodes['l'] = n + 3

    nodes['o'] = n - 1
    nodes['$o^d$'] = n - 0
    nodes['$o^v$'] = n - 2

    custom_node_positions = {}
    custom_node_positions['o'] = (1, nodes['o'])
    custom_node_positions['$o^d$'] = (1.5, nodes['$o^d$'])
    custom_node_positions['$o^v$'] = (1.5, nodes['$o^v$'])

    if include_speed_control:
        custom_node_positions['1-s'] = (1.5, nodes['1-s'])
    if include_turn_control:
        custom_node_positions['r'] = (1.5, nodes['r'])
        custom_node_positions['l'] = (1.5, nodes['l'])

    for i in range(1, N + 1):
        nodes[f'$q_{i}$'] = 4 * (N - i) + 1
        nodes[f'$q^d_{i}$'] = 4 * (N - i) + 2
        nodes[f'$q^v_{i}$'] = 4 * (N - i)
        nodes[f'$b^d_{i}$'] = 4 * (N - i) + 2
        nodes[f'$b^v_{i}$'] = 4 * (N - i)
        nodes[f'$m^d_{i}$'] = 4 * (N - i) + 2
        nodes[f'$m^v_{i}$'] = 4 * (N - i)
        nodes[r'$\overset{..}{q}$' + f'$_{i}$'] = 4 * (N - i) + 1

        custom_node_positions[f'$q_{i}$'] = (1, nodes[f'$q_{i}$'])
        custom_node_positions[f'$q^d_{i}$'] = (1.5, nodes[f'$q^d_{i}$'])
        custom_node_positions[f'$q^v_{i}$'] = (1.5, nodes[f'$q^v_{i}$'])
        custom_node_positions[f'$b^d_{i}$'] = (2, nodes[f'$b^d_{i}$'])
        custom_node_positions[f'$b^v_{i}$'] = (2, nodes[f'$b^v_{i}$'])
        custom_node_positions[f'$m^d_{i}$'] = (2.5, nodes[f'$m^d_{i}$'])
        custom_node_positions[f'$m^v_{i}$'] = (2.5, nodes[f'$m^v_{i}$'])
        custom_node_positions[r'$\overset{..}{q}$' + f'$_{i}$'] = (
            3, nodes[r'$\overset{..}{q}$' + f'$_{i}$']
        )

    for node, layer in nodes.items():
        G.add_node(node, layer=layer)

    if mode == 'NCAP':
        # Add edges between nodes
        edges_colors = ['green', 'orange', 'green', 'green']
        edge_labels = {
            ('o', '$o^d$'): '+1',
            ('o', '$o^v$'): '-1',
            ('$o^d$', '$b^d_1$'): 'o',
            ('$o^v$', '$b^v_1$'): 'o',
        }

        if include_speed_control:
            edges_colors += ['orange']
            edge_labels[('1-s', '$b^d_1$')] = 's, to all b'
        if include_turn_control:
            edges_colors += ['green', 'green']
            edge_labels[('r', '$b^d_1$')] = 't'
            edge_labels[('l', '$b^v_1$')] = 't'

        for i in range(1, N + 1):
            if i < N:
                edges_colors += ['green', 'orange', 'green', 'green']

                edge_labels[(f'$q_{i}$', f'$q^d_{i}$')] = '+1'
                edge_labels[(f'$q_{i}$', f'$q^v_{i}$')] = '-1'
                edge_labels[(f'$q^d_{i}$', f'$b^d_{i+1}$')] = 'p'
                edge_labels[(f'$q^v_{i}$', f'$b^v_{i+1}$')] = 'p'

            edges_colors += ['green', 'orange', 'green', 'orange', 'orange', 'green']

            edge_labels[(f'$b^d_{i}$', f'$m^d_{i}$')] = 'i'
            edge_labels[(f'$b^d_{i}$', f'$m^v_{i}$')] = 'c'
            edge_labels[(f'$b^v_{i}$', f'$m^v_{i}$')] = 'i'
            edge_labels[(f'$b^v_{i}$', f'$m^d_{i}$')] = 'c'
            edge_labels[(f'$m^v_{i}$', r'$\overset{..}{q}$' + f'$_{i}$')] = '-1'
            edge_labels[(f'$m^d_{i}$', r'$\overset{..}{q}$' + f'$_{i}$')] = '+1'

        edges = edge_labels.keys()

    elif mode == 'MLP':
        edges = []
        layers = [1, 1.5, 2, 2.5, 3]
        layers_nodes = [[], [], [], [], []]
        for key, value in custom_node_positions.items():
            ind = layers.index(value[0])
            layers_nodes[ind].append(key)
        for layer_ind in range(len(layers_nodes) - 1):
            for node1 in layers_nodes[layer_ind]:
                for node2 in layers_nodes[layer_ind + 1]:
                    edges.append((node1, node2))
        edges_colors = np.repeat('gray', len(edges))

    G.add_edges_from(edges)

    # Draw the graph using the custom node positions
    options = {
        "edge_color": edges_colors,
        "edgecolors": "tab:gray",
        "node_size": 500,
        'node_color': 'white',
    }
    nx.draw(G, pos=custom_node_positions, with_labels=True, arrowstyle="-", arrowsize=20, **options)
    if mode == 'NCAP':
        nx.draw_networkx_edge_labels(G, pos=custom_node_positions, edge_labels=edge_labels)
