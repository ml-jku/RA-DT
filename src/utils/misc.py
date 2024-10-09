import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm


def maybe_split(dir_name: str) -> str:
    """
    Recursively splits a given dir_name at half, once it exceeds max folder size of 255.
    """
    if len(dir_name) > 255:
        half = len(dir_name) // 2
        dir_name = maybe_split(dir_name[:half]) + "/" + maybe_split(dir_name[half:])
    return dir_name

def safe_mean(arr): 
    return np.nan if len(arr) == 0 else np.mean(arr)


def set_frozen_to_eval(module):
    requires_grad = []
    for p in module.parameters():
        requires_grad.append(p.requires_grad)
    if not any(requires_grad):
        module.eval()


def make_attention_maps(attention_scores, step, lower_triu=True, vmin=None, vmax=None):
    """
    attention_scores: Tuple of `torch.FloatTensor` (one for each layer) of shape
        `(batch_size, num_heads, sequence_length,sequence_length)`.
    step: Int. Current timestep

    """
    figures = {}
    mask = None
    for i, scores in enumerate(attention_scores):
        # first attention head
        if scores is None:
            print(f"Attention scores for layer {i} are None. Skipping")
            continue
        scores = scores.float().detach().cpu().numpy()
        h0_scores = scores[-1, 0]
        fig, ax = plt.subplots()
        if lower_triu:
            mask = np.triu(np.ones_like(h0_scores, dtype=bool))
            np.fill_diagonal(mask, False)
        sns.heatmap(h0_scores, cmap="rocket_r", mask=mask, ax=ax, vmin=vmin, vmax=vmax)
        ax.set_title(f"Timestep: {step}, Layer: {i}, Head: 0")
        figures[f"layer{i}_head0"] = fig
        # avg over all heads
        avg_scores = scores[-1].mean(0)
        fig, ax = plt.subplots()
        if lower_triu:
            mask = np.triu(np.ones_like(avg_scores, dtype=bool))
            np.fill_diagonal(mask, False)
        sns.heatmap(avg_scores, cmap="rocket_r", mask=mask, ax=ax, vmin=vmin, vmax=vmax)
        ax.set_title(f"Timestep: {step}, Layer: {i}, Head: all")
        figures[f"layer{i}_allheads"] = fig
    return figures


def make_qk_dist_plot(key, query, step):
    key, query = key.squeeze(), query.squeeze()
    df_key = pd.DataFrame(key.T, columns=[f"k{i}" for i in range(key.shape[0])])
    df_query = pd.DataFrame(query.T, columns=[f"q{i}" for i in range(query.shape[0])])
    df = pd.concat([df_key, df_query], axis=1).T
    fig, ax = plt.subplots()
    sns.heatmap(df, cmap="rocket_r", ax=ax)
    ax.set_title(f"Timestep: {str(step)}")
    ax.set_xlabel("Feature dimension")
    ax.set_ylabel("Q-K index")
    return fig


def make_sim_plot(sim, step, max_samples=5):
    """
    Make heatmap from given similarity matrix.
    Args:
        sim: np.ndarray of shape (batch_size x pool_size)
        step: Int.
        max_samples: Int. Max samples to use (across batch size). Matrix becomes unreadable for more than 10 samples.

    Returns: Matplotlib figure.

    """
    fig, ax = plt.subplots(figsize=(max_samples, sim.shape[1] * 0.3))
    if sim.shape[0] > max_samples:
        sim = sim[:max_samples]
    sns.heatmap(sim.T, cmap="rocket_r", ax=ax, annot=True)
    ax.set_title(f"Timestep: {str(step)}")
    ax.set_xlabel("Batch idx")
    ax.set_ylabel("Pool idx")
    return fig


def make_retrieved_states_plot(state, action, states_retrieved, actions_retrieved, step):
    """
    Plots retrieved states next to current state in addition to the performed actions as title.

    Args:
        state: np.ndarray of shape (H, W, C)
        states_retrieved: np.ndarray of shape (B, H, W, C)
        action: np.ndarray of shape (1, act_dim)
        actions_retrieved: np.ndarray of shape (B, act_dim)
        step: int

    Returns: Matplotlib figure.

    """
    num_retrieved = len(states_retrieved)
    fig, axs = plt.subplots(1, num_retrieved + 1, figsize=(4 * (num_retrieved + 1), 4))
    state, states_retrieved = state.cpu().numpy(), states_retrieved.cpu().numpy()
    action, actions_retrieved = action.cpu().numpy(), actions_retrieved.cpu().numpy()
    if len(state.shape) == 3: 
        state = state.squeeze(0)
    
    # Plot current state in first subplot
    axs[0].imshow(state)
    axs[0].set_title(f"Current state | Action: {action}")
    
    # Plot retrieved states in remaining subplots
    for i, ret in enumerate(states_retrieved):
        if len(ret.shape) == 3: 
            ret = ret.squeeze(0)
        axs[i + 1].imshow(ret)
        axs[i + 1].set_title(f"Retrieved {i+1} | Action: {actions_retrieved[i]}")
        
    fig.suptitle(f"Evaluation Step: {step}")
    
    return fig


class ProgressParallel(joblib.Parallel):
    # from: https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
