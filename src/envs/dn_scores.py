"""
For DMControl and Gym MuJoCo, there are no human-normalized scores. 
Therefore, we normalize the scores based on the performance the expert agent reaches at the end of training. 
--> Data normalized

Random and expert refernence scores for D4RL are available here: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/infos.py

"""
import math
import dmc2gym
import numpy as np
import pandas as pd
from .env_names import DM_CONTROL_ENVS


# Task: score-tuple dictionary. Each score tuple contains
#  0: score random (float) and 1: mean scores in the datasets (float).
ENVID_TO_DNS = {
    'acrobot-swingup': (8.351, 4.877), 
    'ball_in_cup-catch': (0.0, 926.719), 
    'cartpole-balance': (350.391, 938.506),
    'cartpole-swingup': (27.414, 766.15),
    'cheetah-run': (3.207, 324.045),
    'finger-spin': (0.2, 834.629),
    'finger-turn_easy': (57.8, 800.645),
    'finger-turn_hard': (40.6, 676.144),
    'fish-swim': (67.675, 78.212),
    'fish-upright': (229.406, 547.962),
    'hopper-hop': (0.076, 62.794),
    'hopper-stand': (1.296, 266.783), 
    'humanoid-run': (0.741, 0.794), 
    'humanoid-stand': (4.327, 5.053),
    'humanoid-walk': (0.913, 1.194), 
    'manipulator-bring_ball': (0.0, 0.429), 
    'manipulator-insert_ball': (0.0, 43.307), 
    'manipulator-insert_peg': (0.235, 78.477), 
    'pendulum-swingup': (0.0, 614.491), 
    'point_mass-easy': (1.341, 779.273),
    'reacher-easy': (33.0, 849.241), 
    'reacher-hard': (8.0, 779.947), 
    'swimmer-swimmer15': (78.817, 152.297), 
    'swimmer-swimmer6': (229.834, 167.082),
    'walker-run': (23.427, 344.794),
    'walker-stand': (134.701, 816.322),
    'walker-walk': (30.193, 773.174),
    "HalfCheetah-v3":(-280.178953, 12135.0), 
    "Walker2d-v3": (1.629008, 4592.3),
    "Hopper-v3": (-20.272305, 3234.3),
    "HalfCheetah-v2":(-280.178953, 12135.0), 
    "Walker2d-v2": (1.629008, 4592.3), 
    "Hopper-v2": (-20.272305, 3234.3),
    # extracted from 25M data
    "bigfish":   (0.0, 5.9107),
    "bossfight": (0.0, 2.179),
    "caveflyer": (0.0, 7.6341),
    "chaser":    (0.0, 3.4349),
    "climber":   (0.0, 9.1516),
    "coinrun":   (0.0, 9.6781),
    "dodgeball": (0.0, 3.1873),
    "fruitbot":  (-7.0, 16.9643),
    "heist":     (0.0, 7.9555),
    "jumper":    (0.0, 8.7396),
    "leaper":    (0.0, 4.9065),
    "maze":      (0.0, 9.4536),
    "miner":     (0.0, 11.6814),
    "ninja":     (0.0, 7.7674),
    "plunder":   (0.0, 4.9095),
    "starpilot": (0.0, 17.3367),
    # metaworld - random scores, + mean scores in dataset
    'reach-v2': (333.01, 1842.41),
    'push-v2': (13.79, 1311.81),
    # for pick-place, we use the max score in the datasets, as the mean is ~random
    'pick-place-v2': (4.63, 1300.69),
    'door-open-v2': (96.88, 1484.42),
    'drawer-open-v2': (258.2, 1594.88),
    'drawer-close-v2': (297.9, 1823.24),
    'button-press-topdown-v2': (68.1, 1255.89),
    'peg-insert-side-v2': (3.84, 1418.33),
    'window-open-v2': (86.62, 1480.98),
    'window-close-v2': (117.4, 1395.35),
    'door-close-v2': (12.93, 1523.22),
    'reach-wall-v2': (268.53, 1817.09),
    'pick-place-wall-v2': (0.0, 260.63),
    'push-wall-v2': (24.91, 1361.09),
    'button-press-v2': (68.25, 1366.96),
    'button-press-topdown-wall-v2': (73.82, 1264.01),
    'button-press-wall-v2': (21.23, 1438.6),
    'peg-unplug-side-v2': (8.2, 1044.82),
    'disassemble-v2': (82.99, 1139.05),
    'hammer-v2': (202.6, 1444.97),
    'plate-slide-v2': (120.86, 1634.72),
    'plate-slide-side-v2': (34.02, 1598.42),
    'plate-slide-back-v2': (61.46, 1725.26),
    'plate-slide-back-side-v2': (76.74, 1716.65),
    'handle-press-v2': (130.96, 1824.9),
    'handle-pull-v2': (16.98, 1519.89),
    'handle-press-side-v2': (170.76, 1808.85),
    'handle-pull-side-v2': (5.09, 1484.7),
    'stick-push-v2': (5.53, 925.04),
    'stick-pull-v2': (4.6, 977.54),
    'basketball-v2': (5.08, 702.4),
    'soccer-v2': (13.44, 255.1),
    'faucet-open-v2': (516.4, 1712.81),
    'faucet-close-v2': (508.07, 1741.73),
    'coffee-push-v2': (8.62, 253.72),
    'coffee-pull-v2': (8.92, 973.07),
    'coffee-button-v2': (60.16, 1461.26),
    'sweep-v2': (21.74, 1024.95),
    'sweep-into-v2': (28.63, 1638.45),
    'pick-out-of-hole-v2': (2.73, 891.61),
    'assembly-v2': (92.02, 908.84),
    'shelf-place-v2': (0.0, 997.24),
    'push-back-v2': (2.33, 645.29),
    'lever-pull-v2': (99.86, 731.48),
    'dial-turn-v2': (49.74, 1556.25),
    'bin-picking-v2': (4.27, 249.41),
    'box-close-v2': (100.89, 550.25),
    'hand-insert-v2': (5.35, 1196.92),
    'door-lock-v2': (226.23, 1589.09),
    'door-unlock-v2': (192.42, 1662.54),
}


def get_data_normalized_score(task: str, raw_score: float, random_col=0, data_col=1) -> float:
    """Converts task score to data-normalized score."""
    scores = ENVID_TO_DNS.get(task, (math.nan, math.nan))
    random, data = scores[random_col], scores[data_col]
    return (raw_score - random) / (data - random)


def compute_random_dmcontrol_scores(): 
    random_scores = {}
    for envid in DM_CONTROL_ENVS:
        domain_name, task_name = envid.split("-")
        print(f"Computing random scores for {envid} ...")
        env = dmc2gym.make(domain_name=domain_name, task_name=task_name)
        random_scores[envid] = evaluate_random_policy(env)
    return random_scores


def evaluate_random_policy(env, n_eval_episodes=10):
    returns = []
    for _ in range(n_eval_episodes):
        _ = env.reset()
        done = False
        episode_return = 0
        while not done:
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            episode_return += reward
        returns.append(episode_return)
    return np.mean(returns)
