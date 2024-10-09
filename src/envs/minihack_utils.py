import itertools
import gym
import numpy as np
import cv2
from nle import nethack
from minihack.envs import register
from minihack import MiniHackNavigation, LevelGenerator, RewardManager
from minihack.reward_manager import SequentialRewardManager
from minihack.tiles.window import Window
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from omegaconf.listconfig import ListConfig


class ExtraDictWrapper(gym.ObservationWrapper):
    """
    Wrapper to extract a specific key from a dictionary observation space.
        
    """
    def __init__(self, env: gym.Env, obs_key="tty_cursor") -> None:
        super().__init__(env)
        self.obs_key = obs_key
        self.observation_space = env.observation_space.spaces[obs_key]

    def observation(self, obs: dict):
        return obs[self.obs_key]


class MiniHackRoomCustom(MiniHackNavigation):
    def __init__(
        self,
        *args,
        size=5,
        n_monster=0,
        n_trap=0,
        penalty_step=0,
        random=True,
        lit=True,
        goal_pos=None,
        start_pos=None,
        sparse_reward=False,
        dense_reward=False,
        width=None, 
        **kwargs
    ):
        """
        Custom version of MinihackRoom, empty room in which goal location is present.  
        Action space is up, right, down, left, do nothing.
        
        Args:
            size (int): The size of the grid.
            n_monster (int): The number of monsters in the environment.
            n_trap (int): The number of traps in the environment.
            penalty_step (float): The penalty for each step taken. We turn it off by default. 
            random (bool): Whether to set start_pos and goal_pos randomly or not. 
            lit (bool): Whether the environment is lit or not. For DarkRoom doesn't matter, as agent only sees x-y. 
            goal_pos (tuple): The position of the goal.
            start_pos (tuple): The position of the starting point.
            sparse_reward (bool): Whether to use sparse rewards or not. Reward only obtained once at goal state, episode
                continues after reward is obtained. 
            dense_reward (bool): Whether to use dense rewards or not. Reward in every step at goal state. 
            width (int): The width of the grid.
            **kwargs: Additional keyword arguments.

        """
        self.goal_pos = goal_pos
        self.start_pos = start_pos 
        # sparse_reward --> reward can only be collected once, but episodes do not end when collected
        # dense_reward --> reward can be collected multiple times, but episodes do not end when collected
        self.sparse_reward = sparse_reward
        self.dense_reward = dense_reward
        self.size = size
        self.width = width 
        kwargs["max_episode_steps"] = kwargs.pop(
            "max_episode_steps", 100
        )
        lvl_gen = LevelGenerator(w=size if self.width is None else self.width, h=size, lit=lit)
        
        if not sparse_reward and not dense_reward:
            if random and goal_pos is None and start_pos is None:
                lvl_gen.add_goal_pos()
            else:
                lvl_gen.add_goal_pos((size - 1, size - 1) if goal_pos is None else goal_pos)
                lvl_gen.set_start_pos((0, 0) if start_pos is None else start_pos)
        else:
            lvl_gen.set_start_pos((0, 0) if start_pos is None else start_pos)
            lvl_gen.add_fountain(place=goal_pos)
            reward_manager = RewardManager()
            if sparse_reward:
                # if reaches fountain, give reward of 1, but only once
                # for some reason, the env would stop, once the goal is reached once
                # therefore we make a second event, without reward
                reward_manager.add_location_event(location="fountain", reward=1, repeatable=False)
                reward_manager.add_location_event(location="fountain", reward=0, repeatable=True)
            elif dense_reward:
                # if reaches fountain, give reward of 1, every time
                reward_manager.add_location_event(location="fountain", reward=1, repeatable=True)
            kwargs["reward_manager"] = kwargs.pop("reward_manager", reward_manager)
        for _ in range(n_monster):
            lvl_gen.add_monster()

        for _ in range(n_trap):
            lvl_gen.add_trap()
            
        # up, right, down, left, do nothing
        actions = tuple(nethack.CompassCardinalDirection) + (ord("."),)	       
         
        super().__init__(*args, des_file=lvl_gen.get_des(), 
                         actions=actions, penalty_step=penalty_step, **kwargs)


class MinihackKeyDoor(MiniHackNavigation):
    def __init__(
        self,
        *args,
        size=5,
        n_monster=0,
        n_trap=0,
        penalty_step=0,
        random=False,
        lit=True,
        goal_pos=None,
        start_pos=None,
        key_pos=None,
        width=None, 
        **kwargs
    ):
        """
        Custom version of MinihackRoom in which a key and a goal location is present. 
        The goal location is locked and can only be opened with the key.
        Reward is received ones for picking up the key. If the goal location is found, the agent 
        receives a reward in every timestep it stays on the goal location. 
        Action space is up, right, down, left, do nothing.

        Args:
            size (int): The size of the grid.
            n_monster (int): The number of monsters in the environment.
            n_trap (int): The number of traps in the environment.
            penalty_step (float): The penalty for each step taken. We turn it off by default. 
            random (bool): Whether to set start_pos and goal_pos randomly or not. 
            lit (bool): Whether the environment is lit or not. For DarkRoom doesn't matter, as agent only sees x-y. 
            goal_pos (tuple): The position of the goal.
            start_pos (tuple): The position of the starting point.
            key_pos (tuple): The position of the key.
            width (int): The width of the grid.
            **kwargs: Additional keyword arguments.

        """
        self.goal_pos = goal_pos
        self.start_pos = start_pos 
        self.key_pos = key_pos
        self.size = size
        self.width = width 
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 100)
        # make sure to use autopickup of key, no pickup action required then
        kwargs["autopickup"] = True
        lvl_gen = LevelGenerator(w=size if self.width is None else self.width, h=size, lit=lit)
        lvl_gen.set_start_pos((0, 0) if start_pos is None else start_pos)

        # add key
        lvl_gen.add_object(symbol="(", name="skeleton key", place=key_pos)
        # construct door - open by default, such that agent can walk through
        lvl_gen.add_door(place=goal_pos, state="open")
        
        # sequential reward manager ensures that key has to be collected before reward on door goal can be collected
        reward_manager = SequentialRewardManager()
        # if reaches key automatically pick up, give reward of 1 once
        reward_manager.add_location_event(location="key", reward=1, repeatable=False)
        # if reaches door and has key, give reward of 1, every time. if no key, no reward.
        reward_manager.add_location_event(location="door", reward=1, repeatable=True)
        kwargs["reward_manager"] = kwargs.pop("reward_manager", reward_manager)
        
        for _ in range(n_monster):
            lvl_gen.add_monster()

        for _ in range(n_trap):
            lvl_gen.add_trap()
            
        # up, right, down, left, do nothing
        actions = tuple(nethack.CompassCardinalDirection) + (ord("."),)	       
         
        super().__init__(*args, des_file=lvl_gen.get_des(), 
                         actions=actions, penalty_step=penalty_step, **kwargs)


class MiniHackRoom10x10Dark(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=10, **kwargs)


class MiniHackRoom17x17Dark(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=17, **kwargs)

        
class MiniHackRoom10x10DarkDense(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=10, dense_reward=True, **kwargs)


class MiniHackRoom17x17DarkDense(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=17, dense_reward=True, **kwargs)
        

class MiniHackRoom20x20DarkDense(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        max_episode_steps = kwargs.pop("max_episode_steps", 400)
        super().__init__(*args, size=20, dense_reward=True, max_episode_steps=max_episode_steps, **kwargs)


class MiniHackRoom40x20DarkDense(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        max_episode_steps = kwargs.pop("max_episode_steps", 800)
        super().__init__(*args, width=40, size=20, dense_reward=True, max_episode_steps=max_episode_steps, **kwargs)


class MiniHackRoom10x10DarkSparse(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=10, sparse_reward=True, **kwargs)
    

class MiniHackRoom17x17DarkSparse(MiniHackRoomCustom):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=17, sparse_reward=True, **kwargs)
        
        
class MiniHackKeyDoor10x10DarkDense(MinihackKeyDoor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=10, **kwargs)
        
        
class MiniHackKeyDoor5x5DarkDense(MinihackKeyDoor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=6, **kwargs)
        

class MiniHackKeyDoor20x20DarkDense(MinihackKeyDoor):
    
    def __init__(self, *args, **kwargs):
        max_episode_steps = kwargs.pop("max_episode_steps", 400)
        super().__init__(*args, size=20, max_episode_steps=max_episode_steps, **kwargs)


class MiniHackKeyDoor40x20DarkDense(MinihackKeyDoor):
    
    def __init__(self, *args, **kwargs):
        max_episode_steps = kwargs.pop("max_episode_steps", 800)
        super().__init__(*args, width=40, size=20, max_episode_steps=max_episode_steps, **kwargs)

        
class ToRealCoordinateWrapper(gym.ObservationWrapper):
    """
    Converts the screen minihack coordinates to real coordinates in range [0, env_size]
    
    Args: 
        env: Gym environment.
    """

    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        # extract size from env, get starting x-y positions
        self.env_size = env.size
        self.env_width = env.width if hasattr(env, "width") else None
        self.origin = self.get_origin_xy(self.env_size)

    def observation(self, obs):
        return (obs[0] - self.origin[0], obs[1] - self.origin[1])
    
    def get_origin_xy(self, size): 
        if size == 10:
            return (8, 34)    
        elif size == 17: 
            return (4, 30)
        elif size == 20 and self.env_width == 40: 
            return (2, 20)
        elif size == 20: 
            return (2, 30)
        raise ValueError(f"Size {size} not supported.")
    
    
class WarpFrame(gym.ObservationWrapper):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, grayscale=False) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        assert isinstance(env.observation_space, gym.spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1 if self.grayscale else 3),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        if self.grayscale: 
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame if not self.grayscale else frame[:, :, None]


def register_dark_envs():     
    register(
        id="MiniHack-Room-Dark-10x10-v0",
        entry_point=MiniHackRoom10x10Dark,
    )
    register(
        id="MiniHack-Room-Dark-17x17-v0",
        entry_point=MiniHackRoom17x17Dark,
    )
    register(
        id="MiniHack-Room-Dark-Sparse-10x10-v0",
        entry_point=MiniHackRoom10x10DarkSparse,
    )
    register(
        id="MiniHack-Room-Dark-Sparse-17x17-v0",
        entry_point=MiniHackRoom17x17DarkSparse,
    )

    register(
        id="MiniHack-Room-Dark-Dense-10x10-v0",
        entry_point=MiniHackRoom10x10DarkDense,
    )
    register(
        id="MiniHack-Room-Dark-Dense-17x17-v0",
        entry_point=MiniHackRoom17x17DarkDense,
    )
    
    register(
        id="MiniHack-Room-Dark-Dense-20x20-v0",
        entry_point=MiniHackRoom20x20DarkDense,
    )
    register(
        id="MiniHack-Room-Dark-Dense-40x20-v0",
        entry_point=MiniHackRoom40x20DarkDense,
    )
    register(
        id="MiniHack-KeyDoor-Dark-Dense-10x10-v0",
        entry_point=MiniHackKeyDoor10x10DarkDense,
    )
    register(
        id="MiniHack-KeyDoor-Dark-Dense-5x5-v0",
        entry_point=MiniHackKeyDoor5x5DarkDense,
    )
    register(
        id="MiniHack-KeyDoor-Dark-Dense-20x20-v0",
        entry_point=MiniHackKeyDoor20x20DarkDense,
    )
    register(
        id="MiniHack-KeyDoor-Dark-Dense-40x20-v0",
        entry_point=MiniHackKeyDoor40x20DarkDense,
    )
    
    
register_dark_envs()


def get_minihack_constructor(envid, env_kwargs=None, goal_pos=None, start_pos=None, key_pos=None):
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    # need be tuples if given
    goal_pos = tuple(goal_pos) if goal_pos is not None else goal_pos
    start_pos = tuple(start_pos) if start_pos is not None else start_pos
    key_pos = tuple(key_pos) if key_pos is not None else key_pos
    def make():
        if "Room-Dark" in envid: 
            env = gym.make(envid, goal_pos=goal_pos, start_pos=start_pos, **env_kwargs)
        elif "KeyDoor-Dark" in envid: 
            env = gym.make(envid, goal_pos=goal_pos, start_pos=start_pos, key_pos=key_pos, **env_kwargs)
        else: 
            env = gym.make(envid, **env_kwargs)
        if "Dense" in envid:                 
            env.name = f"{envid}_{str(start_pos).replace(' ', '')}_{str(goal_pos).replace(' ', '')}"
            if key_pos is not None: 
                env.name += f"_{str(key_pos).replace(' ', '')}"
        observation_keys = env_kwargs.get("observation_keys", ["tty_cursor"])
        env = ExtraDictWrapper(env, observation_keys[0])
        if observation_keys[0] == "tty_cursor": 
            env = ToRealCoordinateWrapper(env)
        if observation_keys[0] == "pixel_crop": 
            env = WarpFrame(env, width=84, height=84, grayscale=False)
        return Monitor(env)
    return make


def get_minihack_constructors(envid, env_kwargs=None, goal_pos=None, start_pos=None, key_pos=None):
    # Case 1: None --> convery to list
    # Case 2: single list --> convert to list of lists
    # Case 3: already list of lists --> then Omegaconf list --> do nothing
    if goal_pos is None: 
        goal_pos = [goal_pos]
    elif isinstance(goal_pos, (tuple, list, ListConfig)): 
        goal_pos = [goal_pos] if isinstance(goal_pos[0], int) else goal_pos
    if start_pos is None:
        start_pos = [start_pos] 
    elif isinstance(start_pos, (tuple, list, ListConfig)):
        start_pos = [start_pos] if isinstance(start_pos[0], int) else start_pos 
    if isinstance(key_pos, (tuple, list, ListConfig)):
        key_pos = [key_pos] if isinstance(key_pos[0], int) else key_pos 
        assert len(key_pos) == len(goal_pos), "Number of key positions and goal positions must be the same."
    # repeat shorter one
    if len(start_pos) < len(goal_pos):
        start_pos = itertools.cycle(start_pos)
    elif len(start_pos) > len(goal_pos):
        goal_pos = itertools.cycle(goal_pos)
    if key_pos is not None: 
        return [get_minihack_constructor(envid, env_kwargs, goal, start, key) for goal, start, key in zip(goal_pos, start_pos, key_pos)]
    else:  
        return [get_minihack_constructor(envid, env_kwargs, goal, start) for goal, start in zip(goal_pos, start_pos)]
    

def make_train_test_pos(num_pos=100, size=10, percent_train=0.8, width=None): 
    if width is None: 
        width = size
    n_train = int(num_pos * percent_train)
    pos = [(i, j) for i in range(width) for j in range(size)]
    start_pos = np.random.RandomState(seed=42).permutation(pos)[:num_pos]
    goal_pos = np.random.RandomState(seed=43).permutation(pos)[:num_pos]
    train_start, train_goal = start_pos[:n_train].tolist(), goal_pos[:n_train].tolist()
    test_start, test_goal = start_pos[n_train:].tolist(), goal_pos[n_train:].tolist()
    assert [start != goal for start, goal in zip(train_start, train_goal)], "Start and goal are same in train."
    assert [start != goal for start, goal in zip(test_start, test_goal)], "Start and goal are same in test."
    return train_start, train_goal, test_start, test_goal


def make_minihack_envs(env_params, make_eval_env=True):
    const_kwargs = {
        "envid": env_params.envid,
        "env_kwargs": env_params.get("env_kwargs", {}),
        "goal_pos": env_params.get("train_goal_pos", None), 
        "start_pos": env_params.get("train_start_pos", None)  
    }
    if hasattr(env_params, "train_key_pos"):
        const_kwargs["key_pos"] = env_params.train_key_pos
    env = DummyVecEnv(get_minihack_constructors(**const_kwargs))
    if make_eval_env:
        goal_pos, start_pos = env_params.get("eval_goal_pos", None), env_params.get("eval_start_pos", None)
        const_kwargs.update({"goal_pos": goal_pos, "start_pos": start_pos})
        if hasattr(env_params, "eval_key_pos"):
            const_kwargs["key_pos"] = env_params.eval_key_pos
        eval_env = DummyVecEnv(get_minihack_constructors(**const_kwargs))
        eval_env.num_envs = 1
    env.num_envs = 1
    return env, eval_env


if __name__ == "__main__": 
    # 10x10
    train_start, train_goal, test_start, test_goal = make_train_test_pos()
    """
    # 10x10
    print("Train starts: ", train_start)
    print("Train goals: ", train_goal)
    print("Test starts: ", test_start)
    print("Test goals: ", test_goal)
    
    Train starts:  [[8, 3], [5, 3], [7, 0], [4, 5], [4, 4], [3, 9], [2, 2], [8, 0], [1, 0], [0, 0], [1, 8], [3, 0], [7, 3], [3, 3], [9, 0], [0, 4], [7, 6], [7, 7], [1, 2], [3, 1], [5, 5], [8, 8], [2, 6], [4, 2], [6, 9], [1, 5], [4, 0], [9, 6], [0, 9], [7, 2], [1, 1], [4, 7], [8, 5], [2, 8], [9, 3], [0, 5], [6, 6], [6, 5], [3, 5], [1, 6], [4, 9], [3, 4], [0, 7], [9, 5], [2, 7], [1, 9], [8, 1], [2, 5], [6, 2], [1, 3], [2, 4], [0, 3], [1, 7], [3, 8], [0, 8], [7, 8], [0, 6], [6, 4], [3, 6], [8, 9], [5, 6], [9, 9], [5, 4], [4, 3], [5, 0], [6, 7], [4, 6], [6, 8], [6, 1], [9, 7], [7, 9], [4, 1], [5, 8], [4, 8], [9, 8], [5, 7], [7, 5], [3, 2], [9, 4], [5, 9]]
    Train goals:   [[2, 0], [0, 2], [1, 5], [2, 2], [5, 7], [9, 1], [6, 9], [5, 5], [1, 1], [7, 9], [0, 9], [3, 8], [8, 5], [0, 0], [8, 9], [1, 3], [0, 5], [0, 1], [9, 5], [8, 3], [4, 4], [1, 2], [7, 8], [9, 4], [3, 7], [9, 2], [9, 7], [5, 6], [6, 3], [4, 6], [0, 8], [3, 3], [4, 5], [1, 9], [1, 4], [9, 3], [7, 3], [3, 9], [2, 4], [0, 6], [6, 2], [2, 3], [1, 8], [4, 2], [8, 0], [8, 6], [3, 1], [6, 7], [2, 7], [1, 0], [4, 0], [7, 0], [9, 6], [8, 8], [2, 6], [5, 4], [7, 1], [2, 9], [2, 5], [4, 3], [4, 1], [7, 2], [0, 3], [8, 1], [5, 3], [7, 7], [6, 1], [0, 7], [2, 8], [9, 9], [5, 2], [4, 8], [8, 2], [6, 0], [7, 4], [3, 2], [4, 7], [7, 6], [3, 6], [0, 4]]
    Test starts:   [[6, 3], [8, 4], [3, 7], [2, 9], [0, 1], [5, 2], [2, 1], [0, 2], [2, 3], [8, 7], [9, 1], [7, 4], [8, 6], [8, 2], [2, 0], [6, 0], [7, 1], [1, 4], [9, 2], [5, 1]]
    Test goals:    [[9, 0], [8, 4], [3, 4], [6, 5], [7, 5], [5, 0], [3, 5], [9, 8], [8, 7], [3, 0], [6, 6], [5, 9], [1, 7], [5, 1], [1, 6], [5, 8], [2, 1], [4, 9], [6, 4], [6, 8]]
    
    """  
    
    # 20x20
    train_start, train_goal, test_start, test_goal = make_train_test_pos(size=20)
    """
    # 20x20
    print("Train starts: ", train_start)
    print("Train goals: ", train_goal)
    print("Test starts: ", test_start)
    print("Test goals: ", test_goal)
    
    Train starts:  [[10, 9], [14, 0], [1, 13], [10, 10], [4, 13], [4, 4], [16, 9], [4, 14], [13, 6], [6, 6], [0, 9], [18, 1], [2, 16], [3, 12], [6, 12], [2, 2], [13, 18], [18, 16], [11, 11], [19, 5], [3, 17], [0, 15], [19, 11], [13, 11], [0, 0], [19, 16], [5, 14], [11, 5], [13, 2], [5, 4], [19, 15], [9, 13], [13, 1], [2, 17], [11, 12], [5, 16], [5, 13], [17, 2], [7, 18], [7, 1], [19, 1], [2, 15], [3, 16], [1, 5], [4, 2], [19, 2], [7, 8], [9, 1], [1, 2], [8, 13], [2, 6], [16, 1], [16, 18], [3, 10], [18, 14], [1, 19], [11, 3], [8, 12], [1, 10], [7, 12], [6, 4], [14, 14], [12, 15], [3, 18], [5, 1], [1, 11], [17, 12], [13, 8], [19, 14], [3, 13], [16, 0], [7, 0], [0, 5], [2, 5], [19, 8], [12, 6], [11, 7], [18, 9], [8, 16], [14, 9]]
    Train goals:  [[2, 15], [13, 11], [19, 10], [5, 5], [11, 3], [17, 18], [1, 0], [15, 9], [19, 15], [6, 19], [6, 18], [9, 5], [4, 14], [4, 2], [18, 1], [19, 2], [7, 3], [17, 15], [16, 8], [17, 9], [15, 1], [3, 4], [1, 7], [11, 8], [19, 16], [6, 16], [3, 7], [12, 3], [8, 19], [11, 14], [16, 18], [14, 4], [0, 12], [11, 13], [17, 1], [3, 2], [11, 10], [2, 19], [11, 9], [5, 3], [16, 2], [19, 9], [0, 15], [10, 7], [2, 4], [12, 17], [17, 5], [19, 1], [8, 16], [13, 3], [0, 3], [10, 5], [11, 1], [16, 1], [0, 17], [12, 7], [6, 1], [8, 5], [13, 1], [18, 18], [16, 5], [1, 9], [19, 8], [1, 10], [0, 0], [16, 7], [9, 10], [13, 6], [15, 12], [1, 18], [18, 4], [3, 9], [15, 6], [1, 16], [18, 8], [6, 6], [7, 0], [18, 0], [14, 9], [9, 3]]
    Test starts:  [[0, 3], [0, 18], [10, 2], [12, 10], [13, 14], [3, 3], [12, 8], [15, 1], [5, 8], [4, 10], [11, 13], [16, 15], [5, 18], [11, 0], [9, 0], [15, 14], [18, 13], [19, 0], [11, 19], [3, 15]]
    Test goals:  [[16, 12], [9, 15], [1, 6], [5, 17], [7, 17], [8, 17], [10, 14], [9, 19], [10, 10], [16, 17], [8, 6], [9, 13], [15, 8], [15, 18], [15, 15], [3, 18], [17, 17], [10, 8], [5, 2], [0, 18]]

    """  
    
    # 40x20
    train_start, train_goal, test_start, test_goal = make_train_test_pos(size=20, width=40)    
    """
    # 40x20
    print("Train starts: ", train_start)
    print("Train goals: ", train_goal)
    print("Test starts: ", test_start)
    print("Test goals: ", test_goal)
    
    Train starts:  [[34, 16], [33, 7], [3, 3], [26, 13], [3, 6], [31, 1], [17, 6], [24, 10], [38, 0], [22, 16], [3, 5], [14, 6], [31, 15], [3, 7], [16, 7], [19, 15], [12, 4], [18, 17], [30, 10], [26, 6], [32, 18], [26, 9], [31, 2], [36, 1], [18, 0], [1, 10], [13, 0], [31, 17], [37, 6], [28, 10], [10, 15], [3, 18], [28, 18], [19, 3], [1, 19], [1, 3], [34, 12], [19, 18], [39, 16], [37, 1], [6, 19], [12, 10], [8, 14], [16, 3], [29, 15], [26, 5], [21, 3], [29, 16], [30, 4], [26, 14], [13, 5], [5, 9], [33, 6], [14, 14], [16, 16], [18, 8], [22, 6], [16, 13], [9, 18], [8, 8], [24, 1], [36, 3], [32, 15], [15, 6], [11, 11], [21, 2], [2, 9], [39, 7], [17, 17], [4, 6], [27, 5], [9, 12], [39, 15], [1, 13], [1, 11], [9, 19], [18, 5], [24, 6], [29, 14], [28, 8]]
    Train goals:  [[29, 6], [38, 6], [36, 2], [29, 0], [27, 2], [32, 7], [10, 8], [36, 11], [37, 18], [9, 8], [8, 16], [10, 10], [13, 8], [6, 2], [4, 11], [17, 17], [12, 16], [34, 1], [20, 16], [32, 9], [19, 4], [11, 10], [16, 14], [19, 6], [27, 15], [17, 2], [28, 4], [18, 14], [26, 7], [30, 15], [17, 8], [35, 10], [15, 14], [24, 0], [6, 8], [19, 12], [11, 17], [19, 18], [26, 16], [22, 13], [0, 11], [0, 17], [39, 2], [39, 8], [19, 14], [21, 8], [25, 15], [4, 2], [4, 14], [1, 4], [34, 12], [11, 8], [37, 5], [18, 9], [25, 4], [22, 16], [20, 6], [18, 0], [35, 8], [9, 10], [12, 1], [34, 16], [38, 17], [20, 18], [39, 3], [35, 5], [3, 7], [16, 11], [31, 0], [14, 17], [23, 4], [12, 10], [2, 7], [29, 13], [25, 12], [9, 16], [13, 9], [20, 4], [2, 17], [26, 10]]
    Test starts:  [[21, 8], [6, 17], [3, 12], [3, 17], [25, 12], [39, 3], [39, 6], [13, 15], [18, 1], [4, 16], [32, 1], [35, 15], [14, 1], [19, 13], [31, 8], [30, 8], [35, 5], [37, 14], [2, 14], [37, 10]]
    Test goals:  [[14, 13], [31, 2], [5, 19], [18, 5], [29, 9], [5, 11], [28, 8], [22, 10], [1, 15], [27, 11], [15, 17], [5, 5], [4, 8], [38, 5], [21, 6], [24, 15], [7, 19], [13, 16], [13, 1], [12, 9]]
        
    """  
    
    # 10x10 KeyDoor
    train_key, train_goal, test_key, test_goal = make_train_test_pos(size=10, width=10)  
    print("Train keys: ", train_key)
    print("Train goals: ", train_goal)
    print("Test keys: ", test_key)
    print("Test goals: ", test_goal)  
    """
    # 10x10 KeyDoor
    print("Train keys: ", train_key)
    print("Train goals: ", train_goal)
    print("Test keys: ", test_key)
    print("Test goals: ", test_goal)  
    
    Train keys:  [[8, 3], [5, 3], [7, 0], [4, 5], [4, 4], [3, 9], [2, 2], [8, 0], [1, 0], [0, 0], [1, 8], [3, 0], [7, 3], [3, 3], [9, 0], [0, 4], [7, 6], [7, 7], [1, 2], [3, 1], [5, 5], [8, 8], [2, 6], [4, 2], [6, 9], [1, 5], [4, 0], [9, 6], [0, 9], [7, 2], [1, 1], [4, 7], [8, 5], [2, 8], [9, 3], [0, 5], [6, 6], [6, 5], [3, 5], [1, 6], [4, 9], [3, 4], [0, 7], [9, 5], [2, 7], [1, 9], [8, 1], [2, 5], [6, 2], [1, 3], [2, 4], [0, 3], [1, 7], [3, 8], [0, 8], [7, 8], [0, 6], [6, 4], [3, 6], [8, 9], [5, 6], [9, 9], [5, 4], [4, 3], [5, 0], [6, 7], [4, 6], [6, 8], [6, 1], [9, 7], [7, 9], [4, 1], [5, 8], [4, 8], [9, 8], [5, 7], [7, 5], [3, 2], [9, 4], [5, 9]]
    Train goals:  [[2, 0], [0, 2], [1, 5], [2, 2], [5, 7], [9, 1], [6, 9], [5, 5], [1, 1], [7, 9], [0, 9], [3, 8], [8, 5], [0, 0], [8, 9], [1, 3], [0, 5], [0, 1], [9, 5], [8, 3], [4, 4], [1, 2], [7, 8], [9, 4], [3, 7], [9, 2], [9, 7], [5, 6], [6, 3], [4, 6], [0, 8], [3, 3], [4, 5], [1, 9], [1, 4], [9, 3], [7, 3], [3, 9], [2, 4], [0, 6], [6, 2], [2, 3], [1, 8], [4, 2], [8, 0], [8, 6], [3, 1], [6, 7], [2, 7], [1, 0], [4, 0], [7, 0], [9, 6], [8, 8], [2, 6], [5, 4], [7, 1], [2, 9], [2, 5], [4, 3], [4, 1], [7, 2], [0, 3], [8, 1], [5, 3], [7, 7], [6, 1], [0, 7], [2, 8], [9, 9], [5, 2], [4, 8], [8, 2], [6, 0], [7, 4], [3, 2], [4, 7], [7, 6], [3, 6], [0, 4]]
    Test keys:  [[6, 3], [3, 1], [3, 7], [2, 9], [0, 1], [5, 2], [2, 1], [0, 2], [2, 3], [8, 7], [9, 1], [7, 4], [8, 6], [8, 2], [2, 0], [6, 0], [7, 1], [1, 4], [9, 2], [5, 1]]
    Test goals:  [[9, 0], [8, 4], [3, 4], [6, 5], [7, 5], [5, 0], [3, 5], [9, 8], [8, 7], [3, 0], [6, 6], [5, 9], [1, 7], [5, 1], [1, 6], [5, 8], [2, 1], [4, 9], [6, 4], [6, 8] 
    
    """  
    
    # 20x20 KeyDoor
    train_key, train_goal, test_key, test_goal = make_train_test_pos(size=20, width=20)  
    print("Train keys: ", train_key)
    print("Train goals: ", train_goal)
    print("Test keys: ", test_key)
    print("Test goals: ", test_goal)  
    
    """
    # 10x10 KeyDoor
    print("Train keys: ", train_key)
    print("Train goals: ", train_goal)
    print("Test keys: ", test_key)
    print("Test goals: ", test_goal)  
    
    train_keys = [[10, 9], [14, 0], [1, 13], [10, 10], [4, 13], [4, 4], [16, 9], [4, 14], [13, 6], [6, 6], [0, 9], [18, 1], [2, 16], [3, 12], [6, 12], [2, 2], [13, 18], [18, 16], [11, 11], [19, 5], [3, 17], [0, 15], [19, 11], [13, 11], [0, 0], [19, 16], [5, 14], [11, 5], [13, 2], [5, 4], [19, 15], [9, 13], [13, 1], [2, 17], [11, 12], [5, 16], [5, 13], [17, 2], [7, 18], [7, 1], [19, 1], [2, 15], [3, 16], [1, 5], [4, 2], [19, 2], [7, 8], [9, 1], [1, 2], [8, 13], [2, 6], [16, 1], [16, 18], [3, 10], [18, 14], [1, 19], [11, 3], [8, 12], [1, 10], [7, 12], [6, 4], [14, 14], [12, 15], [3, 18], [5, 1], [1, 11], [17, 12], [13, 8], [19, 14], [3, 13], [16, 0], [7, 0], [0, 5], [2, 5], [19, 8], [12, 6], [11, 7], [18, 9], [8, 16], [14, 9]]
    train_goals =  [[2, 15], [13, 11], [19, 10], [5, 5], [11, 3], [17, 18], [1, 0], [15, 9], [19, 15], [6, 19], [6, 18], [9, 5], [4, 14], [4, 2], [18, 1], [19, 2], [7, 3], [17, 15], [16, 8], [17, 9], [15, 1], [3, 4], [1, 7], [11, 8], [19, 16], [6, 16], [3, 7], [12, 3], [8, 19], [11, 14], [16, 18], [14, 4], [0, 12], [11, 13], [17, 1], [3, 2], [11, 10], [2, 19], [11, 9], [5, 3], [16, 2], [19, 9], [0, 15], [10, 7], [2, 4], [12, 17], [17, 5], [19, 1], [8, 16], [13, 3], [0, 3], [10, 5], [11, 1], [16, 1], [0, 17], [12, 7], [6, 1], [8, 5], [13, 1], [18, 18], [16, 5], [1, 9], [19, 8], [1, 10], [0, 0], [16, 7], [9, 10], [13, 6], [15, 12], [1, 18], [18, 4], [3, 9], [15, 6], [1, 16], [18, 8], [6, 6], [7, 0], [18, 0], [14, 9], [9, 3]]
    test_keys =  [[0, 3], [0, 18], [10, 2], [12, 10], [13, 14], [3, 3], [12, 8], [15, 1], [5, 8], [4, 10], [11, 13], [16, 15], [5, 18], [11, 0], [9, 0], [15, 14], [18, 13], [19, 0], [11, 19], [3, 15]]
    test_goals = [[16, 12], [9, 15], [1, 6], [5, 17], [7, 17], [8, 17], [10, 14], [9, 19], [10, 10], [16, 17], [8, 6], [9, 13], [15, 8], [15, 18], [15, 15], [3, 18], [17, 17], [10, 8], [5, 2], [0, 18]]
        
    """  
    
    # 40x20 KeyDoor
    train_key, train_goal, test_key, test_goal = make_train_test_pos(size=20, width=40)
    
    print("Train keys: ", train_key)
    print("Train goals: ", train_goal)
    print("Test keys: ", test_key)
    print("Test goals: ", test_goal)  
    
    """
    print("Train keys: ", train_key)
    print("Train goals: ", train_goal)
    print("Test keys: ", test_key)
    print("Test goals: ", test_goal)  
    
    train_keys = [[34, 16], [33, 7], [3, 3], [26, 13], [3, 6], [31, 1], [17, 6], [24, 10], [38, 0], [22, 16], [3, 5], [14, 6], [31, 15], [3, 7], [16, 7], [19, 15], [12, 4], [18, 17], [30, 10], [26, 6], [32, 18], [26, 9], [31, 2], [36, 1], [18, 0], [1, 10], [13, 0], [31, 17], [37, 6], [28, 10], [10, 15], [3, 18], [28, 18], [19, 3], [1, 19], [1, 3], [34, 12], [19, 18], [39, 16], [37, 1], [6, 19], [12, 10], [8, 14], [16, 3], [29, 15], [26, 5], [21, 3], [29, 16], [30, 4], [26, 14], [13, 5], [5, 9], [33, 6], [14, 14], [16, 16], [18, 8], [22, 6], [16, 13], [9, 18], [8, 8], [24, 1], [36, 3], [32, 15], [15, 6], [11, 11], [21, 2], [2, 9], [39, 7], [17, 17], [4, 6], [27, 5], [9, 12], [39, 15], [1, 13], [1, 11], [9, 19], [18, 5], [24, 6], [29, 14], [28, 8]]
    train_goals = [[29, 6], [38, 6], [36, 2], [29, 0], [27, 2], [32, 7], [10, 8], [36, 11], [37, 18], [9, 8], [8, 16], [10, 10], [13, 8], [6, 2], [4, 11], [17, 17], [12, 16], [34, 1], [20, 16], [32, 9], [19, 4], [11, 10], [16, 14], [19, 6], [27, 15], [17, 2], [28, 4], [18, 14], [26, 7], [30, 15], [17, 8], [35, 10], [15, 14], [24, 0], [6, 8], [19, 12], [11, 17], [19, 18], [26, 16], [22, 13], [0, 11], [0, 17], [39, 2], [39, 8], [19, 14], [21, 8], [25, 15], [4, 2], [4, 14], [1, 4], [34, 12], [11, 8], [37, 5], [18, 9], [25, 4], [22, 16], [20, 6], [18, 0], [35, 8], [9, 10], [12, 1], [34, 16], [38, 17], [20, 18], [39, 3], [35, 5], [3, 7], [16, 11], [31, 0], [14, 17], [23, 4], [12, 10], [2, 7], [29, 13], [25, 12], [9, 16], [13, 9], [20, 4], [2, 17], [26, 10]]
    test_keys = [[21, 8], [6, 17], [3, 12], [3, 17], [25, 12], [39, 3], [39, 6], [13, 15], [18, 1], [4, 16], [32, 1], [35, 15], [14, 1], [19, 13], [31, 8], [30, 8], [35, 5], [37, 14], [2, 14], [37, 10]]
    test_goals = [[14, 13], [31, 2], [5, 19], [18, 5], [29, 9], [5, 11], [28, 8], [22, 10], [1, 15], [27, 11], [15, 17], [5, 5], [4, 8], [38, 5], [21, 6], [24, 15], [7, 19], [13, 16], [13, 1], [12, 9]]
        
    """ 
