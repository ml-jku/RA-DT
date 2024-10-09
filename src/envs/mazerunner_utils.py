# Adapted from: https://github.com/UT-Austin-RPL/amago
import copy
import random
import numpy as np
# import gymnasium as 
import gym
import matplotlib.pyplot as plt
from gym.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import DummyVecEnv


def random_maze(width=11, height=11, complexity=0.75, density=0.75, seed=None):
    """
    Code from https://github.com/zuoxingdong/mazelab
    """
    if seed is not None: 
        rng = np.random.RandomState(seed)
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        if seed is not None: 
            x = rng.randint(0, shape[1] // 2 + 1) * 2
            y = rng.randint(0, shape[0] // 2 + 1) * 2
        else: 
            x = random.randrange(0, shape[1] // 2 + 1) * 2
            y = random.randrange(0, shape[0] // 2 + 1) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                if seed is not None: 
                    index = rng.randint(0, len(neighbours))
                else: 
                    index = random.randrange(0, len(neighbours))
                y_, x_ = neighbours[index]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_

    return Z.astype(int)


class MazeRunnerGymEnv(gym.Env):
    def __init__(
        self,
        maze_dim: int = 15,
        min_num_goals: int = 1,
        max_num_goals: int = 3,
        goal_in_obs: bool = False,
        reward_on_last_goal: bool = True,
        maze_seed=None,
        # only effective if maze_seed is not None
        rand_goals=False,
        render_mode="rgb_array"
    ):
        self.goal_in_obs = goal_in_obs
        assert min_num_goals <= max_num_goals
        self.max_num_goals = max_num_goals
        self.min_num_goals = min_num_goals
        self.maze_seed = maze_seed
        self.rand_goals = rand_goals
        self.reward_on_last_goal = reward_on_last_goal
        self.render_mode = render_mode
        
        # mazes have odd side length
        self.maze_dim = (maze_dim // 2) * 2 + 1
        self.reset()

        self.goal_space = gym.spaces.Box(
            low=np.array([-1, -1] * max_num_goals),
            high=np.array([maze_dim, maze_dim] * max_num_goals),
        )

        obs_dim = 6
        if goal_in_obs:
            obs_dim += self.goal_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(obs_dim,)
        )
        self.action_space = gym.spaces.Discrete(5)

        self._plotting = False
        self._fig, self._ax = None, None

    def _generate_new_maze(self):
        # generate a random maze
        core_maze = random_maze(self.maze_dim, self.maze_dim, seed=self.maze_seed)
        # set the bottom area empty
        top_spawn = self.maze_dim - 5
        bottom_spawn = self.maze_dim - 1
        core_maze[top_spawn:bottom_spawn, 1:-1] = 0
        core_maze[top_spawn + 1 : bottom_spawn, self.maze_dim // 2 - 1] = 1
        core_maze[top_spawn + 1 : bottom_spawn, self.maze_dim // 2 + 1] = 1
        return core_maze

    def goal_sequence(self):
        goals = copy.deepcopy(self.goal_positions)
        for i in range(self.active_goal_idx):
            goals[i] = (0, 0)

        while len(goals) < self.max_num_goals:
            goals.insert(0, (-1, -1))

        return np.array(goals).flatten().astype(np.float32)

    def reset(self, *args, **kwargs):
        # make a maze as an n x n array
        self.maze = self._generate_new_maze()
        self.start = (self.maze_dim - 2, self.maze_dim // 2)
        empty_locations = [x for x in zip(*np.where(self.maze == 0))]
        empty_locations.remove(self.start)
        if self.maze_seed is not None and not self.rand_goals:
            # deterministic goal resets
            rng = np.random.RandomState(self.maze_seed)
            num_goals = rng.randint(self.min_num_goals, self.max_num_goals + 1)
            self.goal_positions = [empty_locations[i] for i in 
                                   rng.choice(len(empty_locations), size=num_goals, replace=False)]
        else: 
            num_goals = random.randint(self.min_num_goals, self.max_num_goals)
            self.goal_positions = random.sample(empty_locations, k=num_goals)
        self.active_goal_idx = 0
        self.pos = self.start
        self._enforce_reset = False
        self._plotting = False
        self._goal_render_texts = [None for _ in range(num_goals)]
        return self._get_obs()

    def step(self, act):
        assert not self._enforce_reset, "Reset the environment with `env.reset()`"
        # 0 --> west, 1 --> north, 2 --> east, 3 --> south, 4 --> none
        dirs = [[0, -1], [-1, 0], [0, 1], [1, 0], [0, 0]]
        chosen_dir = np.array(dirs[act])
        desired_loc = tuple(self.pos + chosen_dir)

        valid = True
        if self.maze[desired_loc] != 0:
            valid = False
        else:
            for coord in desired_loc:
                if coord < 0 or coord >= self.maze_dim:
                    valid = False

        if valid:
            self.pos = desired_loc

        success = self.pos == self.goal_positions[self.active_goal_idx]
        terminated = False
        rew = float(success)
        obs = self._get_obs()
        
        if success:
            if self.active_goal_idx == len(self.goal_positions) - 1:
                if not self.reward_on_last_goal:
                    # only terminate if reward_on_last_goal is False, 
                    # otherwise reward is obtained on every step in last goal
                    terminated = True
                    self._enforce_reset = True
            else: 
                # Only give success, if all goals are reached!
                success = False
            if not (self.active_goal_idx == len(self.goal_positions) - 1 and self.reward_on_last_goal):
                # only increase if reward_on_last_goal is False, 
                # otherwise reward is obtained on every step in last goal
                self.active_goal_idx += 1

        return obs, rew, terminated, {"success": success, "is_success": success}

    def go_back_to_start(self):
        self.pos = self.start
        return self._get_obs()

    def _get_obs(self):
        i, j = tuple(self.pos)

        space_west = 0
        seek_west = j - 1
        while seek_west > 0:
            if self.maze[i, seek_west] == 0:
                seek_west -= 1
                space_west += 1
            else:
                break

        space_east = 0
        seek_east = j + 1
        while seek_east < self.maze_dim:
            if self.maze[i, seek_east] == 0:
                seek_east += 1
                space_east += 1
            else:
                break

        space_north = 0
        seek_north = i - 1
        while seek_north > 0:
            if self.maze[seek_north, j] == 0:
                seek_north -= 1
                space_north += 1
            else:
                break

        space_south = 0
        seek_south = i + 1
        while seek_south < self.maze_dim:
            if self.maze[seek_south, j] == 0:
                seek_south += 1
                space_south += 1
            else:
                break

        obs = np.array(
            [
                i / self.maze_dim,
                j / self.maze_dim,
                space_west / self.maze_dim,
                space_north / self.maze_dim,
                space_east / self.maze_dim,
                space_south / self.maze_dim,
            ],
            dtype=np.float32,
        )

        if self.goal_in_obs:
            obs = np.concatenate((obs, self.goal_sequence() / self.maze_dim))
        return obs

    def render(self, *args, **kwargs):
        if not self._plotting:
            self.start_plotting()
            plt.ion()
            self._plotting = True

        plt.tight_layout()
        background = np.ones((self.maze_dim, self.maze_dim, 3), dtype=np.uint8)
        maze_img = (
            background * abs(np.expand_dims(self.maze, -1) - 1) * 255
        )  # zero out (white) where there is a valid path
        goal_color_wheel = [
            [240, 3, 252],
            [255, 210, 87],
            [3, 219, 252],
            [252, 2157, 3],
        ]
        for i, goal_pos in enumerate(self.goal_positions):
            if self.active_goal_idx > i:
                continue
            x, y = goal_pos
            maze_img[x, y, :] = goal_color_wheel[i % len(goal_color_wheel)]

        maze_img[self.pos[0], self.pos[1], :] = [110, 110, 110]  # grey
        plt.imshow(maze_img)

        for i, goal_pos in enumerate(self.goal_positions):
            if self.active_goal_idx > i:
                continue
            y, x = goal_pos
            self._goal_render_texts[i] = plt.text(
                x, y, str(i), ha="center", va="center"
            )

        self._ax.set_title(
            f"k={self.goal_positions}, active_goal={self.goal_positions[self.active_goal_idx]}"
        )
        plt.draw()
        plt.pause(0.1)
        
        if self.render_mode == 'rgb_array':
            self._fig.canvas.draw()
            # Convert canvas to image array
            width, height = self._fig.canvas.get_width_height()
            img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
            return img

    def start_plotting(self):
        if self._fig:
            plt.close()
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        
        
def get_mazerunner_constructor(envid, maze_dim=15, maze_seed=None, env_kwargs=None, timelimit=400):
    env_kwargs = dict(env_kwargs) if env_kwargs is not None else {}
    render_mode = env_kwargs.pop("render_mode", None) 
    def make():
        env = MazeRunnerGymEnv(maze_seed=maze_seed, maze_dim=maze_dim, **env_kwargs)
        if timelimit is not None:
            env = TimeLimit(env, max_episode_steps=timelimit)
        env.name = f"{envid}_{'_' + str(maze_seed) if maze_seed is not None else ''}"
        if render_mode is not None: 
            env.metadata.update({"render.modes": [render_mode]})
        return Monitor(env)
    return make


def get_mazerunner_constructors(envid, maze_dim=15, maze_seed=None, timelimit=400, env_kwargs=None):
    # Case 1: None 
    # Case 2: single seed
    # Case 3: multiple seeds
    if maze_seed is None or isinstance(maze_seed, int): 
        maze_seed = [maze_seed]
    return [get_mazerunner_constructor(envid, maze_dim=maze_dim, maze_seed=seed, 
                                       timelimit=timelimit, env_kwargs=env_kwargs)
            for seed in maze_seed]


def make_mazerunner_envs(env_params, make_eval_env=True):
    const_kwargs = {
        "envid": env_params.envid,
        "env_kwargs": env_params.get("env_kwargs", {}),
        "maze_seed": env_params.get("train_maze_seed", None), 
        "maze_dim": env_params.get("maze_dim", 15),
        "timelimit": env_params.get("timelimit", 400),
    }
    env = DummyVecEnv(get_mazerunner_constructors(**const_kwargs))
    if make_eval_env:
        eval_maze_seed = env_params.get("eval_maze_seed", None)
        const_kwargs.update({"maze_seed": eval_maze_seed})
        eval_env = DummyVecEnv(get_mazerunner_constructors(**const_kwargs))
        eval_env.num_envs = 1
    env.num_envs = 1
    return env, eval_env


if __name__ == "__main__": 
    # env = MazeRunnerGymEnv(maze_seed=1, rand_goals=True)
    env = get_mazerunner_constructor("MazeRunner-v1", timelimit=10)()
    obs = env.reset()
    env.render()
    for i in range(100):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if done:
            env.reset()
        print(i, obs, rew, done, info)
        env.render()
        
    all_seeds = { 
        0:   1.0,
        1:   2.0,
        2:  0.01,
        3:  1.93,
        4:   1.9,
        5:  0.24,
        6:  0.98,
        7:   1.0,
        8:  0.89,
        9:   1.0,
        10:   0.0,
        11:   1.0,
        12:   1.0,
        13:   0.0,
        14:   1.0,
        15:   0.0,
        16:   2.0,
        17:  1.01,
        18:  1.88,
        19:  1.98,
        20:   3.0,
        21:   0.0,
        22:  0.61,
        23:  1.98,
        24:   0.0,
        25:   1.0,
        26:   2.0,
        27:   0.0,
        28:  0.38,
        29:   0.0,
        30:  1.96,
        31:  0.04,
        32:  0.66,
        33:  0.91,
        34:  0.95,
        35:   2.0,
        36:  0.01,
        37:   1.0,
        38:  0.48,
        39:  0.01,
        40:  0.93,
        41:   1.0,
        42:  0.63,
        43:   1.0,
        44:   1.0,
        45:  1.04,
        46:   1.0,
        47:   1.0,
        48:   1.0,
        49:  0.96,
        50:   1.0,
        51:  1.96,
        52:  0.09,
        53:  0.04,
        54:  0.78,
        55:  0.86,
        56:  0.95,
        57:  0.08,
        58:   1.0,
        59:   0.0,
        60:   0.6,
        61:   1.0,
        62:   3.0,
        63:   1.0,
        64:   1.0,
        65:  1.62,
        66:  0.33,
        67:   2.0,
        68:  2.05,
        69:  0.78,
        70:  1.21,
        71:  0.43,
        72:   1.0,
        73:  1.99,
        74:  0.45,
        75:   1.0,
        76:   1.0,
        77:   1.0,
        78:   0.0,
        79:  1.85,
        80:   1.0,
        81:   1.0,
        82:  0.41,
        83:  0.78,
        84:  1.02,
        85:  1.02,
        86:   1.0,
        87:  0.91,
        88:   1.0,
        89:  0.84,
        90:   2.0,
        91:  0.52,
        92:   1.0,
        93:   1.0,
        94:  0.03,
        95:   1.0,
        96:  0.91,
        97:  1.02,
        98:  1.35,
        99:   1.0,
        100:   1.0,
        101:   2.0,
        102:   1.0,
        103:  0.95,
        104:  0.92,
        105:   0.0,
        106:  1.96,
        107:  0.03,
        108:  0.98,
        109:  0.95,
        110:  0.96,
        111:  0.36,
        112:   1.0,
        113:  0.96,
        114:  0.07,
        115:   0.0,
        116:  1.94,
        117:  1.76,
        118:   1.0,
        119:   3.0,
        120:   1.0,
        121:  1.21,
        122:  0.92,
        123:  0.62,
        124:  0.94,
        125:  0.72,
        126:  1.85,
        127:  1.47,
        128:  1.89,
        129:   1.0,
        130:  1.93,
        131:  0.86,
        132:  1.34,
        133:   1.3,
        134:  1.61,
        135:  0.01,
        136:   2.0,
        137:  0.03,
        138:   1.0,
        139:   1.0,
        140:   1.0,
        141:  2.92,
        142:  0.22,
        143:  1.56,
        144:   1.0,
        145:  1.17,
        146:  0.89,
        147:  0.99,
        148:  0.84,
        149:   2.0,
        150:   1.0,
        151:   1.0,
        152:   0.0,
        153:   1.0,
        154:  1.88,
        155:   1.0,
        156:   1.0,
        157:   1.0,
        158:   2.0,
        159:   1.0,
        160:   0.0,
        161:  0.83,
        162:  0.92,
        163:   2.0,
        164:   1.0,
        165:  0.08,
        166:   1.0,
        167:   2.0,
        168:  0.87,
        169:  1.79,
        170:  0.52,
        171:   1.0,
        172:  0.93,
        173:   0.0,
        174:  0.94,
        175:   1.0,
        176:   1.0,
        177:  0.89,
        178:  0.76,
        179:  0.62,
        180:   0.0,
        181:   0.0,
        182:  0.89,
        183:  0.49,
        184:   1.1,
        185:  0.52,
        186:  0.97,
        187:  0.45,
        188:   0.0,
        189:  0.14,
        190:  1.98,
        191:   0.5,
        192:  0.61,
        193:   1.6,
        194:  1.68,
        195:  0.99,
        196:  0.99,
        197:  0.82,
        198:   0.0,
        199:  0.01,
        200:   3.0,
        201:   1.0,
        202:   1.0,
        203:   1.0,
        204:   2.0,
        205:  0.18,
        206:   2.0,
        207:  0.51,
        208:  1.24,
        209:  2.98,
        210:  2.13,
        211:  0.99,
        212:   1.0,
        213:  0.99,
        214:   0.0,
        215:  1.76,
        216:   1.0,
        217:   0.0,
        218:  0.97,
        219:  0.97,
        220:  0.95,
        221:   1.0,
        222:  0.01,
        223:   1.0,
        224:   1.0,
        225:   1.0,
        226:  0.99,
        227:   1.0,
        228:   0.0,
        229:  0.59,
        230:   1.0,
        231:   1.0,
        232:   2.0,
        233:   2.0,
        234:   1.0,
        235:  0.96,
        236:  0.72,
        237:   1.0,
        238:   0.0,
        239:   1.0,
        240:   0.0,
        241:  0.03,
        242:  1.89,
        243:   2.0,
        244:   0.0,
        245:  0.04,
        246:   0.0,
        247:  0.94,
        248:  1.06,
        249:   1.0,
        250:   0.0,
        251:  1.88,
        252:  0.92,
        253:  0.58,
        254:   0.0,
        255:   2.0,
        256:   1.0,
        257:  1.59,
        258:  0.71,
        259:   1.0,
        260:  1.72,
        261:   1.0,
        262:  0.79,
        263:  1.69,
        264:  0.71,
        265:  0.04,
        266:  1.14,
        267:   2.0,
        268:   2.0,
        269:   1.0,
        270:   1.0,
        271:   1.9,
        272:   1.0,
        273:  1.72,
        274:  0.02,
        275:  0.97,
        276:  1.01,
        277:  0.72,
        278:   1.0,
        279:   2.0,
        280:  0.97,
        281:   1.0,
        282:   1.0,
        283:   1.0,
        284:   1.0,
        285:   2.0,
        286:  0.27,
        287:  1.56,
        288:   3.0,
        289:   1.0,
        290:  0.93,
        291:   2.0,
        292:   1.0,
        293:  1.77,
        294:   0.0,
        295:   1.0,
        296:   1.0,
        297:  0.96,
        298:   1.0,
        299:  1.87,
    }
    
    postitives = {k:v for k, v in all_seeds.items() if v > 0.1}
    print(len(postitives))
    
    reward_threshold = 0.1
    max_seeds = 100
    first_100, first_100_v = [], []
    for k, v in all_seeds.items(): 
        if v > 0.25: 
            first_100.append(k)
            first_100_v.append(v)
        if len(first_100) >= max_seeds: 
            break
    print(np.mean(first_100_v), np.std(first_100_v))
        
    for s in first_100: 
        print("-", s)    
    print("\n")
    for i in range(300, 320):
        print("-", i)
