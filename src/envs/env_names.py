# MT40 envs without tasks contained in CW10
MT40_ENVS = ['reach-v1', 'pick-place-v1', 'door-open-v1', 'drawer-open-v1', 'drawer-close-v1', 'button-press-topdown-v1',
             'peg-insert-side-v1', 'window-open-v1', 'door-close-v1', 'reach-wall-v1', 'pick-place-wall-v1',
             'button-press-v1', 'button-press-topdown-wall-v1', 'button-press-wall-v1', 'disassemble-v1', 'plate-slide-v1',
             'plate-slide-side-v1', 'plate-slide-back-v1', 'plate-slide-back-side-v1', 'handle-press-v1', 'handle-pull-v1',
             'handle-pull-side-v1', 'stick-push-v1', 'basketball-v1', 'soccer-v1', 'faucet-open-v1', 'coffee-push-v1',
             'coffee-pull-v1', 'coffee-button-v1', 'sweep-v1', 'sweep-into-v1', 'pick-out-of-hole-v1', 'assembly-v1',
             'lever-pull-v1', 'dial-turn-v1', 'bin-picking-v1', 'box-close-v1', 'hand-insert-v1', 'door-lock-v1',
             'door-unlock-v1']

# requires: pip install git+https://github.com/rlworkgroup/metaworld.git@0875192baaa91c43523708f55866d98eaf3facaf
MT50_ENVS = ['reach-v1', 'push-v1', 'pick-place-v1', 'door-open-v1', 'drawer-open-v1', 'drawer-close-v1',
             'button-press-topdown-v1', 'peg-insert-side-v1', 'window-open-v1', 'window-close-v1', 'door-close-v1',
             'reach-wall-v1', 'pick-place-wall-v1', 'push-wall-v1', 'button-press-v1', 'button-press-topdown-wall-v1',
             'button-press-wall-v1', 'peg-unplug-side-v1', 'disassemble-v1', 'hammer-v1', 'plate-slide-v1',
             'plate-slide-side-v1', 'plate-slide-back-v1', 'plate-slide-back-side-v1', 'handle-press-v1',
             'handle-pull-v1', 'handle-press-side-v1', 'handle-pull-side-v1', 'stick-push-v1', 'stick-pull-v1',
             'basketball-v1', 'soccer-v1', 'faucet-open-v1', 'faucet-close-v1', 'coffee-push-v1', 'coffee-pull-v1',
             'coffee-button-v1', 'sweep-v1', 'sweep-into-v1', 'pick-out-of-hole-v1', 'assembly-v1', 'shelf-place-v1',
             'push-back-v1', 'lever-pull-v1', 'dial-turn-v1', 'bin-picking-v1', 'box-close-v1', 'hand-insert-v1',
             'door-lock-v1', 'door-unlock-v1']

# v1 envs attain much lower success scores: https://github.com/rlworkgroup/metaworld/pull/154#discussion_r459096316
# CW was developed for v1 envs, however.
# requires: pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
# requires: pip install git+https://github.com/rlworkgroup/metaworld.git@18118a28c06893da0f363786696cc792457b062b
MT40_ENVS_v2 = ['reach-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2', 'button-press-topdown-v2',
                'peg-insert-side-v2', 'window-open-v2', 'door-close-v2', 'reach-wall-v2', 'pick-place-wall-v2',
                'button-press-v2', 'button-press-topdown-wall-v2', 'button-press-wall-v2', 'disassemble-v2', 'plate-slide-v2',
                'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'handle-press-v2', 'handle-pull-v2',
                'handle-pull-side-v2', 'stick-push-v2', 'basketball-v2', 'soccer-v2', 'faucet-open-v2', 'coffee-push-v2',
                'coffee-pull-v2', 'coffee-button-v2', 'sweep-v2', 'sweep-into-v2', 'pick-out-of-hole-v2', 'assembly-v2',
                'lever-pull-v2', 'dial-turn-v2', 'bin-picking-v2', 'box-close-v2', 'hand-insert-v2', 'door-lock-v2',
                'door-unlock-v2']

MT50_ENVS_v2 = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2',
                'button-press-topdown-v2', 'peg-insert-side-v2', 'window-open-v2', 'window-close-v2', 'door-close-v2',
                'reach-wall-v2', 'pick-place-wall-v2', 'push-wall-v2', 'button-press-v2', 'button-press-topdown-wall-v2',
                'button-press-wall-v2', 'peg-unplug-side-v2', 'disassemble-v2', 'hammer-v2', 'plate-slide-v2',
                'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'handle-press-v2',
                'handle-pull-v2', 'handle-press-side-v2', 'handle-pull-side-v2', 'stick-push-v2', 'stick-pull-v2',
                'basketball-v2', 'soccer-v2', 'faucet-open-v2', 'faucet-close-v2', 'coffee-push-v2', 'coffee-pull-v2',
                'coffee-button-v2', 'sweep-v2', 'sweep-into-v2', 'pick-out-of-hole-v2', 'assembly-v2', 'shelf-place-v2',
                'push-back-v2', 'lever-pull-v2', 'dial-turn-v2', 'bin-picking-v2', 'box-close-v2', 'hand-insert-v2',
                'door-lock-v2', 'door-unlock-v2']

MT45_ENVS_v2 = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2',
                'button-press-topdown-v2', 'peg-insert-side-v2', 'window-open-v2', 'window-close-v2',
                'door-close-v2', 'reach-wall-v2', 'pick-place-wall-v2', 'push-wall-v2', 'button-press-v2',
                'button-press-topdown-wall-v2', 'button-press-wall-v2', 'peg-unplug-side-v2', 'disassemble-v2', 
                'hammer-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 
                'plate-slide-back-side-v2', 'handle-press-v2', 'handle-pull-v2', 'handle-press-side-v2',
                'handle-pull-side-v2', 'stick-push-v2', 'stick-pull-v2', 'basketball-v2', 'soccer-v2',
                'faucet-open-v2', 'faucet-close-v2', 'coffee-push-v2', 'coffee-pull-v2', 'coffee-button-v2',
                'sweep-v2', 'sweep-into-v2', 'pick-out-of-hole-v2', 'assembly-v2', 'shelf-place-v2',
                'push-back-v2', 'lever-pull-v2', 'dial-turn-v2']

MT45_EASY_ENVS_v2 = ['reach-v2', 'push-v2', 'pick-place-v2', 'drawer-close-v2', 'button-press-topdown-v2', 
                     'peg-insert-side-v2', 'window-open-v2', 'door-close-v2', 'reach-wall-v2', 'pick-place-wall-v2', 
                     'push-wall-v2', 'button-press-v2', 'button-press-topdown-wall-v2', 'button-press-wall-v2',
                     'peg-unplug-side-v2', 'disassemble-v2', 'hammer-v2', 'plate-slide-v2', 'plate-slide-side-v2',
                     'plate-slide-back-v2', 'plate-slide-back-side-v2', 'handle-press-v2', 'handle-pull-v2',
                     'handle-press-side-v2', 'handle-pull-side-v2', 'stick-push-v2', 'stick-pull-v2', 'basketball-v2',
                     'soccer-v2', 'faucet-open-v2', 'coffee-push-v2', 'coffee-pull-v2', 'coffee-button-v2', 'sweep-v2',
                     'sweep-into-v2', 'pick-out-of-hole-v2', 'assembly-v2', 'shelf-place-v2', 'push-back-v2',
                     'lever-pull-v2', 'dial-turn-v2', 'bin-picking-v2', 'hand-insert-v2', 
                     'door-lock-v2', 'door-unlock-v2']

MT5_ENVS_v2 = ['bin-picking-v2', 'box-close-v2', 'hand-insert-v2', 'door-lock-v2', 'door-unlock-v2']

MT5_EASY_ENVS_v2 = ['door-open-v2', 'drawer-open-v2', 'window-close-v2', 'faucet-close-v2', 'box-close-v2']

MT5_SMALL_ENVS_v2 = ['door-close-v2', 'drawer-open-v2', 'window-close-v2', 'faucet-open-v2', 'box-close-v2']

MT5_SMALL_EVAL_ENVS_v2 = ['door-open-v2', 'drawer-close-v2', 'window-open-v2', 'faucet-close-v2', 'door-lock-v2']

ML10_ENVS_V2 = ["basketball-v2", 'button-press-v2', "dial-turn-v2", "drawer-close-v2", 
                'peg-insert-side-v2', 'pick-place-v2', "push-v2", "reach-v2", "sweep-into-v2", "window-open-v2"]

ML_5_ENVS_V2 = ["door-close-v2", "drawer-open-v2", "lever-pull-v2", "shelf-place-v2", "sweep-v2"]

CW10_ENVS_V1 = ["hammer-v1", "push-wall-v1", "faucet-close-v1", "push-back-v1", "stick-pull-v1", "handle-press-side-v1",
                "push-v1", "shelf-place-v1", "window-close-v1", "peg-unplug-side-v1"]

CW10_ENVS_V2 = ["hammer-v2", "push-wall-v2", "faucet-close-v2", "push-back-v2", "stick-pull-v2", "handle-press-side-v2",
                "push-v2", "shelf-place-v2", "window-close-v2", "peg-unplug-side-v2"]

TASK_SEQS = {
    "cw10": [
        "hammer-v1",
        "push-wall-v1",
        "faucet-close-v1",
        "push-back-v1",
        "stick-pull-v1",
        "handle-press-side-v1",
        "push-v1",
        "shelf-place-v1",
        "window-close-v1",
        "peg-unplug-side-v1",
    ],
    "cw10_v2": [
        "hammer-v2",
        "push-wall-v2",
        "faucet-close-v2",
        "push-back-v2",
        "stick-pull-v2",
        "handle-press-side-v2",
        "push-v2",
        "shelf-place-v2",
        "window-close-v2",
        "peg-unplug-side-v2",
    ]
}

TASK_SEQS["cw20"] = TASK_SEQS["cw10"] + TASK_SEQS["cw10"]
TASK_SEQS["cw20_v2"] = TASK_SEQS["cw10_v2"] + TASK_SEQS["cw10_v2"]


PROCGEN_ENVS = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist",
                "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]

PROCGEN_12_ENVS = ["bigfish", "bossfight", "caveflyer", "chaser", "coinrun", "dodgeball", 
                   "fruitbot", "heist", "leaper", "maze", "miner", "starpilot"]

PROCGEN_4_ENVS = ["climber", "ninja", "plunder", "jumper"]


ATARI_NAMES = ['adventure', 'air-raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
              'bank-heist', 'battle-zone', 'beam-rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
              'centipede', 'chopper-command', 'crazy-climber', 'defender', 'demon-attack', 'double-dunk',
              'elevator-action', 'enduro', 'fishing-derby', 'freeway', 'frostbite', 'gopher', 'gravitar', 'hero',
              'ice-hockey', 'jamesbond', 'journey-escape', 'kangaroo', 'krull', 'kung-fu-master',
              'montezuma-revenge', 'ms-pacman', 'name-this-game', 'phoenix', 'pitfall', 'pong', 'pooyan',
              'private-eye', 'qbert', 'riverraid', 'road-runner', 'robotank', 'seaquest', 'skiing', 'solaris',
              'space-invaders', 'star-gunner', 'tennis', 'time-pilot', 'tutankham', 'up-n-down', 'venture',
              'video-pinball', 'wizard-of-wor', 'yars-revenge', 'zaxxon']

ATARI_ENVS = ['AdventureNoFrameskip-v4', 'AirRaidNoFrameskip-v4', 'AlienNoFrameskip-v4', 'AmidarNoFrameskip-v4',
              'AssaultNoFrameskip-v4','AsterixNoFrameskip-v4', 'AsteroidsNoFrameskip-v4', 'AtlantisNoFrameskip-v4',
              'BankHeistNoFrameskip-v4', 'BattleZoneNoFrameskip-v4', 'BeamRiderNoFrameskip-v4', 'BerzerkNoFrameskip-v4',
              'BowlingNoFrameskip-v4', 'BoxingNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'CarnivalNoFrameskip-v4',
              'CentipedeNoFrameskip-v4', 'ChopperCommandNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4', 
              'DefenderNoFrameskip-v4', 'DemonAttackNoFrameskip-v4', 'DoubleDunkNoFrameskip-v4', 'ElevatorActionNoFrameskip-v4',
              'EnduroNoFrameskip-v4', 'FishingDerbyNoFrameskip-v4', 'FreewayNoFrameskip-v4', 'FrostbiteNoFrameskip-v4',
              'GopherNoFrameskip-v4', 'GravitarNoFrameskip-v4', 'HeroNoFrameskip-v4', 'IceHockeyNoFrameskip-v4', 
              'JamesbondNoFrameskip-v4', 'JourneyEscapeNoFrameskip-v4', 'KangarooNoFrameskip-v4', 'KrullNoFrameskip-v4', 
              'KungFuMasterNoFrameskip-v4', 'MontezumaRevengeNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 
              'NameThisGameNoFrameskip-v4', 'PhoenixNoFrameskip-v4', 'PitfallNoFrameskip-v4', 'PongNoFrameskip-v4', 
              'PooyanNoFrameskip-v4', 'PrivateEyeNoFrameskip-v4', 'QbertNoFrameskip-v4', 'RiverraidNoFrameskip-v4', 
              'RoadRunnerNoFrameskip-v4', 'RobotankNoFrameskip-v4', 'SeaquestNoFrameskip-v4', 'SkiingNoFrameskip-v4',
              'SolarisNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'StarGunnerNoFrameskip-v4', 'TennisNoFrameskip-v4',
              'TimePilotNoFrameskip-v4', 'TutankhamNoFrameskip-v4', 'UpNDownNoFrameskip-v4', 'VentureNoFrameskip-v4',
              'VideoPinballNoFrameskip-v4', 'WizardOfWorNoFrameskip-v4', 'YarsRevengeNoFrameskip-v4', 'ZaxxonNoFrameskip-v4']


ATARI_NAME_TO_ENVID = {name: [env for env in ATARI_ENVS if name.replace("-", "") in env.lower()][0] for name in ATARI_NAMES}

ATARI_ENVID_TO_NAME = {envid: name for name, envid in ATARI_NAME_TO_ENVID.items()}

# https://www.gymlibrary.dev/environments/atari/
# v0 --> sticky actions, v4 --> no sticky actions
# in MGDT they do not use sticky actions
ATARI_NAMES_10 =["pong", "breakout", "asterix", "qbert", "alien", "beam-rider", 
                 "freeway", "ms-pacman", "space-invaders", "seaquest"]
ATARI_ENVS_10 = [ATARI_NAME_TO_ENVID[name] for name in ATARI_NAMES_10]

ATARI_NAMES_46 = ['alien', 'amidar', 'assault', 'asterix', 'atlantis', 'bank-heist', 'battle-zone',
                  'beam-rider', 'boxing', 'breakout', 'carnival', 'centipede', 'chopper-command', 'crazy-climber',
                  'demon-attack', 'double-dunk', 'enduro', 'fishing-derby', 'freeway', 'frostbite', 'gopher',
                  'gravitar', 'hero', 'ice-hockey', 'jamesbond', 'kangaroo', 'krull', 'kung-fu-master', 'ms-pacman', 
                  'name-this-game', 'phoenix', 'pong', 'pooyan', 'qbert', 'riverraid', 'road-runner', 'robotank',
                  'seaquest', 'space-invaders', 'star-gunner','time-pilot','up-n-down', 'video-pinball', 
                  'wizard-of-wor', 'yars-revenge', 'zaxxon']
ATARI_ENVS_46 = [ATARI_NAME_TO_ENVID[name] for name in ATARI_NAMES_46]

# does not contain: alien, pong, ms-pacman, space-invaders, star-gunner
ATARI_NAMES_41 = ['amidar', 'assault', 'asterix', 'atlantis', 'bank-heist', 'battle-zone',
                  'beam-rider', 'boxing', 'breakout', 'carnival', 'centipede', 'chopper-command', 'crazy-climber',
                  'demon-attack', 'double-dunk', 'enduro', 'fishing-derby', 'freeway', 'frostbite', 'gopher',
                  'gravitar', 'hero', 'ice-hockey', 'jamesbond', 'kangaroo', 'krull', 'kung-fu-master', 
                  'name-this-game', 'phoenix', 'pooyan', 'qbert', 'riverraid', 'road-runner', 'robotank',
                  'seaquest','time-pilot','up-n-down', 'video-pinball', 
                  'wizard-of-wor', 'yars-revenge', 'zaxxon']
ATARI_ENVS_41 = [ATARI_NAME_TO_ENVID[name] for name in ATARI_NAMES_41]

ATARI_NAMES_5 = ['alien', 'pong', 'ms-pacman', 'space-invaders', 'star-gunner']
ATARI_ENVS_5 = [ATARI_NAME_TO_ENVID[name] for name in ATARI_NAMES_5]


DM_CONTROL_ENVS = ['acrobot-swingup', 'acrobot-swingup_sparse', 'ball_in_cup-catch', 'cartpole-balance',
                   'cartpole-balance_sparse', 'cartpole-swingup', 'cartpole-swingup_sparse',
                   'cheetah-run', 'finger-spin', 'finger-turn_easy', 'finger-turn_hard', 
                   'fish-upright', 'fish-swim', 'hopper-stand', 'hopper-hop', 'humanoid-stand',
                   'humanoid-walk', 'humanoid-run', 'manipulator-bring_ball', 'pendulum-swingup',
                   'point_mass-easy', 'reacher-easy', 'reacher-hard', 'swimmer-swimmer6', 
                   'swimmer-swimmer15', 'walker-stand', 'walker-walk', 'walker-run',
                   "manipulator-insert_ball", "manipulator-insert_peg"]

DM_CONTROL_ENVS_17 = ['acrobot-swingup',  'ball_in_cup-catch', 'cartpole-balance', 'cartpole-swingup', 
                      'cheetah-run', 'finger-spin', 'finger-turn_easy', 'finger-turn_hard', 
                      'fish-upright', 'hopper-stand', 'pendulum-swingup', 'point_mass-easy', 'reacher-easy',
                      'reacher-hard', 'walker-stand', 'walker-walk', 'walker-run']

DM_CONTROL_ENVS_6 = ['ball_in_cup-catch', 'cartpole-swingup', 'cheetah-run',
                     'finger-spin', 'reacher-easy', 'walker-walk']

DM_CONTROL_ENVS_10 = ['cartpole-balance', 'finger-turn_easy', 'finger-turn_hard', 
                      'fish-upright', 'hopper-stand', 'pendulum-swingup', 'point_mass-easy', 'reacher-hard',
                      'walker-stand', 'walker-run']

DM_CONTROL_ENVS_6_v2 = ['ball_in_cup-catch', 'cartpole-swingup', 'cheetah-run',
                        'finger-spin', 'reacher-easy', 'walker-walk']

DM_CONTROL_ICL_ENVS_11 = ['finger-turn_easy', 'fish-upright', 'hopper-stand',
                          'point_mass-easy', 'walker-stand', 'walker-run', 'ball_in_cup-catch',
                          'cartpole-swingup', 'cheetah-run', 'finger-spin', 'reacher-easy']

DM_CONTROL_ICL_ENVS_5 = ['cartpole-balance', 'finger-turn_hard', "pendulum-swingup",
                         'reacher-hard', 'walker-walk']

DM_CONTROL_ICL_ENVS_6 = ['ball_in_cup-catch', 'cartpole-swingup', 'finger-spin',
                         'finger-turn_easy',  'reacher-easy', 'walker-stand']

DM_CONTROL_ICL_ENVS_4 = ['cartpole-balance', 'finger-turn_easy', "pendulum-swingup", 'reacher-easy']

DM_CONTROL_ENVS_VISUAL= ["cheetah-run", "humanoid-walk", "walker-walk"]


DM_CONTROL_ENVID_TO_FILE = {f"{envid}.npz": f"{envid.replace('-', '_')}.npz" for envid in DM_CONTROL_ENVS}

DM_CONTROL_ENVID_TO_DIR = {envid: f"{envid.replace('-', '_')}" for envid in DM_CONTROL_ENVS}

MINIHACK_ENVS = ["MiniHack-Room-Dark-10x10-v0", "MiniHack-Room-Dark-17x17-v0",
                 "MiniHack-Room-Dark-Dense-10x10-v0", "MiniHack-Room-Dark-Dense-17x17-v0",
                 "MiniHack-Room-Dark-Sparse-10x10-v0", "MiniHack-Room-Dark-Sparse-17x17-v0",
                 "MiniHack-Room-Dark-Dense-20x20-v0", "MiniHack-Room-Dark-Dense-40x20-v0", 
                 "MiniHack-KeyDoor-Dark-Dense-10x10-v0", "MiniHack-KeyDoor-Dark-Dense-20x20-v0",
                 "MiniHack-KeyDoor-Dark-Dense-40x20-v0"]

GYM_ENVS = ["Hopper-v3", "HalfCheetah-v3", "Walker2d-v3", "Hopper-v2", "HalfCheetah-v2", "Walker2d-v2"]

MAZERUNNER_ENVS = ["MazeRunner-15x15-v0", "MazeRunner-30x30-v0"]

CRAFTER_ENVS = ["Crafter-v0"]


ID_TO_NAMES = {
    "mt40": MT40_ENVS,
    "mt50": MT50_ENVS,
    "mt50_v2": MT50_ENVS_v2,
    "mt40_v2": MT40_ENVS_v2,
    "mt45_v2": MT45_ENVS_v2,
    "mt5_v2": MT5_ENVS_v2,
    "mt45_easy_v2": MT45_EASY_ENVS_v2,
    "mt5_easy_v2": MT5_EASY_ENVS_v2,
    "mt5_small_v2": MT5_SMALL_ENVS_v2,
    "mt5_small_eval_v2": MT5_SMALL_EVAL_ENVS_v2,
    "ml10_v2": ML10_ENVS_V2,
    "ml5_v2": ML_5_ENVS_V2,
    "cw10": CW10_ENVS_V1, 
    "cw10_v2": CW10_ENVS_V2,
    "atari10": ATARI_ENVS_10,
    "atari46": ATARI_ENVS_46,
    "atari41": ATARI_ENVS_41, 
    "atari46_mt40v2": [*ATARI_ENVS_46, *MT40_ENVS_v2],
    "atari41_mt40v2": [*ATARI_ENVS_41, *MT40_ENVS_v2], 
    "dmcontrol": DM_CONTROL_ENVS,
    "dmcontrol6": DM_CONTROL_ENVS_6,
    "dmcontrol6_v2": DM_CONTROL_ENVS_6_v2,
    "dmcontrol6_icl": DM_CONTROL_ICL_ENVS_6,
    "dmcontrol4_icl": DM_CONTROL_ICL_ENVS_4,
    "dmcontrol10": DM_CONTROL_ENVS_10,
    "dmcontrol17": DM_CONTROL_ENVS_17,
    "dmcontrol11_icl": DM_CONTROL_ICL_ENVS_11,
    "dmcontrol5_icl": DM_CONTROL_ICL_ENVS_5,
    "dmcontrol_visual": DM_CONTROL_ENVS_VISUAL,
    "atari5": ATARI_ENVS_5,
    "cw10v2_atari5": [*CW10_ENVS_V2, *ATARI_ENVS_5], 
    "mt40v2_dmc10": [*MT40_ENVS_v2, *DM_CONTROL_ENVS_10], 
    "cw10v2_dmc6": [*CW10_ENVS_V2, *DM_CONTROL_ENVS_6_v2], 
    "atari41_mt40v2_dmc10": [*ATARI_ENVS_41, *MT40_ENVS_v2, *DM_CONTROL_ENVS_10], 
    "cw10v2_atari5_dmc6": [*CW10_ENVS_V2, *ATARI_ENVS_5, *DM_CONTROL_ENVS_6_v2],
    "procgen16": PROCGEN_ENVS,
    "procgen12": PROCGEN_12_ENVS,
    "procgen4": PROCGEN_4_ENVS,
}


ID_TO_DOMAIN = {
    **{envid: "dmcontrol" for envid in [*DM_CONTROL_ENVS, "dmcontrol11_icl", "dmcontrol5_icl", "dmcontrol6_icl", "dmcontrol4_icl"]},
    **{envid: "atari" for envid in [*ATARI_ENVS_46, *ATARI_NAMES]},
    **{envid: "procgen" for envid in [*PROCGEN_ENVS, "procgen16", "procgen12", "procgen4"]},
    **{envid: "mt50" for envid in [*MT50_ENVS, *MT50_ENVS_v2,
                                   "mt40", "mt50", "mt40_v2", "mt50_v2", 
                                   "mt45_v2", "mt5_v2", "mt45_easy_v2", "mt5_easy_v2",
                                   "ml10_v2", "ml5_v2", "mt5_small_v2", "mt5_small_eval_v2"]},
    **{envid: "cw10" for envid in ["cw10", "cw20", "cw10_v2", "cw20_v2"]},
    **{envid: "crafter" for envid in [*CRAFTER_ENVS]},
    **{envid: "mazerunner" for envid in [*MAZERUNNER_ENVS]},

    
}

ENVID_TO_NAME = {**ATARI_ENVID_TO_NAME, **DM_CONTROL_ENVID_TO_FILE, **DM_CONTROL_ENVID_TO_DIR}

NAME_TO_ENVID = {v:k for k, v in ENVID_TO_NAME.items()}
