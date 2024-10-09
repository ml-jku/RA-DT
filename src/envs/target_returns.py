# max return in respective dataset * 2
MT50_targets = {
    'assembly-v1': 1450286.625, 'button-press-v1': 566043.703125, 'disassemble-v1': -9.405543645222982,
    'plate-slide-side-v1': 374616.09375, 'door-lock-v1': 544039.125, 'door-unlock-v1': 501917.859375,
    'pick-place-v1': 477427.6875, 'drawer-close-v1': 521100.375, 'coffee-pull-v1': 520720.3125,
    'handle-pull-v1': 493718.203125, 'peg-insert-side-v1': 307802.578125, 'door-close-v1': 578117.109375,
    'button-press-topdown-v1': 522984.65625, 'lever-pull-v1': -1.220013936360677, 'reach-v1': 638514.28125,
    'door-open-v1': 501558.234375, 'sweep-v1': 122089.7578125, 'box-close-v1': 589573.546875,
    'button-press-wall-v1': 537146.484375, 'basketball-v1': 500070.1875, 'soccer-v1': 514502.15625,
    'handle-press-v1': 493111.78125, 'coffee-button-v1': 552852.75, 'faucet-open-v1': 154400.8125,
    'plate-slide-v1': 339928.546875, 'reach-wall-v1': 626686.5, 'handle-pull-side-v1': 453395.90625,
    'drawer-open-v1': 488456.0625, 'dial-turn-v1': 450512.625, 'sweep-into-v1': 536743.265625,
    'bin-picking-v1': 687246.1875, 'coffee-push-v1': 534557.34375, 'pick-out-of-hole-v1': -11.327592213948568,
    'button-press-topdown-wall-v1': 522397.078125, 'plate-slide-back-side-v1': 530872.96875,
    'plate-slide-back-v1': 532280.8125, 'stick-push-v1': 178049.05078125, 'hand-insert-v1': 585300.9375,
    'window-open-v1': 356100.890625, 'pick-place-wall-v1': 506045.0625, 'hammer-v1': 324996.609375,
    'peg-unplug-side-v1': 549168.65625, 'window-close-v1': 293120.0390625, 'faucet-close-v1': 333695.90625,
    'handle-press-side-v1': 472267.875, 'push-back-v1': 519344.109375, 'push-v1': 539754.421875,
    'push-wall-v1': 522941.578125, 'stick-pull-v1': 1216821.9375, 'shelf-place-v1': 319610.765625
}

MT50_targets_v2 = {
    'assembly-v2': 1285.642, 'button-press-v2': 1604.5227, 'disassemble-v2': 1536.4567,
    'plate-slide-side-v2': 1748.304, 'door-lock-v2': 1831.5373, 'door-unlock-v2': 1794.7213,
    'pick-place-v2': 1300.692, 'drawer-close-v2': 1880.0, 'coffee-pull-v2': 1475.0727,
    'handle-pull-v2': 1759.07, 'peg-insert-side-v2': 1695.9867, 'door-close-v2': 1587.8,
    'button-press-topdown-v2': 1384.7, 'lever-pull-v2': 1677.1973, 'reach-v2': 1905.1067,
    'door-open-v2': 1608.812, 'sweep-v2': 1560.82, 'box-close-v2': 1127.9013,
    'button-press-wall-v2': 1614.572, 'basketball-v2': 1597.018, 'soccer-v2': 1706.2247,
    'handle-press-v2': 1929.8347, 'coffee-button-v2': 1698.32, 'faucet-open-v2': 1783.7487,
    'plate-slide-v2': 1713.0807, 'reach-wall-v2': 1863.1053, 'handle-pull-side-v2': 1710.8367,
    'drawer-open-v2': 1762.81, 'dial-turn-v2': 1887.328, 'sweep-into-v2': 1815.9427,
    'bin-picking-v2': 1308.598, 'coffee-push-v2': 1694.6847, 'pick-out-of-hole-v2': 1588.1173,
    'button-press-topdown-wall-v2': 1388.598, 'plate-slide-back-side-v2': 1833.7033,
    'plate-slide-back-v2': 1806.1567, 'stick-push-v2': 1629.2633, 'hand-insert-v2': 1789.0467,
    'window-open-v2': 1717.4207, 'pick-place-wall-v2': 1654.644, 'hammer-v2': 1724.842,
    'peg-unplug-side-v2': 1619.82, 'window-close-v2': 1559.3093, 'faucet-close-v2': 1798.588,
    'handle-press-side-v2': 1897.76, 'push-back-v2': 1564.7233, 'push-v2': 1791.2247,
    'push-wall-v2': 1746.7047, 'stick-pull-v2': 1551.368, 'shelf-place-v2': 1513.3307
}

MT50_v2_500steps_targets = {
    'assembly-v2_500': 4557.758, 'button-press-v2_500': 3718.285, 'disassemble-v2_500': 4557.419,
    'plate-slide-side-v2_500': 4734.194, 'door-lock-v2_500': 4715.318, 'door-unlock-v2_500': 4739.971,
    'pick-place-v2_500': 2967.195, 'drawer-close-v2_500': 4880.49, 'coffee-pull-v2_500': 515.462, 
    'handle-pull-v2_500': 4771.349, 'peg-insert-side-v2_500': 4690.355, 'door-close-v2_500': 4592.562, 
    'button-press-topdown-v2_500': 3925.996, 'lever-pull-v2_500': 4215.64, 'reach-v2_500': 4899.213, 
    'door-open-v2_500': 4623.019, 'sweep-v2_500': 4578.873, 'box-close-v2_500': 4502.542, 
    'button-press-wall-v2_500': 4350.583, 'basketball-v2_500': 393.772, 'soccer-v2_500': 3463.242, 
    'handle-press-v2_500': 4928.883, 'coffee-button-v2_500': 4492.226, 'faucet-open-v2_500': 4776.849, 
    'plate-slide-v2_500': 4697.033, 'reach-wall-v2_500': 4856.4, 'handle-pull-side-v2_500': 4700.831,
    'drawer-open-v2_500': 4791.439, 'dial-turn-v2_500': 4750.286, 'sweep-into-v2_500': 4763.013, 
    'bin-picking-v2_500': 1199.102, 'coffee-push-v2_500': 4525.045, 'pick-out-of-hole-v2_500': 565.242, 
    'button-press-topdown-wall-v2_500': 3920.311, 'plate-slide-back-side-v2_500': 4834.989,
    'plate-slide-back-v2_500': 4790.534, 'stick-push-v2_500': 4557.222, 'hand-insert-v2_500': 4792.846, 
    'window-open-v2_500': 4550.688, 'pick-place-wall-v2_500': 78.686, 'hammer-v2_500': 4702.326, 
    'peg-unplug-side-v2_500': 4701.794, 'window-close-v2_500': 4643.951, 'faucet-close-v2_500': 4785.459, 
    'handle-press-side-v2_500': 4893.379, 'push-back-v2_500': 4578.973, 'push-v2_500': 4700.407,
    'push-wall-v2_500': 4716.214, 'stick-pull-v2_500': 4519.777, 'shelf-place-v2_500': 1137.201
}


ATARI_targets = {
    'AlienNoFrameskip-v4': 218.0, 'AmidarNoFrameskip-v4': 204.0, 'AssaultNoFrameskip-v4': 160.0,
    'AsterixNoFrameskip-v4': 104.0, 'AtlantisNoFrameskip-v4': 1444.0, 'BankHeistNoFrameskip-v4': 86.0, 
    'BattleZoneNoFrameskip-v4': 32.0, 'BeamRiderNoFrameskip-v4': 152.0, 'BoxingNoFrameskip-v4': 88.0, 
    'BreakoutNoFrameskip-v4': 95.0, 'CarnivalNoFrameskip-v4': 45.0, 'CentipedeNoFrameskip-v4': 137.0, 
    'ChopperCommandNoFrameskip-v4': 70.0, 'CrazyClimberNoFrameskip-v4': 735.0, 'DemonAttackNoFrameskip-v4': 231.0, 
    'DoubleDunkNoFrameskip-v4': 14.0, 'EnduroNoFrameskip-v4': 1091.0, 'FishingDerbyNoFrameskip-v4': 53.0,
    'FreewayNoFrameskip-v4': 34.0, 'FrostbiteNoFrameskip-v4': 71.0, 'GopherNoFrameskip-v4': 627.0, 
    'GravitarNoFrameskip-v4': 6.0, 'HeroNoFrameskip-v4': 201.0, 'IceHockeyNoFrameskip-v4': 2.0,
    'JamesbondNoFrameskip-v4': 17.0, 'KangarooNoFrameskip-v4': 74.0, 'KrullNoFrameskip-v4': 741.0,
    'KungFuMasterNoFrameskip-v4': 275.0, 'MsPacmanNoFrameskip-v4': 403.0, 'NameThisGameNoFrameskip-v4': 640.0, 
    'PhoenixNoFrameskip-v4': 67.0, 'PongNoFrameskip-v4': 21.0, 'PooyanNoFrameskip-v4': 427.0,
    'QbertNoFrameskip-v4': 588.0, 'RiverraidNoFrameskip-v4': 250.0, 'RoadRunnerNoFrameskip-v4': 217.0, 
    'RobotankNoFrameskip-v4': 73.0, 'SeaquestNoFrameskip-v4': 315.0, 'SpaceInvadersNoFrameskip-v4': 318.0, 
    'StarGunnerNoFrameskip-v4': 261.0, 'TimePilotNoFrameskip-v4': 24.0, 'UpNDownNoFrameskip-v4': 246.0, 
    'VideoPinballNoFrameskip-v4': 1078.0, 'WizardOfWorNoFrameskip-v4': 46.0, 
    'YarsRevengeNoFrameskip-v4': 207.0, 'ZaxxonNoFrameskip-v4': 30.0
}

# unfortunately DMControl produces these wird environment names
DMCONTROL_targets = {
    'acrobot-swingup': 158.764,
    'ball_in_cup-catch': 1000.0,
    'cartpole-balance': 993.194, 
    'cartpole-swingup': 857.492,
    'cheetah-run': 451.274,
    'finger-spin': 987.0, 
    'finger-turn_easy': 1000.0,
    'finger-turn_hard': 1000.0, 
    'fish-swim': 794.356, 
    'fish-upright': 999.433, 
    'hopper-hop': 116.777, 
    'hopper-stand': 975.575,
    'humanoid-run': 2.567, 
    'humanoid-stand': 17.218, 
    'humanoid-walk': 12.589, 
    'manipulator-bring_ball': 19.022, 
    'manipulator-insert_ball': 1000.0,
    'manipulator-insert_peg': 1000.0,
    'pendulum-swingup': 1000.0, 
    'point_mass-easy': 998.373,
    'reacher-easy': 1000.0, 
    'reacher-hard': 1000.0,
    'swimmer-swimmer15': 1000.0, 
    'swimmer-swimmer6': 1000.0,
    'walker-run': 513.464, 
    'walker-stand': 999.82,
    'walker-walk': 982.159
}


DARK_ROOM = {
    "MiniHack-Room-Dark-10x10-v0": 1,
    "MiniHack-Room-Dark-Dense-10x10-v0": 82,
    "MiniHack-Room-Dark-Sparse-10x10-v0": 1,
    "MiniHack-Room-Dark-17x17-v0": 1,
    "MiniHack-Room-Dark-Dense-17x17-v0": 82,
    "MiniHack-Room-Dark-Sparse-17x17-v0": 1,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(2,0)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(0,2)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(1,5)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(2,2)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(5,7)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(9,1)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(6,9)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(5,5)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(1,1)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(7,9)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(0,9)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(3,8)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(8,5)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(0,0)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(8,9)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(1,3)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(0,5)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(0,1)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(9,5)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(8,3)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(4,4)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(1,2)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(7,8)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(9,4)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(3,7)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(9,2)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(9,7)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(5,6)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(6,3)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(4,6)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(0,8)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(3,3)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(4,5)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(1,9)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(1,4)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(9,3)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(7,3)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(3,9)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(2,4)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(0,6)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(6,2)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(2,3)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(1,8)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(4,2)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(8,0)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(8,6)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(3,1)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(6,7)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(2,7)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(1,0)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(4,0)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(7,0)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(9,6)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(8,8)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(2,6)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(5,4)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(7,1)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(2,9)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(2,5)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(4,3)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(4,1)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(7,2)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(0,3)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(8,1)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(5,3)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(7,7)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(6,1)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(0,7)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(2,8)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(9,9)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(5,2)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(4,8)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(8,2)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(6,0)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(7,4)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(3,2)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(4,7)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(7,6)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(3,6)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(0,4)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(9,0)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(8,4)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(3,4)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(6,5)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(7,5)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(5,0)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(3,5)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(9,8)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(8,7)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(3,0)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(6,6)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(5,9)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(1,7)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(5,1)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(1,6)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(5,8)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(2,1)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(4,9)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(6,4)": 82,
    "MiniHack-Room-Dark-Dense-10x10-v0_(0,0)_(6,8)": 82,
}

GYM_MUJOCO_targets = {
    "HalfCheetah-v3": 6000.0,
    "Hopper-v3": 3600.0,
    "Walker2d-v3": 5000.0,
    "HalfCheetah-v2": 6000.0,
    "Hopper-v2": 3600.0,
    "Walker2d-v2": 5000.0,
}

PROCGEN_targets = {
    "bigfish": 40.0,
    "bossfight": 15.0,
    "caveflyer": 13.0,
    "chaser": 13.3128,
    "climber": 16.0,
    "coinrun": 10.0,
    "dodgeball": 20.0,
    "fruitbot": 38.0,
    "heist": 10.0,
    "jumper": 10.0,
    "leaper": 10.0,
    "maze": 10.0,
    "miner": 13.0,
    "ninja": 10.0,
    "plunder": 18.0,
    "starpilot": 67.0,
}


ALL_TARGETS = {**MT50_targets, **MT50_targets_v2, **ATARI_targets, **DMCONTROL_targets, **DARK_ROOM,
               **GYM_MUJOCO_targets, **MT50_v2_500steps_targets, **PROCGEN_targets}
