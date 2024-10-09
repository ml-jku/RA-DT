def make_buffer_class(kind): 
    if kind == "cache":
        from .cache import Cache
        return Cache
    from .trajectory_buffer import TrajectoryReplayBuffer
    return TrajectoryReplayBuffer
