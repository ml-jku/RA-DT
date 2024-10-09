import hydra
import omegaconf
import functools
import torch
import numpy as np
import h5py
from joblib import delayed
from tqdm import tqdm
from src.utils import maybe_split
from src.utils.misc import ProgressParallel
from src.envs import make_env
from src.algos.builder import make_agent
from src.buffers.buffer_utils import load_hdf5, append_to_hdf5


def encode_single_img_seq(img_encoder, path, device, max_batch_size=512): 
    assert path.suffix == ".hdf5", "Only .hdf5 files are supported."
    try: 
        observations, _, _, _, _ = load_hdf5(path)
    except Exception as e:
        print(f"Error reading from {path}.")
        raise e
    observations = torch.from_numpy(observations).float().to(device) / 255.0
    if observations.shape[0] > max_batch_size: 
        img_embeds = []
        for i in range(0, observations.shape[0], max_batch_size):
            with torch.no_grad():
                # amp here? 
                embeds = img_encoder(observations[i : i + max_batch_size]).detach().cpu().numpy()
            img_embeds.append(embeds)
        img_embeds = np.concatenate(img_embeds, axis=0)
    else: 
        with torch.no_grad():
            img_embeds = img_encoder(observations).detach().cpu().numpy()
    try: 
        append_to_hdf5(path, {"states_encoded": img_embeds})
    except Exception as e: 
        print(f"Error writing to {path}.")
        raise e
    del observations, img_embeds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_image_sequences(img_encoder, paths, device, n_jobs=-1):
    img_encoder.eval()
    fn = functools.partial(encode_single_img_seq, img_encoder=img_encoder, device=device)
    ProgressParallel(n_jobs=n_jobs, total=len(paths), timeout=5000)(delayed(fn)(path=p) for p in paths)
    

@hydra.main(config_path="configs", config_name="config")
def main(config):
    """
    For every sequence path, loads the sequence, encodes images and write encoded images to .hdf5 files.
    """
    print("Config: \n", omegaconf.OmegaConf.to_yaml(config, resolve=True, sort_keys=True))
    logdir = None
    env, _, _ = make_env(config, logdir)
    agent = make_agent(config, env, logdir)
    paths = agent.replay_buffer.trajectories
    if config.get("missing_only", False): 
        missing = []
        for p in tqdm(paths):
            try:
                with h5py.File(p, "r") as f:
                    if "states_encoded" not in f: 
                        missing.append(p)
            except Exception as e:
                print(f"Error reading {p}.")
                raise e
        print(f"Found {len(missing)} missing paths: ", missing)
        paths = missing
    encode_image_sequences(agent.policy.embed_image, paths, 
                           device=agent.device, n_jobs=config.get("n_jobs", -1))

if __name__ == "__main__":
    omegaconf.OmegaConf.register_new_resolver("maybe_split", maybe_split)
    main()
