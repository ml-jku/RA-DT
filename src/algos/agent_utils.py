import itertools
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from ..utils.loss_functions import DistanceSmoothedCrossEntropyLoss


def get_param_count(model, prefix="model"):
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {f"{prefix}_total": params, f"{prefix}_trainable": trainable_params}


def make_random_proj_matrix(in_dim, proj_dim, seed=42, norm=False, device=None, batch_size=None):
    # to deterministically get the same projection matrix (for every size), fix the rng seed
    rng = np.random.RandomState(seed)
    shape = (proj_dim, in_dim) if batch_size is None else (batch_size, proj_dim, in_dim)
    # scale = np.sqrt(in_dim / proj_dim)
    scale = np.sqrt(1 / proj_dim)
    rand_matrix = rng.normal(loc=0, scale=scale, size=shape).astype(dtype=np.float32)
    if norm: 
        norms = np.linalg.norm(rand_matrix, axis=0) + 1e-8
        rand_matrix = rand_matrix / norms
    if device is not None: 
        rand_matrix = torch.from_numpy(rand_matrix).to(device)
    return rand_matrix


def make_random_proj_matrix_torch(in_dim, proj_dim, seed=42, device="cpu"):
    # Create a local generator with the specified seed
    # issue with having numpy a and torch version is, that their rngs are not the same    
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    rand_matrix = torch.normal(mean=0.0, std=np.sqrt(in_dim / proj_dim),
                                size=(proj_dim, in_dim), 
                                generator=generator, 
                                device=device)
    return rand_matrix


def aggregate_embeds(x_embed, tok_to_pos, agg_token="a", dropout=0,
                     attention_mask=None, max_embed_len=None, chunk_len=None):
    """
    Aggregate embeddings based on the given token positions and aggregation method.

    Args:
        x_embed (torch.Tensor): tensor of shape (batch_size, seq_len, embed_dim) containing the embeddings to be aggregated.
        tok_to_pos (dict): dictionary mapping tokens to their positions in the sequence.
        agg_token (str): token to be used for aggregation.
        attention_mask (torch.Tensor): tensor of shape (batch_size, seq_len) containing the attention mask.
        max_embed_len (int): maximum number of embeddings to be used for aggregation. 
            This is measured in timesteps, not tokens. 
        dropout (float): dropout probability.
        chunk_len: Int. If given, max_embed_len is not considered. Instead each sequence is split up into several chunks
            along the sequence dimension. This is measured in timesteps, not tokens.

    Returns:
    - x_embed_mean (torch.Tensor): tensor of shape (batch_size, embed_dim) containing the aggregated embeddings.
    """
    if chunk_len is not None: 
        # set None, as we consider the sequence in chunks
        max_embed_len = None
    
    # mostly copied from base_prompt.py
    num_tokens = max([pos for tokpos in tok_to_pos.values() 
                        for pos in ([tokpos] if isinstance(tokpos, int) else list(tokpos))]) + 1
    attention_mask = attention_mask.repeat_interleave(num_tokens, dim=1)
    if agg_token != "all":
        if tok_to_pos is not None:
            # handle case if we have multiple tokens per token type
            batch_size, seq_len, embed_dim = x_embed.shape
            x_embed = x_embed.reshape(batch_size, seq_len // num_tokens, num_tokens, embed_dim)
            if max_embed_len is not None and seq_len > max_embed_len: 
                x_embed = x_embed[:, -max_embed_len:]
                attention_mask = attention_mask.reshape(batch_size, seq_len // num_tokens, num_tokens)
                attention_mask = attention_mask[:, -max_embed_len:].flatten(1)
                seq_len = max_embed_len * num_tokens
            if agg_token == "concat": 
                x_embed = x_embed.reshape(batch_size, seq_len // num_tokens, -1)
                attention_mask = attention_mask.reshape(batch_size, seq_len // num_tokens, num_tokens)[..., -1].flatten(1)
            else: 
                if "_" in agg_token: 
                    token_pos = list(itertools.chain(
                        *[[tok_to_pos[tok]] if isinstance(tok_to_pos[tok], int) else tok_to_pos[tok] 
                            for tok in agg_token.split("_")]
                    ))
                else: 
                    token_pos = tok_to_pos[agg_token]
                
                # select respective tokens
                x_embed = x_embed[:, :, token_pos]
                attention_mask = attention_mask.reshape(batch_size, seq_len // num_tokens, num_tokens)[..., token_pos]
                
                if chunk_len is not None: 
                    # split up chunks --> increase batch size by n_chunks
                    # x_embed: [batch_size, timesteps, num_tokens, embed_dim]
                    # --> [batch_size * n_chunks, chunk_len * num_tokens, embed_dim]
                    n_chunks = x_embed.shape[1] // chunk_len
                    x_embed = x_embed.reshape(x_embed.shape[0] * n_chunks, chunk_len, -1)
                    attention_mask = attention_mask.reshape(attention_mask.shape[0] * n_chunks, chunk_len)
                else: 
                    x_embed = x_embed.reshape(batch_size, -1, embed_dim)
                    attention_mask = attention_mask.flatten(1)
        else: 
            token_pos, num_tokens = tok_to_pos[agg_token], len(tok_to_pos)
            assert x_embed.shape[1] % num_tokens == 0 and attention_mask.shape[1] % num_tokens == 0
            x_embed = x_embed[:, token_pos::num_tokens]
            attention_mask = attention_mask[:, token_pos::num_tokens]
    else: 
        if max_embed_len is not None: 
            x_embed = x_embed[:, -(max_embed_len * num_tokens):]
            attention_mask = attention_mask[:, -(max_embed_len * num_tokens):]
    
    if dropout > 0: 
        # torch.nn.functional.dropout1d scales non-dropped-out values by 1 / (1 - dropout) to account for
        # scale difference. we want to avoid this. 
        # x_embed = torch.nn.functional.dropout1d(x_embed, p=dropout, training=True)
        mask = torch.empty(x_embed.shape[:2], device=x_embed.device).bernoulli_(1 - dropout).long()
        x_embed = x_embed * mask.unsqueeze(-1)
    x_embed_mean = torch.sum(x_embed * attention_mask.float().unsqueeze(-1), dim=1) \
                    / (torch.sum(attention_mask.float(), -1, keepdim=True) + 1e-8)
    return x_embed_mean


def dropout_dims(x, p=0.5, dim=None):
    """
    Drops out dimnensions of given vector. E.g., useful for continous state/actions

    Args:
        x (Tensor): Input tensor.
        p (float, optional): Dropout probability. Default is 0.5.
    Returns:
        Tensor: Input tensor with dropped out dims
    """
    if dim is not None: 
        shape = list(x.shape)
        shape[dim] = 1
        mask = torch.bernoulli(torch.full(shape, 1-p, device=x.device)).long()
    else: 
        mask = torch.bernoulli(torch.full_like(x, 1-p)).long()
    return x * mask


def make_gaussian_noise(x, mean=0.0, std=0.1, nonzero=True, constant=True):
    """
    Makes Gaussian noise for a tensor input.

    Args:
        x (Tensor): Input tensor with shape [batch_size, seq_len, dim].
        mean (float, optional): Mean of the Gaussian distribution. Default is 0.0.
        std (float, optional): Standard deviation of the Gaussian distribution. Default is 1.0.
    Returns:
        Tensor: Noise.
    """
    if std is None: 
        std = 0.1
    if len(x.shape) == 1: 
        noise = torch.normal(mean=mean, std=std, size=(x.shape[0],), device=x.device)
    else: 
        if constant: 
            batch_size, seq_len, dim = x.shape
            # constant noise along seq_len
            noise = torch.normal(mean=mean, std=std, size=(batch_size, 1, dim), device=x.device)
        else: 
            noise = torch.normal(mean=mean, std=std, size=x.shape, device=x.device)
    if nonzero: 
        # handles padding + 0-dims in metaworld/dmc
        noise = noise * (x != 0)
    return noise


def add_gaussian_noise(x, mean=0.0, std=0.1, nonzero=True, constant=True):
    return x + make_gaussian_noise(x, mean=mean, std=std, nonzero=nonzero, constant=constant)


class HLGaussLoss(torch.nn.Module):
    
    def __init__(self, min_value=-1, max_value=1, num_bins=128, sigma=0.01, bin_std_ratio=0.75, reduction="mean"):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.bin_width = (max_value - min_value) / num_bins
        self.sigma = sigma
        self.bin_std_ratio = bin_std_ratio
        self.reduction = reduction
        if bin_std_ratio is not None: 
            # set as as proposed by: https://arxiv.org/abs/2403.03950
            # distributes probability mass to ~6 locations. 
            self.sigma = self.bin_width * bin_std_ratio
        self.register_buffer('support', torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, self.transform_to_probs(target), reduction=self.reduction)
    
    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        target = torch.clamp(target, self.min_value, self.max_value)
        cdf_evals = torch.special.erf((self.support - target.unsqueeze(-1)) / (torch.sqrt(torch.tensor(2.0)) * self.sigma))
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)
    

def make_loss_fn(kind, reduction="mean", label_smoothing=0.0, loss_fn_kwargs=None):
    if kind in ["mse", "td3+bc"]:
        loss_fn = torch.nn.MSELoss(reduction=reduction)
    elif kind in ["smooth_l1", "dqn"]:
        loss_fn = torch.nn.SmoothL1Loss(reduction=reduction)
    elif kind == "huber":
        loss_fn = torch.nn.HuberLoss(reduction=reduction)
    elif kind == "nll":
        loss_fn = torch.nn.NLLLoss(reduction=reduction)
    elif kind == "ce":
        loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
    elif kind == "dist_ce":
        loss_fn = DistanceSmoothedCrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
    elif kind in ["td3", "ddpg", "sac"]:
        loss_fn = None
    elif kind == "hl_gauss": 
        loss_fn_kwargs = {} if loss_fn_kwargs is None else loss_fn_kwargs
        loss_fn = HLGaussLoss(**loss_fn_kwargs)
    else:
        raise ValueError(f"Unknown loss kind: {kind}")
    return loss_fn


class CustomDDP(DistributedDataParallel):
    """
    The default DistributedDataParallel enforces access to class the module attributes via self.module. 
    This is impractical for our use case, as we need to access certain module access throughout. 
    We override the __getattr__ method to allow access to the module attributes directly.
    
    For example: 
    ```
        # default behaviour
        model = OnlineDecisionTransformerModel()
        model = DistributedDataParallel(model)
        model.module.some_attribute
        
        # custom behaviour using this class
        model = OnlineDecisionTransformerModel()
        model = CustomDDP(model)
        model.some_attribute
        
    ```        
    Shoudl not cause any inconsistencies: 
    https://discuss.pytorch.org/t/access-to-attributes-of-model-wrapped-in-ddp/130572
    
    """
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
