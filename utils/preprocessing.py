import numpy as np


def extract_features(obs: np.ndarray, n_bins: int = 4, n_regions: int = 4) -> tuple:
    """Compress a raw pixel frame into a small discrete feature vector.

    Detects the road by colour (grey-ish pixels) then quantises the fraction
    of road pixels in each horizontal strip into `n_bins` buckets.
    """
    r, g, b = obs[:, :, 0], obs[:, :, 1], obs[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    road_mask = (
        (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 20) &
        (np.abs(g.astype(np.int16) - b.astype(np.int16)) < 20) &
        (gray > 80) & (gray < 180)
    )
    strip_h = road_mask.shape[0] // n_regions
    features = []
    for i in range(n_regions):
        strip = road_mask[i * strip_h:(i + 1) * strip_h, :]
        features.append(min(int(float(np.mean(strip)) * n_bins), n_bins - 1))
    return tuple(features)


def to_tuple(a):
    """Recursively convert a nested array/iterable into a nested tuple.

    Used by the 'raw' observation method to make numpy arrays hashable
    for tabular Q-tables. Note: the raw method is impractical at scale
    (state space explodes); prefer the default 'feature' method.
    """
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a


def preprocess(raw_obs: np.ndarray, obs_cfg: dict) -> tuple:
    """Dispatch to the configured observation representation."""
    if obs_cfg.get("method", "feature") == "feature":
        return extract_features(
            raw_obs,
            obs_cfg.get("n_bins", 4),
            obs_cfg.get("n_regions", 4),
        )
    return to_tuple(raw_obs)
