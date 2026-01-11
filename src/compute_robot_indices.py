"""Compute robot mask for observation separation."""

import json
import jax.numpy as jnp


def compute_robot_mask(level_path: str, obs_dim: int = 679) -> jnp.ndarray:
    """Compute which observation dimensions belong to robot. Returns boolean array (obs_dim,)."""
    with open(level_path) as f:
        level_data = json.load(f)
    
    polygons = level_data['env_state']['polygon']
    circles = level_data['env_state']['circle']
    polygon_dim, circle_dim = 26, 18
    walls = {1, 2, 3}
    
    # Robot states: role=0, active, dynamic (inverse_mass > 0), not wall
    robot_polygons = [
        idx for idx, p in enumerate(polygons)
        if p.get('active', False) and p.get('role', -1) == 0 
        and p.get('inverse_mass', 1) > 0 and idx not in walls
    ]
    robot_circles = [
        idx for idx, c in enumerate(circles)
        if c.get('active', False) and c.get('role', -1) == 0 and c.get('inverse_mass', 1) > 0
    ]
    
    # Map polygon indices to obs indices (walls 1,2,3 removed)
    polygon_to_obs_idx = {p: i for i, p in enumerate(j for j in range(12) if j not in walls)}
    
    mask = jnp.zeros(obs_dim, dtype=bool)
    
    # Mark robot polygons
    for idx in robot_polygons:
        obs_idx = polygon_to_obs_idx[idx]
        mask = mask.at[obs_idx * polygon_dim:(obs_idx + 1) * polygon_dim].set(True)
    
    # Mark robot circles (start at 9*26=234)
    circle_start = 9 * polygon_dim
    for idx in robot_circles:
        mask = mask.at[circle_start + idx * circle_dim:circle_start + (idx + 1) * circle_dim].set(True)
    
    return mask
