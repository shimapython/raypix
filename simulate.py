from dataclasses import dataclass
import numpy as np
from typing import Any
import time
import tqdm
import core
from utils import iround
from typing import Callable

from core import (
    single_ray_simulate, 
    single_ray_opaque_simulate,
    RefractiveIndex,
    reliable_gradient_field
)


@dataclass
class SimConfig:
    medium: np.ndarray
    initial_direction: np.ndarray
    initial_coord: np.ndarray
    max_itr: int
    n_void: RefractiveIndex
    n_matter: RefractiveIndex
    wvlen: float # Note: This is the free space wavelength 
    pixel_length: float
    grad_field: np.ndarray = None
    propagate_norm: float = 1
    power_tolerance: float = 1e-6


@dataclass
class SimRes:
    path: np.ndarray
    itr: int
    coord: np.ndarray
    direction: np.ndarray


class Simulator:
    def __init__(self, sim_cfg: SimConfig):
        self.sim_cfg = sim_cfg
        self.time_measures = []

    def __call__(self, rand=None):
        stime = time.time()
        vals = single_ray_simulate(**self.sim_cfg.__dict__, rand=rand)  
        etime = time.time()
        self.time_measures.append(etime - stime)
        return vals

    def mcmc_simulte(self, num_sim, rand=None):
        path_total = np.zeros_like(self.sim_cfg.medium)
        end_coords = []
        stats = None
        for num in range(num_sim):
            print(num)
            path, itr, coord, direction = self(rand=rand)
            # By keeping track of all ending coords and ending powers we can measure radiative properties
            end_coords.append(coord)
            path_total += path / num_sim
            if coord[0] <= 2:
                raise ValueError()
        return path_total, end_coords 


def simulate_from_bottom_edge(med, initial_direction, wvlen, n_void, n_matter, pixel_length, grad_field=None, rand=None, num=10, propagate_norm=1):
    if grad_field is None:
        grad_field = reliable_gradient_field(med, smooth_sigma=1, normalize=True)
    path_total = np.zeros_like(med)
    num_rows, num_cols = med.shape
    num_sims = 0
    for j in tqdm.tqdm(range(num_cols)):
        initial_coord = np.array([num_rows - 1, j])
        for nsim in range(num):
            path, _, _, _ = single_ray_simulate(
                med,
                initial_direction=initial_direction,
                initial_coord=initial_coord,
                max_itr=2000,
                n_void=n_void,
                n_matter=n_matter,
                wvlen=wvlen,
                pixel_length=pixel_length,
                grad_field=grad_field,
                rand=rand,
                propagate_norm=propagate_norm
            )
            path_total += path
            num_sims += 1
    return path_total / num_sims


def simulate_onesided_reflection(med, beta, num_sim=100):
    import media
    if med is None:
        med = media.square_medium(500, 1500, 20, 0.3, seed=1)
    grad_field = reliable_gradient_field(med, smooth_sigma=1, normalize=True)
    initial_direction = np.array([-1, 0])
    ref_vals = []
    all_paths = []
    for _ in range(num_sim):
        try:
            initial_coord = np.array([
                med.shape[0] - 2, 
                np.random.choice(np.arange(200, med.shape[1] - 200)) 
            ])
            path, a_, coord, _, _, power = single_ray_opaque_simulate(
                med, initial_direction, initial_coord=initial_coord, max_itr=20000,
                beta=beta,
                grad_field=grad_field,
            )
            if coord[0] >= 0.9 * med.shape[0]:
                ref_vals.append(power)
            else:
                ref_vals.append(0)
            print(f"_: success  reflection={np.mean(ref_vals)}")
            all_paths.append(path)
        except AssertionError:
            print(f"_: failed")
    
    return np.array(ref_vals), all_paths
            
