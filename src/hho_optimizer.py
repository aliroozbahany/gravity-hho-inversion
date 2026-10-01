"""
Harris Hawks Optimization (HHO) Algorithm
Paper: Heidari, A. A., et al. (2019). 'Harris hawks optimization: Algorithm and applications.'
Future Generation Computer Systems, 97, 849-872.
"""

from typing import Callable, Tuple
import numpy as np
from scipy.special import gamma


class HarrisHawksOptimizer:

    def __init__(
        self,
        cost_func: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        dim: int,
        search_agents_no: int = 30,
        max_iter: int = 100,
        random_seed: int = None,
    ):
        self.cost_func = cost_func
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.dim = dim
        self.search_agents_no = search_agents_no
        self.max_iter = max_iter
        self.random_seed = random_seed

    @staticmethod
    def _levy(dim: int) -> np.ndarray:
        beta = 1.5
        sigma = (
            gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = 0.01 * np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        zz = np.power(np.abs(v), (1 / beta))
        step = u / zz
        return step

    def optimize(self) -> Tuple[np.ndarray, float, np.ndarray]:
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Initialize Rabbit (best location) & Hawks
        rabbit_location = np.zeros(self.dim)
        rabbit_energy = float("inf")

        # Uniform initialization within bounds
        x = self.lb + np.random.rand(self.search_agents_no, self.dim) * (
            self.ub - self.lb
        )
        convergence_curve = np.zeros(self.max_iter)

        for t in range(self.max_iter):
            # Check bounds and evaluate fitness
            for i in range(self.search_agents_no):
                x[i, :] = np.clip(x[i, :], self.lb, self.ub)
                fitness = self.cost_func(x[i, :])

                if fitness < rabbit_energy:
                    rabbit_energy = fitness
                    rabbit_location = x[i, :].copy()

            # Escaping energy of rabbit E
            e0 = 2 * np.random.rand() - 1  # [-1, 1]
            e = 2 * e0 * (1 - (t / self.max_iter))  # Factor decreases linearly

            # Update position of hawks
            for i in range(self.search_agents_no):
                if np.abs(e) >= 1:
                    # 1. Exploration phase
                    q = np.random.rand()
                    rand_idx = np.random.randint(0, self.search_agents_no)
                    x_rand = x[rand_idx, :]

                    if q >= 0.5:
                        x[i, :] = x_rand - np.random.rand() * np.abs(
                            x_rand - 2 * np.random.rand() * x[i, :]
                        )
                    else:
                        x_mean = np.mean(x, axis=0)
                        x[i, :] = (rabbit_location - x_mean) - np.random.rand() * (
                            (self.ub - self.lb) * np.random.rand() + self.lb
                        )
                else:
                    # 2. Exploitation phase
                    r = np.random.rand()
                    jump_strength = 2 * (1 - np.random.rand())

                    # Phase 2.1: Soft besiege
                    if r >= 0.5 and np.abs(e) >= 0.5:
                        delta_x = rabbit_location - x[i, :]
                        x[i, :] = delta_x - e * np.abs(
                            jump_strength * rabbit_location - x[i, :]
                        )

                    # Phase 2.2: Hard besiege
                    elif r >= 0.5 and np.abs(e) < 0.5:
                        delta_x = rabbit_location - x[i, :]
                        x[i, :] = rabbit_location - e * np.abs(delta_x)

                    # Phase 2.3: Soft besiege with progressive rapid dives
                    elif r < 0.5 and np.abs(e) >= 0.5:
                        y = rabbit_location - e * np.abs(
                            jump_strength * rabbit_location - x[i, :]
                        )
                        y = np.clip(y, self.lb, self.ub)
                        if self.cost_func(y) < self.cost_func(x[i, :]):
                            x[i, :] = y.copy()
                        else:
                            z_vec = y + np.random.randn(self.dim) * self._levy(
                                self.dim
                            )
                            z_vec = np.clip(z_vec, self.lb, self.ub)
                            if self.cost_func(z_vec) < self.cost_func(x[i, :]):
                                x[i, :] = z_vec.copy()

                    # Phase 2.4: Hard besiege with progressive rapid dives
                    elif r < 0.5 and np.abs(e) < 0.5:
                        x_mean = np.mean(x, axis=0)
                        y = rabbit_location - e * np.abs(
                            jump_strength * rabbit_location - x_mean
                        )
                        y = np.clip(y, self.lb, self.ub)
                        if self.cost_func(y) < self.cost_func(x[i, :]):
                            x[i, :] = y.copy()
                        else:
                            z_vec = y + np.random.randn(self.dim) * self._levy(
                                self.dim
                            )
                            z_vec = np.clip(z_vec, self.lb, self.ub)
                            if self.cost_func(z_vec) < self.cost_func(x[i, :]):
                                x[i, :] = z_vec.copy()

            convergence_curve[t] = rabbit_energy

        return rabbit_location, rabbit_energy, convergence_curve
