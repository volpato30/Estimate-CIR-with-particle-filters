import time
import numpy as np
from smc_algo import simulate_path, get_ll
from scipy.optimize import differential_evolution
from functools import partial
import pickle


def print_fun(x, f, accepted):
    print(f"at x: {x}, get {f}. accepted: {accepted}")


def _target_function(search_vector, y_observation, y_size, maturity_vector):
    full_vector = np.concatenate((search_vector, np.array([0.0481, -32.03])))
    return -get_ll(full_vector, y_observation, y_size, maturity_vector)


if __name__ == '__main__':
    # np.random.seed(123)
    search_x_result = []
    num_rep = 20
    param_vector = np.array([0.1862, 0.0654, 0.0481, -32.03])
    # fake_vector = np.array([0.3, 0.0654, 0.0481, -32.03])
    maturity_vector = np.array([0.25, 1, 3, 5, 10])
    h=1e-3
    # generate observations
    for ii in range(num_rep):
        y_list = []
        y_size = 50
        for jj in range(y_size):
            y_list.append(simulate_path(param_vector, 1. / 52, 150, maturity_vector, h=h))
        y_observation = np.array(y_list)
        start_time = time.time()
        # optimization target
        target_function = partial(_target_function, y_observation=y_observation, y_size=y_size,
                                  maturity_vector=maturity_vector)
        print(target_function(np.array([0.1862, 0.0654])))
        kappa_bound = [(0.15, 0.3), (0.03, 0.08)]
        result = differential_evolution(target_function, kappa_bound, maxiter=2, disp=True)
        print(result.x, result.fun)
        search_x_result.append(result.x)
    with open("result.pkl", 'wb') as f:
        pickle.dump(search_x_result, f)

