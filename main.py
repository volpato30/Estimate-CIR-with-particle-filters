import time
import numpy as np
# from smc_algo import simulate_path, likelihood_estimate_by_partical_filter
from smc_algo_py import simulate_path, likelihood_estimate_by_partical_filter


if __name__ == '__main__':
    np.random.seed(1234)
    start_time = time.time()
    param_vector = np.array([0.1862, 0.0654, 0.0481, -32.03])
    fake_vector  = np.array([0.3, 0.05, 0.0481, -32.03])
    y_list = []
    for i in range(20):
        y_list.append(simulate_path(param_vector, 1. / 52, 150, np.array([0.25, 1, 3, 5, 10])))

    def get_ll(vector):
        result = []
        for y in y_list:
            result.append(
                likelihood_estimate_by_partical_filter(vector, 200, 1. / 52, y, np.array([0.25, 1, 3, 5, 10]), 1e-3))
        return np.mean(result)

    true_ll = get_ll(param_vector)
    fake_ll = get_ll(fake_vector)

    print(f"run 20 times in {time.time() - start_time}s")
    print(f"true ll: {true_ll}")
    print(f"fake ll: {fake_ll}")
