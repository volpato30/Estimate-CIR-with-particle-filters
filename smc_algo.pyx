import numpy as np
import scipy.integrate as integrate
from scipy.stats import gamma, ncx2, norm
import logging


## for conveniance, we concatenate the 4 params into a vector: (kappa, theta, sigma, lambda / sigma^2)
cdef double get_const_c(double[:] param_vector, double delta_t):
    return 2 * param_vector[0] / ( param_vector[2]*param_vector[2] * (1 - np.exp(-param_vector[0] * delta_t)) )

cdef double get_const_q(double[:] param_vector):
    return 2 * param_vector[0] * param_vector[1] / (param_vector[2]*param_vector[2]) - 1

def get_coef(time_to_maturity, param_vector: np.ndarray):
    gamma = np.sqrt( np.square(param_vector[0]+param_vector[3]*np.square(param_vector[2])) \
                     + 2*np.square(param_vector[2]) )
    q = get_const_q(param_vector)
    temp_sum = param_vector[0] + param_vector[3]*np.square(param_vector[2]) + gamma

    b_numerator = 2 * (np.exp(gamma * time_to_maturity) - 1)
    denominator = 2 * gamma + temp_sum * (np.exp(gamma * time_to_maturity) - 1)
    b = b_numerator / (denominator * time_to_maturity)

    a = (q + 1) / time_to_maturity * np.log(2*gamma*np.exp(time_to_maturity*temp_sum/2) / denominator)
    return a, b


def generate_truncated_normal(mean, std, n, clip=0):
    if not n:
        return None
    result = np.random.normal(mean, std, size=n)
    clipped_index = result<0
    if np.any(clipped_index):
        result[clipped_index] = generate_truncated_normal(mean, std, np.sum(clipped_index), clip=clip)
    return result


cdef double get_a(double[:] coef_B, double h):
    return np.sum(np.square(coef_B)) / (2 * h)

def get_b_c(observed_y, coef_A, coef_B, h):
    return -np.sum((observed_y - coef_A) * coef_B) / h, 1 / (2*h) *np.sum(np.square(observed_y - coef_A))

cdef double phi(double x, double a, double b, double c, double h, int N):
    return 1 / np.power(2 * np.pi * h, N / 2.0) * np.exp(-a*np.square(x) - b * x - c)

# cdef double get_K(double a, double b, double c, double h, int N):
#     """
#     numerical integration method(slow)
#     """
#     result = integrate.quad(lambda x: phi(x, a, b, c, h, N), 0, float("inf"))
#     return result[0]

cdef double get_K(double a, double b, double c, double h, int N):
    """
    for numerical stability, first compute logK then take exponential
    :param a: 
    :param b: 
    :param c: 
    :param h: 
    :param N: 
    :return: 
    """
    logK = -(N-1) / 2. * np.log(2*np.pi) - N / 2. * np.log(h) + (b*b / (4*a) - c) - 0.5*np.log(2*a)
    logK += np.log(1-norm.cdf(0, loc=-b/(2*a), scale=np.sqrt(1/(2*a))))
    return np.exp(logK)


def resample(alpha, w):
    assert len(alpha) == len(w)
    cdf = np.cumsum(w) / np.sum(w)
    cdf_hat = np.ones_like(w) * 1. / len(w)
    cdf_hat[0] = 0
    cdf_hat = np.cumsum(cdf_hat)
    cdf_hat += np.random.uniform(0, 1. / len(w))

    result = np.zeros_like(alpha)
    cdef double[:] result_view = result
    cdef int j = 0
    cdef Py_ssize_t i
    for i in range(len(w)):
        while cdf[j] < cdf_hat[i]:
            j += 1
        result[i] = alpha[j]
    return result

cdef double get_n_tilda(double[:] w):
    return np.square(np.sum(w)) / np.sum(np.square(w))

def likelihood_estimate_by_partical_filter(double[:] param_vector, int num_particles, double delta_t,
                                        double[:, :]observed_y_matrix, time_to_maturity_array, double h):
    """
    observed_y_matrix: T by N matrix, T is number of steps and N is number of maturities
    return the loglikelihood
    """
    cdef int T, N
    cdef double const_c, const_q, a, b, c, shape_param, scale_param, K, n_tilda
    cdef double log_likelihood = 0
    T = observed_y_matrix.shape[0]
    N = observed_y_matrix.shape[1]
    assert N == time_to_maturity_array.size
    coef_A, coef_B = get_coef(time_to_maturity_array, param_vector)
    const_c = get_const_c(param_vector, delta_t)
    const_q = get_const_q(param_vector)
    a = get_a(coef_B, h)
    # initial alpha and w
    b, c = get_b_c(observed_y_matrix[0], coef_A, coef_B, h)
    alpha = generate_truncated_normal(mean=-b/(2*a), std=np.sqrt(1/(2*a)), n=num_particles)
    # alpha_1 follows gamma distribution
    shape_param = const_q+1
    scale_param = 1 / (const_c * (1 - np.exp(-param_vector[0]*delta_t)))
    K = get_K(a, b, c, h, N)
    w = K * gamma.pdf(x=alpha, a=shape_param, scale=scale_param)
    log_likelihood += np.log(np.mean(w))
    cdef Py_ssize_t t
    for t in range(1, T):
        b, c = get_b_c(observed_y_matrix[t], coef_A, coef_B, h)
        new_alpha = generate_truncated_normal(mean=-b/(2*a), std=np.sqrt(1/(2*a)), n=num_particles)
        # alpha_t conditioning on alpha_{t-1} follows non-central X2 distribution.
        nc = 2 * const_c * np.exp(-param_vector[0]*delta_t) * alpha
        K = get_K(a, b, c, h, N)
        conditional_prob = 2 * const_c * ncx2.pdf(x=2*const_c*new_alpha, df=2*const_q+2, nc=nc)
        assert len(conditional_prob) == num_particles
        new_w = K * conditional_prob * w / np.sum(w)
        assert len(new_w) == num_particles
        n_tilda = get_n_tilda(new_w)
        if n_tilda < num_particles / 2.:
            #resampling
            new_alpha = resample(new_alpha, new_w)
            new_w = np.ones_like(new_w) * np.sum(new_w) / num_particles
        assert len(new_alpha) == num_particles
        log_likelihood += np.log(np.sum(new_w))
        w = new_w
        alpha = new_alpha
    return log_likelihood

def simulate_path(param_vector, delta_t, T, time_to_maturity_array, h):
    """

    :param param_vector:
    :param delta_t:
    :param T:
    :param time_to_maturity_array:
    :param h:
    :return:
        observed_y: (T, num_maturities)
    """
    observed_y = np.ones((T, time_to_maturity_array.size), dtype=np.float64)

    const_c = get_const_c(param_vector, delta_t)
    const_q = get_const_q(param_vector)
    alpha = np.random.gamma(shape=const_q+1, scale=1/(const_c * (1 - np.exp(-param_vector[0]*delta_t))))
    coef_A, coef_B = get_coef(time_to_maturity_array, param_vector)
    observed_y[0] = -coef_A + coef_B * alpha + np.random.normal(0, np.sqrt(h), size=time_to_maturity_array.size)
    cdef Py_ssize_t ii
    for ii in range(1, T):
        nc = 2 * const_c * np.exp(-param_vector[0]*delta_t) * alpha
        alpha = np.random.noncentral_chisquare(df=2*const_q+2, nonc=nc) / (2 * const_c)
        observed_y[ii] = -coef_A + coef_B * alpha + np.random.normal(0, np.sqrt(h), size=time_to_maturity_array.size)
    return observed_y


def check_param_vector(vector):
    # for numerical stability, return false for out of bound param vectors
    if vector[0] < 0.1 or vector[0] > 0.5:
        return False
    if vector[1] < 0.01 or vector[1] > 0.1:
        return False
    if vector[2] < 0.02 or vector[2] > 0.08:
        return False
    if np.abs(vector[3]) > 50:
        return False
    return True

def get_ll(vector, double[:, :, :] y_observation, int y_size, maturity_vector, h):
    if not check_param_vector(vector):
        return -float("inf")
    cdef double sum = 0
    cdef int i
    for i in range(y_size):
        sum += likelihood_estimate_by_partical_filter(vector, 200, 1. / 52, y_observation[i], maturity_vector, h)
    return sum / y_size