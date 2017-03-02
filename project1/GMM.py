"""The following code is developed based on Gaussian Mixture Model, introduced by """
# Wei Xue <xuewei4d@gmail.com>

import numpy as np
from scipy import linalg
from six.moves import zip
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import row_norms
from sklearn.cluster import KMeans
from sklearn.utils.extmath import logsumexp



def estimate_gauss_params(X, resp, cov_reg, covariance_type):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    if covariance_type == 'full':
        n_components, n_features = means.shape
        covariances = np.empty((n_components, n_features, n_features))
        for k in range(n_components):
            diff = X - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
            covariances[k].flat[::n_features + 1] += cov_reg

    elif covariance_type == 'diag':
        avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
        avg_means2 = means ** 2
        avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
        covariances =  avg_X2 - 2 * avg_X_means + avg_means2 + cov_reg
    else:
        covariances = estimate_gaussian_covariances_diag(resp, X, nk, means, cov_reg).mean(1)
    return nk, means, covariances


def compute_cholesky(covariances, covariance_type):
    if covariance_type in 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            cov_chol = linalg.cholesky(covariance, lower=True)
            precisions_chol[k] = linalg.solve_triangular(cov_chol, np.eye(n_features), lower=True).T
    else:
        precisions_chol = 1.0 / np.sqrt(covariances)
    return precisions_chol


def compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    if covariance_type == 'full':
        n_components = matrix_chol.shape[0]
        log_det_chol = (np.sum(np.log(matrix_chol.reshape(n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """This function estimate the log Gaussian probability."""

    num_samples, n_features = X.shape
    n_components = means.shape[0]
    log_det = compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        log_prob = np.empty((num_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = (np.sum((means ** 2 * precisions), 1) -
                    2. * np.dot(X, (means * precisions).T) +
                    np.dot(X ** 2, precisions.T))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means ** 2, 1) * precisions -
                    2 * np.dot(X, means.T * precisions) +
                    np.outer(row_norms(X, squared=True), precisions))
    return -(n_features * np.log(2 * np.pi) + log_prob)/2 + log_det


class GaussianMixture():
    """Gaussian Mixture Model 
    """
    def __init__(self, n_components=1, covariance_type='full', max_iter=100, n_init=1,
                 weights_init=None, means_init=None,random_state=None
                  ):
        self.n_components = n_components
        self.tol = 1e-3
        self.cov_reg = 1e-6
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.precisions_init = None
      
    def initialize_params(self, X, random_state):
        """Initialize the model's parameters.
        """
        num_samples, _ = X.shape
        resp = np.zeros((num_samples, self.n_components))
        label = KMeans(n_clusters=self.n_components, n_init=1,random_state=random_state).fit(X).labels_
        resp[np.arange(num_samples), label] = 1

        # Intialize weights, means and covariances using Kmean method 
        weights, means, covariances = estimate_gauss_params(X, resp, self.cov_reg, self.covariance_type)
        weights /= num_samples

        self.weights = weights 
        self.means = means  

        self.covariances = covariances
        self.prec_cholesky = compute_cholesky(covariances, self.covariance_type)

    def E_step(self, X):
        log_prob_norm, log_resp = self.estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def M_step(self, X, log_resp):
        num_samples = X.shape[0]
        self.weights, self.means, self.covariances = (
            estimate_gauss_params(X, np.exp(log_resp), self.cov_reg,
                                          self.covariance_type))
        self.weights /= num_samples
        self.prec_cholesky = compute_cholesky(
            self.covariances, self.covariance_type)

    def compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def get_params(self):
        return (self.weights, self.means, self.covariances,
                self.prec_cholesky)

    def set_params(self, params):
        (self.weights, self.means, self.covariances,
         self.prec_cholesky) = params

        # Attributes computation
        n_features = self.means.shape[1]

        if self.covariance_type == 'full':
            self.precisions_ = np.empty(self.prec_cholesky.shape)
            for k, prec_chol in enumerate(self.prec_cholesky):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)
        else:
            self.precisions_ = self.prec_cholesky ** 2

    def fit(self, X, y=None):
        X = check_array(X, dtype=[np.float64, np.float32])
        do_init = not(hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1
        max_lower_bound = -np.infty
        self.converged = False
        random_state = check_random_state(self.random_state)
        num_samples = X.shape[0]
        for init in range(n_init):

            if do_init:
                self.initialize_params(X, random_state)
                self.lower_bound= -np.infty

            for n_iter in range(self.max_iter):
                prev_lower_bound = self.lower_bound

                log_prob_norm, log_resp = self.E_step(X)
                self.M_step(X, log_resp)
                self.lower_bound= log_prob_norm
                change = self.lower_bound- prev_lower_bound

                if abs(change) < self.tol:
                    self.converged = True
                    break
            if self.lower_bound> max_lower_bound:
                max_lower_bound = self.lower_bound
                best_params = self.get_params()
                best_n_iter = n_iter

        self.set_params(best_params)
        self.n_iter = best_n_iter
        return self

    def estimate_log_prob_resp(self, X):
        weighted_log_prob = self.estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp
    def set_params(self, params):
        (self.weights, self.means, self.covariances,self.prec_cholesky) = params

        n_features = self.means.shape[1]

        if self.covariance_type == 'full':
            self.precisions_ = np.empty(self.prec_cholesky.shape)
            for k, prec_chol in enumerate(self.prec_cholesky):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)
        else:
            self.precisions_ = self.prec_cholesky ** 2

    def predict(self, X, y=None):

        X = check_array(X, dtype=[np.float64, np.float32])
        return self.estimate_weighted_log_prob(X).argmax(axis=1)
 
    def estimate_weighted_log_prob(self, X):
        return estimate_log_gaussian_prob(X, self.means, self.prec_cholesky, self.covariance_type)+ np.log(self.weights)

    def score_samples(self, X):
        X = check_array(X, dtype=[np.float64, np.float32])
        return logsumexp(self.estimate_weighted_log_prob(X), axis=1)

