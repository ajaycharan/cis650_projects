import numpy as np 
# create mixture of three Gaussians
num_components=3
num_max_samples=100
gmm=GMM(num_components)

dimension=3

# set means (TODO interface should be to construct mixture from individuals with set parameters)
means = np.zeros((num_components, dimension))
means[0]=[-5.0, -4.0]
means[1]=[7.0, 3.0]
means[2]=[0, 0.]
[gmm.set_nth_mean(means[i], i) for i in range(num_components)]

# set covariances
covs=zeros((num_components, dimension, dimension))
covs[0]=array([[2, 1.3],[.6, 3]])
covs[1]=array([[1.3, -0.8],[-0.8, 1.3]])
covs[2]=array([[2.5, .8],[0.8, 2.5]])
[gmm.set_nth_cov(covs[i],i) for i in range(num_components)]

# set mixture coefficients, these have to sum to one (TODO these should be initialised automatically)
weights=array([0.5, 0.3, 0.2])
gmm.set_coef(weights)


