import numpy as np 

class GMM:
	def __init__(self,numbDis=1,cov='diagonal'):
		# Initialize number of Gaussian distributions 
		self.numbDis = numbDis
		# Type of Covariance matrix 
		self.covType = cov 
		# weights of components 
		self.z = np.array([1/numbDis for i in range(numbDis)])
		# means 


	def set_coef(self,z=0):
		""" Initialize Z parameters"""
		# By default, z are uniform 
		self.z = z 

	def set_nth_mean(self, mean, nth_component): 
		# [gmm.set_nth_mean(means[i], i) for i in range(num_components)]

	def set_nth_cov(self,cov_matrix, nth_component):
		# [gmm.set_nth_cov(covs[i],i) for i in range(num_components)]


