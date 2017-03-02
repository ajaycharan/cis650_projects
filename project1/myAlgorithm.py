
#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from skimage import data, util
import skimage
from skimage.measure import label, regionprops
import numpy as np
from scipy.spatial import distance as dist
from numpy.linalg import inv 
import os 
import cv2 
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn import datasets, linear_model
from GMM import GaussianMixture
from six.moves import cPickle as pickle
import time
from sklearn.cluster import KMeans
print 'Import has been done!!'
RED_LIKELIHOOD_TOL = 7e-10
AREA_TOL = 1400 # threshold of pixel area of a barrel in the image 
MIN_AREA_TOL = 500 # min threshold of pixel area of a barrel in the image 
BIG_DENS_TOL = 0.74# hard threshold of pixel density in a contour to be a barrel 
SMALL_DENS_TOL = 0.4 # soft threshold of pixel density in a contour to be a barrel 

class DATA:
	def __init__(self,data_set,target_set,ids_set):
		self.data = data_set
		self.target = target_set
		self.ids = ids_set


def load_train_data(transform_color='on',colorspace='hsv'):
	print '----- Loading training data.... -----'
	train_img_folder = "data/Proj1_Train"
	img_files = os.listdir(train_img_folder)
	pickle_folder = "data/pickle_folder"
	train_file = os.path.join(pickle_folder,'train.pickle')	
	target_file = os.path.join(pickle_folder,'target.pickle')	
	ids_file = os.path.join(pickle_folder,'ids.pickle') 

	try:
		with open(train_file, 'rb') as f:
			print 'Open train file:',train_file
			train_set_pixels = pickle.load(f)
			print np.shape(train_set_pixels)	
	except Exception as e:
		print('Unable to process data from', train_file, ':', e)
		raise

	try:
		with open(target_file, 'rb') as f:
			print 'Open target file:',target_file
			target_set = pickle.load(f)
			print np.shape(target_set)	
	except Exception as e:
		print('Unable to process data from', target_file, ':', e)
		raise
	try:
		with open(ids_file, 'rb') as f:
			print 'Open ids file:',ids_file
			ids_set = pickle.load(f)
			print np.shape(ids_set)	
	except Exception as e:
		print('Unable to process data from', ids_file, ':', e)
		raise

	if transform_color=='off':
		return [train_set_pixels,target_set,ids_set]

	# # Test segmented image by displaying 
	# for image_name in img_files:
	# 	# image_name = img_files[0] 
	# 	command = raw_input("continue display?")
	# 	if command == 'N' or command == 'n':
	# 		break 
	# 	image_file = os.path.join(train_img_folder,image_name)
	# 	img = cv2.imread(image_file)
	# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# 	# plt.imshow(img,aspect='auto')
	# 	# plt.show()
		
	# 	print 'displaying..'
	# 	# segment_indices = (ids_set == image_name)
	# 	barrel_indices = (ids_set == image_name)*(target_set == 1)
	# 	locate_pixels = train_set[barrel_indices]
	# 	print np.shape(locate_pixels)
	# 	print np.shape(img)
	# 	img[:,:] = (0,0,0)
	# 	img[locate_pixels[:,0], locate_pixels[:,1]] = (255,255,255)	
	# 	plt.imshow(img,aspect='auto')
	# 	plt.show()

	# Assign RGB values to the train_set 

	train_set = np.zeros((len(target_set),3))
	for image_name in img_files:
		# command = raw_input("continue display?")
		# if command == 'N' or command == 'n':
		# 	break 
		image_file = os.path.join(train_img_folder,image_name)
		img = cv2.imread(image_file)
		if colorspace == 'RGB':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		elif colorspace == 'YCrCb':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
		elif colorspace == 'LUV':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
		elif colorspace == 'HLS':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
		else:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# plt.imshow(img,aspect='auto')
		# plt.show()
		segment_indices = (ids_set == image_name)
		segment_pixels = train_set_pixels[segment_indices]
		# print segment_pixels[0]
		# print np.shape(segment_pixels)
		train_set[ids_set==image_name] = img[segment_pixels[:,0], segment_pixels[:,1]]
		
		# # For debugging: show segmented image
		# img[segment_pixels[:,0], segment_pixels[:,1]] = (255,255,255)	
		# plt.imshow(img,aspect='auto')
		# plt.show()
	
	print '----- Completed loading training data! -----'
	return [train_set,target_set,ids_set]

def train_GaussianMixtures(GMM_model_file,color_space='hsv'):
	[train_set,target_set,ids_set] = load_train_data(color_space)
	print np.shape(train_set)
	n_classes = len(np.unique(target_set))
	print '-- Number of color classes:',n_classes
	print '---------------- Training GMM models.... ----------------'

	# for c in np.unique(target_set):
		# print 'class ', c 
		# print len(target_set[target_set == c])

	data_set = DATA(train_set,target_set,ids_set)

	# print np.shape(data_set.data)
	# print np.shape(data_set.target)
	# print np.shape(data_set.target)
	# print sum(data_set.target)


	# X_train = data_set.data
	# y_train = data_set.target 

	# Break up the dataset into non-overlapping training (75%) and testing		
	# (25%) sets.		
	skf = StratifiedKFold(data_set.target, n_folds=25)		
	# Only take the first fold.		
	train_index, test_index = next(iter(skf))
	X_train = data_set.data[train_index]	
	y_train = data_set.target[train_index]
	X_test = data_set.data[test_index]		
	y_test = data_set.target[test_index]

	n_classes = len(np.unique(y_train))

	# Extract X_train, X_test... for each class 
	c1_X_train = X_train[y_train==1]
	c1_X_test = X_test[y_test==1]
	c1_y_train = y_train[y_train==1]
	c1_y_test = y_test[y_test==1]

	c2_X_train = X_train[y_train==2]
	c2_X_test = X_test[y_test==2]
	c2_y_train = y_train[y_train==2]
	c2_y_test = y_test[y_test==2]

	c3_X_train = X_train[y_train==3]
	c3_X_test = X_test[y_test==3]
	c3_y_train = y_train[y_train==3]
	c3_y_test = y_test[y_test==3]

	c4_X_train = X_train[y_train==4]
	c4_X_test = X_test[y_test==4]
	c4_y_train = y_train[y_train==4]
	c4_y_test = y_test[y_test==4]

	c5_X_train = X_train[y_train==5]
	c5_X_test = X_test[y_test==5]
	c5_y_train = y_train[y_train==5]
	c5_y_test = y_test[y_test==5]

	c6_X_train = X_train[y_train==6]
	c6_X_test = X_test[y_test==6]
	c6_y_train = y_train[y_train==6]
	c6_y_test = y_test[y_test==6]

	c7_X_train = X_train[y_train==7]
	c7_X_test = X_test[y_test==7]
	c7_y_train = y_train[y_train==7]
	c7_y_test = y_test[y_test==7]

	# print  'len c2 train:',len(c2_X_train)
	# print 'len c2 test:', len(c2_y_test)

	# Number of components for each class 
	c1_comps = 2
	c2_comps = 2 
	c3_comps = 2 
	c4_comps = 2
	c5_comps = 2
	c6_comps = 3
	c7_comps = 2 

	
	c1_classifier = GaussianMixture(n_components=c1_comps, covariance_type='diag', max_iter=20)	                  
	c1_kmeans = KMeans(n_clusters=c1_comps,random_state=0).fit(c1_X_train)

	c2_classifier = GaussianMixture(n_components=c2_comps, covariance_type='diag', max_iter=20)
	c2_kmeans = KMeans(n_clusters=c2_comps,random_state=0).fit(c2_X_train)


	c3_classifier = GaussianMixture(n_components=c3_comps, covariance_type='diag', max_iter=20)
	c3_kmeans = KMeans(n_clusters=c3_comps,random_state=0).fit(c3_X_train)

	c4_classifier = GaussianMixture(n_components=c4_comps, covariance_type='diag', max_iter=20)
	c4_kmeans = KMeans(n_clusters=c4_comps,random_state=0).fit(c4_X_train)

	c5_classifier = GaussianMixture(n_components=c5_comps, covariance_type='diag', max_iter=20)
	c5_kmeans = KMeans(n_clusters=c5_comps,random_state=0).fit(c5_X_train)

	c6_classifier = GaussianMixture(n_components=c6_comps, covariance_type='diag', max_iter=20)
	c6_kmeans = KMeans(n_clusters=c6_comps,random_state=0).fit(c6_X_train)

	c7_classifier = GaussianMixture(n_components=c7_comps, covariance_type='diag', max_iter=20)
	c7_kmeans = KMeans(n_clusters=c7_comps,random_state=0).fit(c7_X_train)


	# Since we have class labels for the training data, we can
	# initialize the GaussianMixture parameters in a supervised manner.

	c1_classifier.means_ = np.array([c1_X_train[c1_kmeans.labels_==i].mean(axis=0) for i in xrange(c1_comps)])
	# Train the other parameters using the EM algorithm.
	c1_classifier.fit(c1_X_train)
	

	# Display the GaussianMixture of c1 
	# ax = fig.add_subplot(231, projection='3d')
	# ax.scatter(c1_X_train[c1_y_train_pred==0][:, 0], c1_X_train[c1_y_train_pred==0][:, 1], c1_X_train[c1_y_train_pred==0][:, 2], color='r',label='red')
	# ax.scatter(c1_X_train[c1_y_train_pred==1][:, 0], c1_X_train[c1_y_train_pred==1][:, 1], c1_X_train[c1_y_train_pred==1][:, 2], color='b',label='red')
	# plt.show()
	

	c2_classifier.means_ = np.array([c2_X_train[c2_kmeans.labels_==i].mean(axis=0) for i in xrange(c2_comps)])
	# Train the other parameters using the EM algorithm.
	c2_classifier.fit(c2_X_train)
	
	c3_classifier.means_ = np.array([c3_X_train[c3_kmeans.labels_==i].mean(axis=0) for i in xrange(c3_comps)])
	# Train the other parameters using the EM algorithm.
	c3_classifier.fit(c3_X_train)



	c4_classifier.means_ = np.array([c4_X_train[c4_kmeans.labels_==i].mean(axis=0) for i in xrange(c4_comps)])
	# Train the other parameters using the EM algorithm.
	c4_classifier.fit(c4_X_train)


	c5_classifier.means_ = np.array([c5_X_train[c5_kmeans.labels_==i].mean(axis=0) for i in xrange(c5_comps)])
	# Train the other parameters using the EM algorithm.
	c5_classifier.fit(c5_X_train)
	

	c6_classifier.means_ = np.array([c6_X_train[c6_kmeans.labels_==i].mean(axis=0) for i in xrange(c6_comps)])
	# Train the other parameters using the EM algorithm.
	c6_classifier.fit(c6_X_train)

	c7_classifier.means_ = np.array([c7_X_train[c7_kmeans.labels_==i].mean(axis=0) for i in xrange(c7_comps)])
	# Train the other parameters using the EM algorithm.
	c7_classifier.fit(c7_X_train)


	# If test_mode = 'on', check accuracy of test set
	# if test_mode == 'on': 
	# 	c1_X_train = c1_X_test
	# 	c1_y_train = c1_y_test

	c1_X_train_score_on_c1 = np.exp(c1_classifier.score_samples(c1_X_train))
	c1_X_train_score_on_c2 = np.exp(c2_classifier.score_samples(c1_X_train))
	c1_X_train_score_on_c3 = np.exp(c3_classifier.score_samples(c1_X_train))
	c1_X_train_score_on_c4 = np.exp(c4_classifier.score_samples(c1_X_train))
	# ignore yellow color as it leads to large error in choosing a red color as yellow. 
	c1_X_train_score_on_c5 = np.exp(c5_classifier.score_samples(c1_X_train))
	c1_X_train_score_on_c6 = np.exp(c6_classifier.score_samples(c1_X_train))
	c1_X_train_score_on_c7 = np.exp(c7_classifier.score_samples(c1_X_train))


	# print np.shape(np.array(c1_X_train_score_on_c1[0]))
	# Min of likelihood of c1_score_on_c1 is 8.69e-12
	print c1_X_train_score_on_c1.min(),c1_X_train_score_on_c1.max()
	print c1_X_train_score_on_c2.min(),c1_X_train_score_on_c2.max()
	print c1_X_train_score_on_c3.min(),c1_X_train_score_on_c3.max()
	print c1_X_train_score_on_c4.min(),c1_X_train_score_on_c4.max()
	print c1_X_train_score_on_c4.min(),c1_X_train_score_on_c5.max()
	print c1_X_train_score_on_c6.min(),c1_X_train_score_on_c6.max()
	print c1_X_train_score_on_c7.min(),c1_X_train_score_on_c7.max()
	# Add a row of 0,0,0.... to make sure that the returned index matches classes [1, 2, ... ,6]
	c1_zeros_row = np.zeros(np.shape(c1_X_train_score_on_c1))
	
	# c1_y_train_score_matrix = np.array([c1_zeros_row,c1_X_train_score_on_c1,c1_X_train_score_on_c2,c1_X_train_score_on_c3,c1_X_train_score_on_c4,c1_X_train_score_on_c5,c1_X_train_score_on_c6])
	c1_y_train_score_matrix = np.array([c1_zeros_row,c1_X_train_score_on_c1,c1_X_train_score_on_c2,c1_X_train_score_on_c3,c1_X_train_score_on_c4,c1_X_train_score_on_c5,c1_X_train_score_on_c6,c1_X_train_score_on_c7])
	c1_y_train_pred = np.argmax(c1_y_train_score_matrix,axis=0) 
	# print c1_y_train_pred, type(c1_y_train_pred), np.shape(c1_y_train_pred)
	# print np.unique(c1_y_train_pred)

	# Train accuracy = 94.81%
	# Test accuracy = 91.%

	c1_train_accuracy = np.mean(c1_y_train_pred == c1_y_train) * 100
	print 'Training Accuracy: ',c1_train_accuracy
	print np.mean(c1_y_train_pred==1),np.mean(c1_y_train_pred==2),np.mean(c1_y_train_pred==3),np.mean(c1_y_train_pred==4),np.mean(c1_y_train_pred==5),np.mean(c1_y_train_pred==6),np.mean(c1_y_train_pred==7)
	

	GMM_models = [c1_classifier,c2_classifier,c3_classifier,c4_classifier,c5_classifier,c6_classifier,c7_classifier]

	# Save the trained GMM_models 
	try:
		with open(GMM_model_file,'wb') as f:
			pickle.dump(GMM_models,f,pickle.HIGHEST_PROTOCOL) 
	except Exception as e:
		print 'Could not save GMM models to the GMM_model file ', GMM_model_file,e
		raise 
		return GMM_models
	print '--------------- Completed Training GMM models! ----------------'
	return [c1_classifier,c2_classifier,c3_classifier,c4_classifier,c5_classifier,c6_classifier,c7_classifier]


def visual_color_distribution(color_space='hsv',color_ids=[1],show='on'):
	[train_set,target_set,ids_set] = load_train_data(color_space)
	fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	subplot_ids = {1:331,2:332,3:333,4:334,5:335,6:336, 7:337}
	colors = cm.rainbow(np.linspace(0, 1, len(color_ids)+1))
	for id in color_ids: 
		ax = fig.add_subplot(subplot_ids[id], projection='3d')
		color_data =  train_set[target_set==id]
		ax.scatter(color_data[:,0],color_data[:,1],color_data[:,2], c=colors[id], marker='.')
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		title = 'Color ' + str(id) 
		ax.set_title(title) 

	image_file = '/home/tynguyen/cis650_projects/project1/figures/' + color_space + '.png'
	fig.savefig(image_file)   # save the figure to file
	
	if show == 'off':
		plt.close(fig)    # close the figure
	else: 
		plt.show()

def segment_colors(img,display='off',colorspace='hsv'):
	"""Image is in BRG format"""
	# Check whether a trained GMM exists. If there is a file entitled trained_GMM.pickle, it means that
	# there exists. We just need to load it. Otherwise, we have to train from the beginning 
	GMM_model_file = 'trained_GMM.pickle'
	if os.path.exists(GMM_model_file):
		print '- There exists a trained GMM_model! Loading it....'
		try:
			with open(GMM_model_file,'rb') as f:
				GMM_models =pickle.load(f) 
				[c1_classifier,c2_classifier,c3_classifier,c4_classifier,c5_classifier,c6_classifier,c7_classifier] = GMM_models
		except Exception as e:
			print 'Could not load the GMM_model file ', GMM_model_file,e
			raise 
	else: 
		print '- There is NO any trained GMM_model! Start training...'
		[c1_classifier,c2_classifier,c3_classifier,c4_classifier,c5_classifier,c6_classifier,c7_classifier] = train_GaussianMixtures(GMM_model_file)


	
	# Change color space 
	if colorspace == 'RGB':
		transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	elif colorspace == 'YCrCb':
		transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	elif colorspace == 'LUV':
		transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
	elif colorspace == 'HLS':
		transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	else:
		transformed_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Load image to a test array. 
	nx, ny, nz = np.shape(transformed_img)
	X_test = np.ones((nx*ny,3)) # (nx*ny) x 3 array containing pixels' color values 
	X_test_pixels = np.ones((nx*ny,2),dtype=np.uint32) # (nx*ny) x 2 array containing pixels' coordinate 
	
	for i in range(nx):
		for j in range(ny):
			X_test[i*ny+j] = transformed_img[i,j]
			X_test_pixels[i*ny+j] = [i,j]
 	
	# 
	X_test_score_on_c1 = np.exp(c1_classifier.score_samples(X_test))
	X_test_score_on_c2 = np.exp(c2_classifier.score_samples(X_test))
	X_test_score_on_c3 = np.exp(c3_classifier.score_samples(X_test))
	X_test_score_on_c4 = np.exp(c4_classifier.score_samples(X_test))
	X_test_score_on_c5 = np.exp(c5_classifier.score_samples(X_test))
	X_test_score_on_c6 = np.exp(c6_classifier.score_samples(X_test))
	X_test_score_on_c7 = np.exp(c7_classifier.score_samples(X_test))

	
	# Min of likelihood of c1_score_on_c1 is 8.69e-12
	print X_test_score_on_c1.min(),X_test_score_on_c1.max()
	print X_test_score_on_c2.min(),X_test_score_on_c2.max()
	print X_test_score_on_c3.min(),X_test_score_on_c3.max()
	print X_test_score_on_c4.min(),X_test_score_on_c4.max()
	print X_test_score_on_c5.min(),X_test_score_on_c5.max()
	print X_test_score_on_c6.min(),X_test_score_on_c6.max()
	print X_test_score_on_c7.min(),X_test_score_on_c7.max()

	# Add a row of 0,0,0.... to make sure that the returned index matches classes [1, 2, ... ,6]
	# We can set a threshold here so that if the likelihood smaller than this threshold, just get rid of. 
	zeros_row = np.zeros(np.shape(X_test_score_on_c1))
	min_red_likelihood_row = np.zeros(np.shape(X_test_score_on_c1)) + RED_LIKELIHOOD_TOL
	
	y_test_score_matrix = np.array([min_red_likelihood_row,X_test_score_on_c1,X_test_score_on_c2,X_test_score_on_c3,zeros_row,X_test_score_on_c5,X_test_score_on_c6,X_test_score_on_c7])	
	y_test_pred = np.argmax(y_test_score_matrix,axis=0) 		
	print '>>>>Number of red prediction:', sum(y_test_pred==1)

	img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

	# Transform to grayscale images: comment the following lines 
	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img_gray[:,:] = 0
	c1_locate_pixels = X_test_pixels[y_test_pred==1]
	img_gray[c1_locate_pixels[:,0],c1_locate_pixels[:,1]] = 1


	# Transform to RGB images: comment the following lines 
	img[:,:] = (0,0,0)
	c1_locate_pixels = X_test_pixels[y_test_pred==1]
	img[c1_locate_pixels[:,0],c1_locate_pixels[:,1]] = (153,0,0) # red 
	c2_locate_pixels = X_test_pixels[y_test_pred==2]
	img[c2_locate_pixels[:,0],c2_locate_pixels[:,1]] = (255,153,153) # pink 
	c3_locate_pixels = X_test_pixels[y_test_pred==3] 
	img[c3_locate_pixels[:,0],c3_locate_pixels[:,1]] = (51,0,0) # dark red 
	c4_locate_pixels = X_test_pixels[y_test_pred==4]
	img[c4_locate_pixels[:,0],c4_locate_pixels[:,1]] = (102,102,155) # blue #   
	c5_locate_pixels = X_test_pixels[y_test_pred==5]
	img[c5_locate_pixels[:,0],c5_locate_pixels[:,1]] = (255,255,0) # yellow 
	c6_locate_pixels = X_test_pixels[y_test_pred==6] 
	img[c6_locate_pixels[:,0],c6_locate_pixels[:,1]] = (64,64,64) # gray 
	c7_locate_pixels = X_test_pixels[y_test_pred==7] 
	img[c7_locate_pixels[:,0],c7_locate_pixels[:,1]] = (0,154,0) # green 

	# Display images 
	if display == 'on':
		fig = plt.figure()
		ax1 = fig.add_subplot(121)
		plt.imshow(img_gray,aspect='auto',cmap='gray')
		ax1.set_title('Raw segmented image with only barrel (white)')
		ax2 =fig.add_subplot(122)
		plt.imshow(img,aspect='auto')
		ax2.set_title('Segmented image in RBG')
		plt.show()
	return img_gray

def order_points(box):
	"""Copyright: http://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/"""
	# sort the points based on their x-coordinates
	xSorted = box[np.argsort(box[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([bl,tl, tr, br], dtype="float32")

def dist2Camera(w,h):
	"""Find the distance of the object bounded by a box to the camera"""
	K = np.array([[413.8794311523438, 0.0, 667.1858329372201], 
		[0.0, 415.8518917846679, 500.6681745269121], [0.0, 0.0, 1.0]],dtype=np.float32)
	X = 40 # diameter
	Y = 57 # tall 
	# Vector representing the position of barrel in pixel space 
	P_pixel = np.array([w,h,1])
	# Vector representing the position of barrel in metrix space after normalizing Z 
	P_metric = np.dot(inv(K),P_pixel)
	# Distance 
	d1 = -X/P_metric[0]
	d2 = -Y/P_metric[1]
	print 'w,h: ', w, h 
	print 'Found distance: ', max(d1,d2)
	return max(d1,d2)


def pixel2DistModel(X,y):
	num_data = np.shape(X)[0]
	X_train = X[0:int(0.8*num_data),:]
	X_test = X[int(0.8*num_data):num_data,:]
	y_train = y[0:int(0.8*num_data)]
	y_test = y[int(0.8*num_data):num_data,:]

	reg_model = linear_model.LinearRegression()
	reg_model.fit(X_train,  y_train)

	model_file = 'pixel2DistModel.pickle'
	try:
		with open(model_file,'wb') as f:
			pickle.dump(reg_model,f,pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print 'Could not save pixel2DistModel to file', model_file, e 
		raise  

	print 'Coeff:', reg_model.coef_
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(X_test[:,0],X_test[:,1],y_test,color='black')
	ax.plot_surface(X_test[:,0],X_test[:,1],reg_model.predict(X_test))
	plt.show()

def predictDistFromPixel(w,h):
	model_file = 'pixel2DistModel.pickle'
	try:
		with open(model_file,'rb') as f:
			reg_model = pickle.load(f)
	except Exception as e:
		print 'Could not load pixel2DistModel from file', model_file, e 
		raise  
	X = np.array([w,h])
	X.reshape(-1,1)
	y_test_pred  = abs(reg_model.predict([w,h])[0])
	print 'predicted value:', y_test_pred
	return y_test_pred

def display_bouncing_box(img,image_name):
	# Denoise image 
	img_denoised = cv2.fastNlMeansDenoisingColored(img,None,5,5,3,5)
	img_RGB = cv2.cvtColor(img_denoised,cv2.COLOR_BGR2RGB)
	img_width, img_height, _ = np.shape(img_denoised)
	
	print '--------------Opening image.......------------------ ' 

	# Segment colors and get gray image with only barrel pixels (in white)
	img = segment_colors(img_denoised,display='off')
	# Apply Eroding and Dilating to enhance contours 
	se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
	se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (16,16))
	mask1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
	mask2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, se2)
	out = img * mask1
	out = out*mask2 


	# Find contours 
	contours, hierarchy = cv2.findContours(out,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	areas = np.array([cv2.contourArea(c) for c in contours])
	# Remove small contours 	
	fine_countours = np.array(contours)[areas >= AREA_TOL]
	max_area_found = max(areas)
	area_tol = MIN_AREA_TOL
	while len(fine_countours) == 0 and area_tol >= MIN_AREA_TOL:
		# reduce the threshold 
		area_tol -= 100 
		fine_countours = np.array(contours)[areas >= area_tol]
	if len(fine_countours) == 0:
		print '--- There is no Barrel as all contour are two small!'
		plt.show()
		return 0 
	# Eliminate some small contours based on the largest contounr
	large_countours = np.array(contours)[areas >= max_area_found*0.2]
	large_areas = np.array([cv2.contourArea(c) for c in large_countours])
	num_large_contours = len(large_countours)


	# The following procedure will eliminate contours with small density 
	decent_boxs = []
	decent_pix_dens = [] 
	decent_contours = [] 
	decent_rects = [] 
	
	
	print '--- Num of contours to consider density: ', num_large_contours
	max_index = np.argmax(areas)
	index = -1 
	for c in large_countours:
		index += 1 
		print '-----contour number:',index 
		# rectangle that bounds the contour 
		# rect := (x,y),(w,h),theta.
		# (x,y) - center point of the box
		# (w,h) - width and height of the box
		# theta - angle of rotation
		rect = cv2.minAreaRect(c) 
		# BoxPoints(rect) return 4 points defining the rotated rectangle provided to it.
			# In opencv points are plotted as (x,y) not (row,column), and the y axis is positive downward. 
			# So the first point would be plotted x pixels right of the left side of the image and y 
			# pixels down from the top of the image. In other words, the first point is the bottom left point of the image.
		box = cv2.cv.BoxPoints(rect)
		box = order_points(np.array(box))
		box = np.int0(box)

		# Compute density of white pixels in the contour. If it < DENS_TOL, ignore it! 
		pix_density = large_areas[index]/(rect[1][0]*rect[1][1])
		print 'pix_density is: ', pix_density 
		if pix_density < SMALL_DENS_TOL or (num_large_contours > 1 and pix_density < BIG_DENS_TOL):
			print 'Pix density is too small.', pix_density, '<', BIG_DENS_TOL 
			continue 
		print '            Accept! '
		decent_contours.append(c)
		decent_boxs.append(box)
		decent_pix_dens.append(pix_density)
		decent_rects.append(rect)

	# In the final step, we will get final_boxs and final_contours by concatenating two nearby contours if they are too close
	# and at the same level (row or column)
	final_contours = []
	final_boxs = []
	final_widths = []
	final_heights = []
	final_centers = [] 
	num_decent_contours = len(decent_contours)
	print '--- Number of decent contours to consider combination:', num_decent_contours
	last_box_concatenated = False 
	if num_decent_contours >= 2:
		for index in range(num_decent_contours-1):
			center_i = decent_rects[index][0]
			box_i = decent_boxs[index]
			for j in range(index+1, num_decent_contours):
				center_j = decent_rects[j][1]
				box_j = decent_boxs[j]
				print '-----Percent distance in row:',abs(box_i[3][0]-box_j[0][0])/decent_rects[j][1][0] , abs(box_j[3][0]-box_i[0][0])/decent_rects[j][1][0]
				print '-----Percent distance in col:',abs(box_i[1][1]-box_j[0][1])/decent_rects[j][1][1], abs(box_j[1][1]-box_i[0][1])/decent_rects[j][1][1] 
				# print 'Center index:', center_i
				# print 'Center j:', center_j 
				# Concatenate in row: i to j 
				if abs(box_i[3][0]-box_j[0][0])*1.0/decent_rects[j][1][0] < 0.25 and (center_i[0]-center_j[0]) <= 0:
					print 'Concatenate in row: i to j '
					box = [box_i[0], box_i[1], box_j[2], box_j[3]]
					final_boxs.append(box)
					final_centers.append(np.mean([center_i,center_j],axis=0))
					final_heights.append(abs(box[0][1]-box[1][1])) # same height
					final_widths.append(abs(box[0][0]-box[2][0])) # increase width 
					if j == num_decent_contours-1:
						last_box_concatenated = True 
				# Concatenate in row: j to i 					
				elif abs(box_j[3][0]-box_i[0][0])/decent_rects[j][1][0] < 0.25 and (center_j[0]-center_i[0]) <= 0:
					# print 'Concatenate in row: j to i '
					# print 'box i:', box_i
					# print 'box j:', box_j 
					box = [box_j[0], box_j[1], box_i[2], box_i[3]]
					final_boxs.append(box)
					final_centers.append(np.mean([center_i,center_j],axis=0))
					final_heights.append(abs(box[0][1]-box[1][1])) # same height
					final_widths.append(abs(box[0][0]-box[2][0])) # increase width 
					if j == num_decent_contours-1:
						last_box_concatenated = True 
				# Concatenate in col: i to j 
				elif abs(box_i[1][1]-box_j[0][1])/decent_rects[j][1][1]  < 0.25  and (center_j[1]-center_i[1]) <= 0 and abs(box_i[1][0]-box_j[0][0])/decent_rects[j][1][0]  < 0.25:
					print 'Concatenate in col: i to j '
					box = [box_i[0], box_j[1], box_j[2], box_i[3]]
					final_boxs.append(box)
					final_centers.append(np.mean([center_i,center_j],axis=0))
					final_heights.append(abs(box[0][1]-box[1][1])) # increase height
					final_widths.append(abs(box[0][0]-box[2][0])) # same width 
					if j == num_decent_contours-1:
						last_box_concatenated = True 
				# Concatenate in col: j to i 
				elif abs(box_j[1][1]-box_i[0][1])/decent_rects[j][1][1]  < 0.25  and (center_i[1]-center_j[1]) <= 0 and  abs(box_j[1][0]-box_i[0][0])/decent_rects[j][1][0]  < 0.25:
					print 'Concatenate in col: j to i '
					box = [box_j[0], box_i[1], box_i[2], box_j[3]]
					final_boxs.append(box)
					final_centers.append(np.mean([center_i,center_j],axis=0))
					final_heights.append(abs(box[0][1]-box[1][1])) # increase height
					final_widths.append(abs(box[0][0]-box[2][0])) # same width 
					if j == num_decent_contours-1:
						last_box_concatenated = True 
				else:
					final_boxs.append(decent_boxs[index])
					rect = decent_rects[index] 
					final_widths.append(rect[1][0])
					final_heights.append(rect[1][1])
					final_centers.append(rect[0])
	if not last_box_concatenated:
		box = decent_boxs[num_decent_contours-1]
		final_boxs.append(box)
		rect = decent_rects[num_decent_contours-1] 
		final_widths.append(rect[1][0])
		final_heights.append(rect[1][1])
		final_centers.append(rect[0])
	print '---Final Boxes: '
	print final_boxs
	print final_widths
	print final_heights
	final_result = [] 
	final_title = 'Distance of the barrel to the camera: ' 
	for index in range(len(final_boxs)):
		w = final_widths[index]
		h = final_heights[index]
		d = predictDistFromPixel(w,h)
		print '--- Distance to the camera: ', d
		final_title += str(d)
		box = final_boxs[index]
		box = np.array(box)
		final_result.append([box,d])
		cv2.drawContours(img_RGB,[box],0,(0,255,0),4)
	print 'Number of barrel: ', len(final_boxs)

	# Display segmented image in gray 
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	plt.imshow(img,aspect='auto',cmap='gray')
	ax1.set_title('Segmented Image in gray',fontsize= 10)
	ax2 = fig.add_subplot(122)
	plt.imshow(img_RGB,aspect='auto')
	ax2.set_title(final_title,fontsize= 10)
	# plt.show()


	# Fig result 
	final_fig_folder = 'final_figures'
	if not os.path.exists(final_fig_folder):
		os.makedirs(final_fig_folder)
	image_file = os.path.join(final_fig_folder, image_name  + '.png') 
	fig.savefig(image_file)   # save the figure to file

	return np.array(final_result)	




		


if __name__ == '__main__':
	# train_GaussianMixtures('hsv',test_mode='off')
	# display_segmented_image('hsv',segmented_train_img_folder = "data/segmented_images_all_7")
	display_bouncing_box(img,img_name)
	# load_train_data()
	# visual_color_distribution('RGB',[1,2,3,4,5,6,7],show='off')
	# visual_color_distribution('LUV',[1,2,3,4,5,6,7],show='off')
	# visual_color_distribution('YCrCb',[1,2,3,4,5,6,7],show='off')
	# visual_color_distribution('HLS',[1,2,3,4,5,6,7],show='off')

	# #  Test pixel2DistModel() 
	# X_train = [[199,201],[144,120],[103,48], [44, 35], [144,120],[80,60],[200,189],[300,250],[150,120],[220,180]]
	# y_train = [[2],[1.8],[1.7],[20],[1.7],[2.2],[0.8], [0.6],[1.4],[0.9]]
	# X_train = np.array(X_train)
	# y_train = np.array(y_train)
	# pixel2DistModel(X_train,y_train)

	# # Test predictDistFromPixel() 
	# w = 60 
	# h = 40 
	# predictDistFromPixel(w,h)
	
