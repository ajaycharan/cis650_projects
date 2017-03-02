
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
BIG_DENS_TOL = 0.715# hard threshold of pixel density in a contour to be a barrel 
SMALL_DENS_TOL = 0.4 # soft threshold of pixel density in a contour to be a barrel 

class DATA:
	def __init__(self,data_set,target_set,ids_set):
		self.data = data_set
		self.target = target_set
		self.ids = ids_set


def load_train_data(transform_color='on',colorspace='hsv'):
	image_folder = "data/Proj1_Train"
	image_files = os.listdir(image_folder)
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

	# # Test segmented image by visualization 
	# for image_name in image_files:
	# 	# image_name = image_files[0] 
	# 	command = raw_input("continue display?")
	# 	if command == 'N' or command == 'n':
	# 		break 
	# 	image_file = os.path.join(image_folder,image_name)
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
	for image_name in image_files:
		# command = raw_input("continue display?")
		# if command == 'N' or command == 'n':
		# 	break 
		image_file = os.path.join(image_folder,image_name)
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
		print segment_pixels[0]
		print np.shape(segment_pixels)
		train_set[ids_set==image_name] = img[segment_pixels[:,0], segment_pixels[:,1]]
		
		# # For debugging: show segmented image
		# img[segment_pixels[:,0], segment_pixels[:,1]] = (255,255,255)	
		# plt.imshow(img,aspect='auto')
		# plt.show()



	return [train_set,target_set,ids_set]

def tune_GaussianMixtures(color_space='hsv',test_mode='off'):
	iris = datasets.load_iris()
	[train_set,target_set,ids_set] = load_train_data()
	print np.shape(train_set)
	n_classes = len(np.unique(target_set))
	print 'number of classes:',n_classes

	for c in np.unique(target_set):
		print 'class ', c 
		print len(target_set[target_set == c])

	# exit()
	# # get red color only 
	# red_train = train_set[target_set == 1] #[0:1000]
	# red_target = target_set[target_set == 1]#[0:1000]
	# red_ids = ids_set[target_set == 1] # [0:1000]
	# data_set = DATA(red_train,red_target,red_ids)

	data_set = DATA(train_set,target_set,ids_set)

	print np.shape(data_set.data)
	print np.shape(data_set.target)
	print np.shape(data_set.target)
	print sum(data_set.target)

	# exit()
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

	print  'len c2 train:',len(c2_X_train)
	print 'len c2 test:', len(c2_y_test)

	# Number of components for each class 
	c1_comps = 2
	c2_comps = 2 
	c3_comps = 2 
	c4_comps = 2
	c5_comps = 2
	c6_comps = 3
	c7_comps = 2 

	# c2_classifiers = dict((covar_type, GaussianMixture(n_components=c2_comps,
	#                     covariance_type=covar_type, init_params='kmeans', max_iter=20))
	#                    for covar_type in ['spherical', 'diag', 'tied','full'])
	# apply Kmeans to find the means of components of each class 
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
	if test_mode == 'on': 
		c1_X_train = c1_X_test
		c1_y_train = c1_y_test

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

def segment_training_images(display='on',colorspace='hsv',segmented_image_folder = "data/segmented_images"):
	image_folder = "data/Proj1_Train"
	if not os.path.exists(segmented_image_folder):
		os.makedirs(segmented_image_folder)
	image_files = os.listdir(image_folder)
	# [train_set_pixels,target_set,ids_set] = load_train_data(transform_color='off')
	[c1_classifier,c2_classifier,c3_classifier,c4_classifier,c5_classifier,c6_classifier,c7_classifier] = tune_GaussianMixtures()
	count = 1 

	for image_name in image_files:	
		fig = plt.figure()
		print '%d. Image %s ---------------'%(count, image_name)
		count +=1 
		image_file = os.path.join(image_folder,image_name)
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

		nx, ny, nz = np.shape(img)
		# X_train = np.array([x for x in img[:,2]])
		X_train = np.ones((nx*ny,3))
		X_train_pixels = np.ones((nx*ny,2),dtype=np.uint32)
		
		for i in range(nx):
			for j in range(ny):
				X_train[i*ny+j] = img[i,j]
				X_train_pixels[i*ny+j] = [i,j]

		# x, y = np.meshgrid(np.arange(nx), np.arange(ny))
		# x, y = x.flatten(), y.flatten()
		# X_train_pixels = np.vstack((x,y)).T
		# y_train = np.ones((1,num_pixels))
		# index = 0 
		# for x in X_train_pixels:
		# 	print 'pixel: ', x 
		# 	same_image_indices = np.where(ids_set==image_name)[0]
		# 	print 'Possible pixels in train_set:', train_set_pixels[same_image_indices]
		# 	same_pixels_indices = np.where((train_set_pixels==x).all(axis=1))[0]
		# 	# print same_pixels_indices
		# 	# print same_image_indices
		# 	# print sum(np.in1d(same_image_indices,same_pixels_indices))
		# 	intersect_index = same_image_indices[np.in1d(same_image_indices,same_pixels_indices)]
		# 	print intersect_index
		# 	if len(intersect_index) == 0:
		# 		y_train = 0 
		# 		continue 
		# 	print 'Intersect index: ', intersect_index
		# 	print '-Where image is: ', image_name, ids_set[intersect_index]
		# 	print '- And pixels are: ', x, train_set[intersect_index]
		# 	y_train[index] = target_set[intersect_index]
		
		
		X_train_score_on_c1 = np.exp(c1_classifier.score_samples(X_train))
		X_train_score_on_c2 = np.exp(c2_classifier.score_samples(X_train))
		X_train_score_on_c3 = np.exp(c3_classifier.score_samples(X_train))
		X_train_score_on_c4 = np.exp(c4_classifier.score_samples(X_train))
		X_train_score_on_c5 = np.exp(c5_classifier.score_samples(X_train))
		X_train_score_on_c6 = np.exp(c6_classifier.score_samples(X_train))
		X_train_score_on_c7 = np.exp(c7_classifier.score_samples(X_train))

		
		# Min of likelihood of c1_score_on_c1 is 8.69e-12
		print X_train_score_on_c1.min(),X_train_score_on_c1.max()
		print X_train_score_on_c2.min(),X_train_score_on_c2.max()
		print X_train_score_on_c3.min(),X_train_score_on_c3.max()
		print X_train_score_on_c4.min(),X_train_score_on_c4.max()
		print X_train_score_on_c5.min(),X_train_score_on_c5.max()
		print X_train_score_on_c6.min(),X_train_score_on_c6.max()
		print X_train_score_on_c7.min(),X_train_score_on_c7.max()

		# Add a row of 0,0,0.... to make sure that the returned index matches classes [1, 2, ... ,6]
		# We can set a threshold here so that if the likelihood smaller than this threshold, just get rid of. 
		zeros_row = np.zeros(np.shape(X_train_score_on_c1)) + RED_LIKELIHOOD_TOL
		
		# y_train_score_matrix = np.array([zeros_row,X_train_score_on_c1,X_train_score_on_c2,X_train_score_on_c3,X_train_score_on_c4,X_train_score_on_c5,X_train_score_on_c6])
		y_train_score_matrix = np.array([zeros_row,X_train_score_on_c1,X_train_score_on_c2,X_train_score_on_c3,X_train_score_on_c5,X_train_score_on_c6,X_train_score_on_c7])	
		y_train_pred = np.argmax(y_train_score_matrix,axis=0) 		
		print '>>>>Number of red prediction:', sum(y_train_pred==1)

		img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

		# Transform to grayscale images: comment the following lines 
		img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img_gray[:,:] = 0
		c1_locate_pixels = X_train_pixels[y_train_pred==1]
		img_gray[c1_locate_pixels[:,0],c1_locate_pixels[:,1]] = 1


		# Transform to RGB images: comment the following lines 
		img[:,:] = (0,0,0)
		c1_locate_pixels = X_train_pixels[y_train_pred==1]
		img[c1_locate_pixels[:,0],c1_locate_pixels[:,1]] = (153,0,0) # red 
		c2_locate_pixels = X_train_pixels[y_train_pred==2]
		img[c2_locate_pixels[:,0],c2_locate_pixels[:,1]] = (255,153,153) # pink 
		c3_locate_pixels = X_train_pixels[y_train_pred==3] 
		img[c3_locate_pixels[:,0],c3_locate_pixels[:,1]] = (51,0,0) # dark red 
		c4_locate_pixels = X_train_pixels[y_train_pred==4]
		img[c4_locate_pixels[:,0],c4_locate_pixels[:,1]] = (102,102,155) # blue #  (0,154,0) # green  
		c5_locate_pixels = X_train_pixels[y_train_pred==5]
		img[c5_locate_pixels[:,0],c5_locate_pixels[:,1]] = (255,255,0) # yellow 
		c6_locate_pixels = X_train_pixels[y_train_pred==6] 
		img[c6_locate_pixels[:,0],c6_locate_pixels[:,1]] = (64,64,64) # gray 

		# Display images 
		if display == 'on':
			fig.add_subplot(121)
			plt.imshow(img_gray,aspect='auto',cmap='gray')
			fig.add_subplot(122)
			plt.imshow(img,aspect='auto')
			plt.show()
			command = raw_input("continue display?")
			if command == 'N' or command == 'n':
				break 

		# save images to pickle files 
		segmented_image_file = image_name + '.pickle'
		segmented_image_file = os.path.join(segmented_image_folder,segmented_image_file) 
		try: 
			with open(segmented_image_file,'wb') as f:
				pickle.dump(img_gray,f,pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to', segmented_image_file, ':', e)
		print '--- Complete saving segmented images to folder', segmented_image_folder

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
	y_test_pred = reg_model.predict(X_test)
	y_test_pred = np.array([abs(y) for y in y_test_pred])	
	print 'Test y:',y_test
	print 'Predicted y:',y_test_pred
	error = np.sqrt(np.sum((y_test_pred-y_test)**2))/y_test_pred.shape[0]
	print 'Error is: %.3f'%(error)
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(X_test[:,0],X_test[:,1],y_test,color='black')
	ax.plot_surface(X_test[:,0],X_test[:,1],y_test_pred)
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
	print 'predicted value:', reg_model.predict(X)[0], np.shape(reg_model.predict(X))
	return reg_model.predict([w,h])[0]

def train_distance_model(display='on',segmented_image_folder = "data/segmented_images"):
	"""Train a model that represent the correlation between barrel's pixels and its distance
	   to camera"""
	image_folder = "data/Proj1_Train"
	count = 0 
	X_train = []
	y_train = [] 
	for segmented_file_name in os.listdir(segmented_image_folder):
		image_name = segmented_file_name.strip('.pickle')
		image_file = os.path.join(image_folder,image_name)
		orig_image = cv2.imread(image_file)
		orig_image = cv2.cvtColor(orig_image,cv2.COLOR_BGR2RGB)
		image_width, image_height, _ = np.shape(orig_image)
		
		# count += 1 
		# if count == 50:
		# 	break 
		# command = raw_input("continue display?")
		# if command == 'N' or command == 'n':
		# 	break 
		print '-- Opening image ', segmented_file_name
		segmented_file = os.path.join(segmented_image_folder,segmented_file_name)
		try:
			with open(segmented_file, 'rb') as f:
				img = pickle.load(f)
		except Exception as e:
			print('Unable to process data from', segmented_file, ':', e)
			raise

		# fig = plt.figure()
		# ax = fig.add_subplot(121)
		# plt.imshow(img,aspect='auto',cmap='gray')
		# ax.set_title('Before eroding and dilating')
		

		se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
		se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (16,16))
		mask1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
		mask2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, se2)

		dil_rod_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
		dil_rod_img= cv2.morphologyEx(dil_rod_img, cv2.MORPH_OPEN, se2)


		# ax = fig.add_subplot(122)
		# plt.imshow(mask2,aspect='auto',cmap='gray')
		# ax.set_title('After eroding and dilating')
		# plt.show()
		# command = raw_input("continue display?")
		# if command == 'N' or command == 'n':
		# 	break

		out = img * mask1
		out = out*mask2 

		

		# out = util.img_as_ubyte(out)  # > 110
		# label_img = label(out, connectivity=out.ndim)
		# print 'Number of labels:', np.max(skimage.measure.label(out,background=0))
		# props_list = skimage.measure.regionprops(label_img)
		# print 'Number of props: ', len(props_list)

		# fine_props_list = []
		# for props in props_list:
		# 	print props.area, props['centroid']
		# 	print props.bbox
		# 	if props.area > AREA_TOL:
		# 		fine_props_list.append(props)
		# 		# display contour 
		# 		contour = skimage.measure.find_contours(label_img, 1, fully_connected='low', positive_orientation='low')
		# 		print 'contour:', contour, len(contour)
		# if not fine_props_list:
		# 	print 'There is no barrel found!'
		# 	return ['There is no barrel found!']

		contours, hierarchy = cv2.findContours(out,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		# Find the index of the largest contour
		# 1. Set contour area min
		# 2. Set the maximum number of barrels: 2 
		# 3. Eliminate barrels that are close to the rear of images 
		# 4. The way we extend rect box as following: if w/h > constant => extend h and vice versal! 
		# 5. The last thing that I will do is to create one more model for the green color and truck blue color!
		# 6. Combine several consecutive rects into one rect 
		areas = np.array([cv2.contourArea(c) for c in contours])
		# print areas >= AREA_TOL
		
		fine_countours = np.array(contours)[areas >= AREA_TOL]
		max_area_found = max(areas)
		area_tol = MIN_AREA_TOL
		while len(fine_countours) == 0 and area_tol >= MIN_AREA_TOL:
			# reduce the threshold 
			area_tol -= 100 
			fine_countours = np.array(contours)[areas >= area_tol]
		if len(fine_countours) == 0:
			print 'There is no Barrel as all contour are two small!'
			# return 0 
		# Iliminate some small contours 
		large_countours = np.array(contours)[areas >= max_area_found*0.2]
		large_areas = np.array([cv2.contourArea(c) for c in large_countours])
		
		# The following procedure will iliminate contours with small density and/or located at extreme positions 
		decent_boxs = []
		decent_pix_dens = [] 
		decent_contours = [] 
		decent_rects = [] 
		
		num_large_contours = len(large_countours)
		print '-Num of contours to consider density and position: ', num_large_contours
		max_index = np.argmax(areas)
		index = -1 
		for c in large_countours:
			index += 1 
			print '--- contour number:',index 
			# rectangle that bounds the contour 
			# rect := (x,y),(w,h),theta.
			# (x,y) - center point of the box
			# (w,h) - width and height of the box
			# theta - angle of rotation
			rect = cv2.minAreaRect(c) 
			# BoxPoints(rect) return 4 points defining the rotated rectangle provided to it.
 			# In opencv points are plotted as (x,y) not (row,column), and the y axis is positive downward. 
 			# So the first point would be plotted 200 pixels right of the left side of the image and 472 
 			# pixels down from the top of the image. In other words, the first point is the bottom left point of the image.
			box = cv2.cv.BoxPoints(rect)
			box = order_points(np.array(box))
			# print ' 		         Box: ', box
			box = np.int0(box)

			# Compute density of white pixels in the contour. If it < DENS_TOL, ignore it! 
			pix_density = large_areas[index]/(rect[1][0]*rect[1][1])
			print 'pix_density is: ', pix_density 
			if pix_density < SMALL_DENS_TOL or (num_large_contours > 1 and pix_density < BIG_DENS_TOL):
				print 'Pix density is too small.', pix_density, '<', BIG_DENS_TOL 
				# np.delete(fine_countours,index)
				continue 
			
			# # Eliminate if box is at extreme cases 
			# LB,LT,RT,RB = box 
			# if LB[0] < 0.04*image_width:
			# 	print 'Contour is too close to the left'
			# 	continue 
			# if LT[1] < 0.04*image_height:
			# 	print 'Contour is too close to the top'
			# 	continue 
			# if RT[0] > 0.96*image_width:
			# 	print 'Contour is too close to the right'
			# 	continue 
			# if RB[1] > 0.96*image_height:
			# 	print 'Contour is too close to the bottom'
			# 	continue 

			print '            Accept! '
			# print ' 		         Box: ', box
			decent_contours.append(c)
			decent_boxs.append(box)
			decent_pix_dens.append(pix_density)
			decent_rects.append(rect)

		# In the final step, we will get final_boxs and final_contours 
		# final_contours = decent_contours
		# final_boxs = decent_boxs
		# num_decent_contours = len(decent_contours)
		print 'Decent Box: ', decent_boxs
		final_contours = []
		final_boxs = []
		final_widths = []
		final_heights = []
		final_centers = [] 
		num_decent_contours = len(decent_contours)
		print 'Number of decent contours to consider:', num_decent_contours
		# If there is more than one box and there areas are too different, ignore one 
		# If there is more than one box and they are close together enough, concatenate them
		last_box_concatenated = False 
		if num_decent_contours >= 2:
			for index in range(num_decent_contours-1):
				center_i = decent_rects[index][0]
				box_i = decent_boxs[index]
				for j in range(index+1, num_decent_contours):
					center_j = decent_rects[j][1]
					box_j = decent_boxs[j]
					print 'Percent distance in row:',abs(box_i[3][0]-box_j[0][0])/decent_rects[j][1][0] , abs(box_j[3][0]-box_i[0][0])/decent_rects[j][1][0]
					print 'Percent distance in col:',abs(box_i[1][1]-box_j[0][1])/decent_rects[j][1][1], abs(box_j[1][1]-box_i[0][1])/decent_rects[j][1][1] 
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
					elif abs(box_i[1][1]-box_j[0][1])/decent_rects[j][1][1]  < 0.25  and (center_j[1]-center_i[1]) <= 0:
						print 'Concatenate in col: i to j '
						box = [box_i[0], box_j[1], box_j[2], box_i[3]]
						final_boxs.append(box)
						final_centers.append(np.mean([center_i,center_j],axis=0))
						final_heights.append(abs(box[0][1]-box[1][1])) # increase height
						final_widths.append(abs(box[0][0]-box[2][0])) # same width 
						if j == num_decent_contours-1:
							last_box_concatenated = True 
					# Concatenate in col: j to i 
					elif abs(box_j[1][1]-box_i[0][1])/decent_rects[j][1][1]  < 0.25  and (center_i[1]-center_j[1]) <= 0:
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
		print 'Final Boxes: '
		print final_boxs
		print final_widths
		print final_heights
		final_result = [] 
		for index in range(len(final_boxs)):
			w = final_widths[index]
			h = final_heights[index]
			X_train.append([w,h])
			try:
				d = image_name.strip('.png').split('_')[index]
			except:
				d =  image_name.strip('.png').split('_')[0]
			y_train.append([float(d)])
			box = final_boxs[index]
			box = np.array(box)
			final_result.append([box,d])
			cv2.drawContours(orig_image,[box],0,(0,255,0),4)
		print 'Number of barrel: ', len(final_boxs)

		if display=='on':
			fig = plt.figure()
			ax1 = fig.add_subplot(121)
			plt.imshow(dil_rod_img,aspect='auto',cmap='gray')
			ax2 = 'Segmented image'
			ax = fig.add_subplot(122)
			plt.imshow(orig_image,aspect='auto')
			ax.set_title('Bounding box')
			plt.show()
			command = raw_input("continue display?")
			if command == 'N' or command == 'n':
				break

		 

	X_train = np.array(X_train)
	y_train = np.array(y_train)
	pixel2DistModel(X_train,y_train)
	print '--- Completed training distance model!'



if __name__ == '__main__':
	# tune_GaussianMixtures('hsv',test_mode='off')
	# segment_training_images('off','hsv',segmented_image_folder = "data/segmented_images_all_7")
	train_distance_model(display='on',segmented_image_folder = "data/segmented_images_all_7")
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
	
