
#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from skimage import data, util
import skimage
from skimage.measure import label, regionprops
import numpy as np
import os 
import cv2 
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GaussianMixture
from six.moves import cPickle as pickle
import time
from sklearn.cluster import KMeans
print 'Import has been done!!'
min_red_likelihood = 7e-10
AREA_TOL = 100 # minimum pixel area of a barrel in the image 

class DATA:
	def __init__(self,data_set,target_set,ids_set):
		self.data = data_set
		self.target = target_set
		self.ids = ids_set


def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

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

	# # Test segmented image by displaying 
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

def tune_GaussianMixtures(color_space='hsv'):
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
	skf = StratifiedKFold(data_set.target, n_folds=4)
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

	print  'len c2 train:',len(c2_X_train)
	print 'len c2 test:', len(c2_y_test)

	# Number of components for each class 
	c1_comps = 2
	c2_comps = 2 
	c3_comps = 2 
	c4_comps = 2
	c5_comps = 2
	c6_comps = 3

	# c2_classifiers = dict((covar_type, GaussianMixture(n_components=c2_comps,
	#                     covariance_type=covar_type, init_params='kmeans', max_iter=20))
	#                    for covar_type in ['spherical', 'diag', 'tied','full'])
	# apply Kmeans to find the means of components of each class 
	c1_classifier = GaussianMixture(n_components=c1_comps, covariance_type='diag', init_params='kmeans', max_iter=20)	                  
	c1_kmeans = KMeans(n_clusters=c1_comps,random_state=0).fit(c1_X_train)

	c2_classifier = GaussianMixture(n_components=c2_comps, covariance_type='diag', init_params='kmeans', max_iter=20)
	c2_kmeans = KMeans(n_clusters=c2_comps,random_state=0).fit(c2_X_train)


	c3_classifier = GaussianMixture(n_components=c3_comps, covariance_type='diag', init_params='kmeans', max_iter=20)
	c3_kmeans = KMeans(n_clusters=c3_comps,random_state=0).fit(c3_X_train)

	c4_classifier = GaussianMixture(n_components=c4_comps, covariance_type='diag', init_params='kmeans', max_iter=20)
	c4_kmeans = KMeans(n_clusters=c4_comps,random_state=0).fit(c4_X_train)

	c5_classifier = GaussianMixture(n_components=c5_comps, covariance_type='diag', init_params='kmeans', max_iter=20)
	c5_kmeans = KMeans(n_clusters=c5_comps,random_state=0).fit(c5_X_train)

	c6_classifier = GaussianMixture(n_components=c6_comps, covariance_type='diag', init_params='kmeans', max_iter=20)
	c6_kmeans = KMeans(n_clusters=c6_comps,random_state=0).fit(c6_X_train)



	# print c2_kmeans.labels_
	# print len(c2_X_train[c2_kmeans.labels_==0])
	# print np.mean(c2_X_train[c2_kmeans.labels_==0],axis=0)
	# print len(c2_X_train[c2_kmeans.labels_==1])
	# print np.mean(c2_X_train[c2_kmeans.labels_==1],axis=0)
	# print np.mean(c2_X_train,axis=0)


	# fig = plt.figure()
	# plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05, left=.01, right=.99)


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


	# uncommet two following lines to check accuracy of test set 
	c1_X_train = c1_X_test
	c1_y_train = c1_y_test

	c1_X_train_score_on_c1 = np.exp(c1_classifier.score_samples(c1_X_train))
	c1_X_train_score_on_c2 = np.exp(c2_classifier.score_samples(c1_X_train))
	c1_X_train_score_on_c3 = np.exp(c3_classifier.score_samples(c1_X_train))
	c1_X_train_score_on_c4 = np.exp(c4_classifier.score_samples(c1_X_train))
	# ignore yellow color as it leads to large error in choosing a red color as yellow. 
	c1_X_train_score_on_c5 = np.exp(c5_classifier.score_samples(c1_X_train))
	c1_X_train_score_on_c6 = np.exp(c6_classifier.score_samples(c1_X_train))


	# print np.shape(np.array(c1_X_train_score_on_c1[0]))
	# Min of likelihood of c1_score_on_c1 is 8.69e-12
	print c1_X_train_score_on_c1.min(),c1_X_train_score_on_c1.max()
	print c1_X_train_score_on_c2.min(),c1_X_train_score_on_c2.max()
	print c1_X_train_score_on_c3.min(),c1_X_train_score_on_c3.max()
	print c1_X_train_score_on_c4.min(),c1_X_train_score_on_c4.max()
	print c1_X_train_score_on_c4.min(),c1_X_train_score_on_c5.max()
	print c1_X_train_score_on_c6.min(),c1_X_train_score_on_c6.max()

	# Add a row of 0,0,0.... to make sure that the returned index matches classes [1, 2, ... ,6]
	c1_zeros_row = np.zeros(np.shape(c1_X_train_score_on_c1))
	
	# c1_y_train_score_matrix = np.array([c1_zeros_row,c1_X_train_score_on_c1,c1_X_train_score_on_c2,c1_X_train_score_on_c3,c1_X_train_score_on_c4,c1_X_train_score_on_c5,c1_X_train_score_on_c6])
	c1_y_train_score_matrix = np.array([c1_zeros_row,c1_X_train_score_on_c1,c1_X_train_score_on_c2,c1_X_train_score_on_c3,c1_X_train_score_on_c4,c1_X_train_score_on_c5,c1_X_train_score_on_c6])
	c1_y_train_pred = np.argmax(c1_y_train_score_matrix,axis=0) 
	# print c1_y_train_pred, type(c1_y_train_pred), np.shape(c1_y_train_pred)
	# print np.unique(c1_y_train_pred)

	# Train accuracy = 94.81%
	# Test accuracy = 91.%

	c1_train_accuracy = np.mean(c1_y_train_pred == c1_y_train) * 100
	print 'Training Accuracy: ',c1_train_accuracy
	print np.mean(c1_y_train_pred==1),np.mean( c1_y_train_pred==2),np.mean(c1_y_train_pred==3),np.mean( c1_y_train_pred==4),np.mean( c1_y_train_pred==5),np.mean( c1_y_train_pred==6)
	
	return [c1_classifier,c2_classifier,c3_classifier,c4_classifier,c5_classifier,c6_classifier]
	exit(0)

	# print '--------- C1 ------------'
	# print '%f'%c1_train_accuracy,'\t %f'%c1_test_accuracy
	# print '--------- c2 ------------'
	# print '%f'%c2_train_accuracy,'\t %f'%c2_test_accuracy
	# print '--------- c3 ------------'
	# print '%f'%c3_train_accuracy,'\t %f'%c3_test_accuracy
	# print '--------- c4 ------------'
	# print '%f'%c4_train_accuracy,'\t %f'%c4_test_accuracy
	# print '--------- c5 ------------'
	# print '%f'%c5_train_accuracy,'\t %f'%c5_test_accuracy
	# print '--------- c6 ------------'
	# print '%f'%c6_train_accuracy,'\t %f'%c6_test_accuracy
	


def visual_color_distribution(color_space='hsv',color_ids=[1],show='on'):
	[train_set,target_set,ids_set] = load_train_data(color_space)
	fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	subplot_ids = {1:231,2:232,3:233,4:234,5:235,6:236}
	subplot_colors = {1:'r',2:'b',3:'y',4:[100,200,5],5:[40,100,5],6:[20,100,100]}
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

	# else: 	
	# 	red =  train_set[target_set==1]
	# 	ax.scatter(red[:,0],red[:,1],red[:,2], c='r', marker='^')
	# 	color_2 =  train_set[target_set==2]
	# 	ax.scatter(color_2[:,0],color_2[:,1],color_2[:,2], c='b', marker='^')
	# 	color_3 =  train_set[target_set==3]
	# 	ax.scatter(color_3[:,0],color_3[:,1],color_3[:,2], c='y', marker='^')
	# if color_id != 1:
	# 	color_data =  train_set[target_set==color_id]
	# 	ax.scatter(color_data[:,0],color_data[:,1],color_data[:,2], c='r', marker='^')
	# else: 	
	# 	red =  train_set[target_set==1]
	# 	ax.scatter(red[:,0],red[:,1],red[:,2], c='r', marker='^')
	# 	color_2 =  train_set[target_set==2]
	# 	ax.scatter(color_2[:,0],color_2[:,1],color_2[:,2], c='b', marker='^')
	# 	color_3 =  train_set[target_set==3]
	# 	ax.scatter(color_3[:,0],color_3[:,1],color_3[:,2], c='y', marker='^')
	# ax.set_xlabel('X Label')
	# ax.set_ylabel('Y Label')
	# ax.set_zlabel('Z Label')
	# ax.set_title('Color distribution id: %s',str(color_id))
	image_file = '/home/tynguyen/cis650_projects/project1/figures/' + color_space + '.png'
	fig.savefig(image_file)   # save the figure to file
	if show == 'off':
		plt.close(fig)    # close the figure
	else: 
		plt.show()

def display_segmented_image(colorspace='hsv'):
	image_folder = "data/Proj1_Train"
	segmented_image_folder = "data/segmented_images"
	if not os.path.exists(segmented_image_folder):
		os.makedirs(segmented_image_folder)
	image_files = os.listdir(image_folder)
	# [train_set_pixels,target_set,ids_set] = load_train_data(transform_color='off')
	[c1_classifier,c2_classifier,c3_classifier,c4_classifier,c5_classifier,c6_classifier] = tune_GaussianMixtures()
	count = 1 

	for image_name in image_files:	
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


		# print np.shape(np.array(X_train_score_on_c1[0]))
		# Min of likelihood of c1_score_on_c1 is 8.69e-12
		print X_train_score_on_c1.min(),X_train_score_on_c1.max()
		print X_train_score_on_c2.min(),X_train_score_on_c2.max()
		print X_train_score_on_c3.min(),X_train_score_on_c3.max()
		print X_train_score_on_c4.min(),X_train_score_on_c4.max()
		print X_train_score_on_c5.min(),X_train_score_on_c5.max()
		print X_train_score_on_c6.min(),X_train_score_on_c6.max()

		# Add a row of 0,0,0.... to make sure that the returned index matches classes [1, 2, ... ,6]
		# We can set a threshold here so that if the likelihood smaller than this threshold, just get rid of. 
		zeros_row = np.zeros(np.shape(X_train_score_on_c1)) + min_red_likelihood
		
		# y_train_score_matrix = np.array([zeros_row,X_train_score_on_c1,X_train_score_on_c2,X_train_score_on_c3,X_train_score_on_c4,X_train_score_on_c5,X_train_score_on_c6])
		y_train_score_matrix = np.array([zeros_row,X_train_score_on_c1,X_train_score_on_c2,X_train_score_on_c3,X_train_score_on_c4,X_train_score_on_c5,X_train_score_on_c6])
		
		y_train_pred = np.argmax(y_train_score_matrix,axis=0) 
		# print y_train_pred, type(y_train_pred), np.shape(y_train_pred)
		# print np.unique(y_train_pred)

		# print 'Shape of two arrays:', np.shape(y_train_pred), np.shape(y_train)
		# error = y_train_pred != y_train
		# print 'Error in not recognizing red color'
		# print np.mean(error[y_train_pred==2*y_train==1]),np.mean(error[y_train_pred==3*y_train==1]),np.mean(error[y_train_pred==4*y_train==1]),np.mean(error[y_train_pred==5*y_train==1]),np.mean(error[y_train_pred==6*y_train==1])
		# print 'Error in mistakenly recognizing red color'
		# print np.mean(error[y_train_pred==1*y_train==2]),np.mean(error[y_train_pred==1*y_train==3]),np.mean(error[y_train_pred==1*y_train==4]),np.mean(error[y_train_pred==1*y_train==5]),np.mean(error[y_train_pred==1*y_train==6])
		# train_accuracy = np.mean(y_train_pred == y_train) * 100
		# print 'Accuracy: ',train_accuracy
		# print 'Error in each color'
		# print np.mean(error[y_train==1]),np.mean(error[y_train==2]), np.mean(error[y_train==3]),np.mean(error[y_train==4]),np.mean(error[y_train==5]),np.mean(error[y_train==6])

		
		print '>>>>Number of red prediction:', sum(y_train_pred==1)
		# Transform to grayscale images: uncomment the following lines 
		# img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
		# img[:,:] = (0,0,0) 
		# c1_locate_pixels = X_train_pixels[y_train_pred==1]
		# # print 'Red pixels: ', c1_locate_pixels
		# img[c1_locate_pixels[:,0],c1_locate_pixels[:,1]] = (255,255,255)	
		# plt.imshow(img,aspect='auto')
		# plt.show()
		# command = raw_input("continue display?")
		# if command == 'N' or command == 'n':
		# 	break 

		# Transform to grayscale images: comment the following lines 
		img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img[:,:] = 0
		c1_locate_pixels = X_train_pixels[y_train_pred==1]
		# print 'Red pixels: ', c1_locate_pixels
		img[c1_locate_pixels[:,0],c1_locate_pixels[:,1]] = 1
		# print 'Shape of gray image:', np.shape(img)
		# plt.imshow(img,aspect='auto',cmap='gray')
		# plt.show()
		# command = raw_input("continue display?")
		# if command == 'N' or command == 'n':
		# 	break 

		# save images to pickle files 
		segmented_image_file = image_name + '.pickle'
		segmented_image_file = os.path.join(segmented_image_folder,segmented_image_file) 
		try: 
			with open(segmented_image_file,'wb') as f:
				pickle.dump(img,f,pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to', segmented_image_file, ':', e)


def tune_bouncing_box():
	segmented_image_folder = "data/segmented_images"
	image_folder = "data/Proj1_Train"
	count = 0 
	
	for segmented_file_name in os.listdir(segmented_image_folder):
		image_name = segmented_file_name.strip('.pickle')
		image_file = os.path.join(image_folder,image_name)
		orig_image = cv2.imread(image_file)
		orig_image = cv2.cvtColor(orig_image,cv2.COLOR_BGR2RGB)
		fig = plt.figure()
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
		# fig.add_subplot(121)
		# plt.imshow(img,aspect='auto',cmap='gray')
		
		

		se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
		se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
		mask1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
		mask2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, se2)
		out = img * mask1
		out = out*mask2 

		fig.add_subplot(121)
		plt.imshow(out,aspect='auto',cmap='gray')
		

		out = util.img_as_ubyte(out)  # > 110
		label_img = label(out, connectivity=out.ndim)
		print 'Number of labels:', np.max(skimage.measure.label(out,background=0))
		props_list = skimage.measure.regionprops(label_img)
		print 'Number of props: ', len(props_list)

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
		# 4. 
		areas = [cv2.contourArea(c) for c in contours]
		max_index = np.argmax(areas)
		cnt=contours[max_index]

		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,255,0),2)
		fig.add_subplot(122)
		plt.imshow(orig_image,aspect='auto')
		plt.show()

		command = raw_input("continue display?")
		if command == 'N' or command == 'n':
			break 




		


if __name__ == '__main__':
	# tune_GaussianMixtures('hsv')
	# display_segmented_image()
	tune_bouncing_box()
	# load_train_data()
	# visual_color_distribution('RGB',[1,2,3,4,5,6],show='off')
	# visual_color_distribution('LUV',[1,2,3,4,5,6],show='off')
	# visual_color_distribution('YCrCb',[1,2,3,4,5,6],show='off')
	# visual_color_distribution('HLS',[1,2,3,4,5,6],show='off')
	
### temporary 
# import numpy as np 
# # create mixture of three Gaussians
# num_components=3
# num_max_samples=100
# gmm=GaussianMixture(num_components)

# dimension=3

# # set means (TODO interface should be to construct mixture from individuals with set parameters)
# means = np.zeros((num_components, dimension))
# means[0]=[-5.0, -4.0]
# means[1]=[7.0, 3.0]
# means[2]=[0, 0.]
# [gmm.set_nth_mean(means[i], i) for i in range(num_components)]

# # set covariances
# covs=zeros((num_components, dimension, dimension))
# covs[0]=array([[2, 1.3],[.6, 3]])
# covs[1]=array([[1.3, -0.8],[-0.8, 1.3]])
# covs[2]=array([[2.5, .8],[0.8, 2.5]])
# [gmm.set_nth_cov(covs[i],i) for i in range(num_components)]

# # set mixture coefficients, these have to sum to one (TODO these should be initialised automatically)
# weights=array([0.5, 0.3, 0.2])
# gmm.set_coef(weights)


