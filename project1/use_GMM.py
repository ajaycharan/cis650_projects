
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import os 
import cv2 
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
from six.moves import cPickle as pickle
import time
print 'Import has been done!!'

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

def load_train_data(colorspace='hsv'):
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

def main(color_space='hsv'):
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

	# Try GMMs using different types of covariances.
	# classifiers = dict((covar_type, GMM(n_components=n_classes,
	#                     covariance_type=covar_type, init_params='wc', n_iter=20))
	#                    for covar_type in ['spherical', 'diag', 'tied', 'full'])
	classifiers = dict((covar_type, GMM(n_components=n_classes,
	                    covariance_type=covar_type, init_params='wc', n_iter=20))
	                   for covar_type in ['spherical', 'diag', 'tied','full'])

	n_classifiers = len(classifiers)

	plt.figure(figsize=(3 * n_classifiers / 2, 6))
	plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
	                    left=.01, right=.99)

	# Training using one GMM consists of a number of components equal to number of classes. 
	for index, (name, classifier) in enumerate(classifiers.items()):
	    # Since we have class labels for the training data, we can
	    # initialize the GMM parameters in a supervised manner.

	    classifier.means_ = np.array([X_train[y_train==i].mean(axis=0)
	                                  for i in xrange(1,n_classes+1)])
	    print classifier.means_
	    # exit()
	    # Train the other parameters using the EM algorithm.
	    classifier.fit(X_train)

	    h = plt.subplot(2, n_classifiers / 2, index + 1)
	    make_ellipses(classifier, h)

	    for n, color in enumerate('rgb'):
	        data = data_set.data[data_set.target == n]
	        plt.scatter(data[:, 0], data[:, 1], 0.8, color=color)
	    # Plot the test data with crosses
	    for n, color in enumerate('rgb'):
	        data = X_test[y_test == n]
	        plt.plot(data[:, 0], data[:, 1], 'x', color=color)

	    y_train_pred = classifier.predict(X_train)
	    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
	    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
	             transform=h.transAxes)

	    y_test_pred = classifier.predict(X_test)
	    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
	    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
	             transform=h.transAxes)

	    plt.xticks(())
	    plt.yticks(())
	    plt.title(name)

	plt.legend(loc='lower right', prop=dict(size=12))


	plt.show()

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
	image_file = '/home/tynguyen/proj1_cis650/' + color_space + '.png'
	fig.savefig(image_file)   # save the figure to file
	if show == 'off':
		plt.close(fig)    # close the figure
	else: 
		plt.show()

if __name__ == '__main__':
	# main('hsv')
	# load_train_data()
	visual_color_distribution('RGB',[1,2,3,4,5,6],show='off')
	
### temporary 
# import numpy as np 
# # create mixture of three Gaussians
# num_components=3
# num_max_samples=100
# gmm=GMM(num_components)

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


