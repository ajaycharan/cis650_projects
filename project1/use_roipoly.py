from roipoly import roipoly 
# import pylab as pl 
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pl
import matplotlib.path as mplPath
import cv2 
import numpy as np
import os
import sys
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import time 
"""
We will store RGB of pixels to each color dataset it belongs to. 
We encode colors as following: 
barrel_red : 1 
floor_red:   2 
"""
color_dict = {1:"barrel RED", 2:"floor red: 2", 3: "wall red: 3", 4: "chair red: 4", 5: "wooden yellow: 5", 6:"gray brick: 6", 7:"green grass: 7"}

def makeGrid(ROI, currentImage):
    ny, nx, three = np.shape(currentImage)
    poly_verts = [(ROI.allxpoints[0], ROI.allypoints[0])]
    for i in range(len(ROI.allxpoints)-1, -1, -1):
        poly_verts.append((ROI.allxpoints[i], ROI.allypoints[i]))

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T

    ROIpath = mplPath.Path(poly_verts)
    grid = ROIpath.contains_points(points).reshape((ny,nx))
    return grid

def dump_pickle(folder, data_name, data):
	file=  data_name + '.pickle'
	file_name = os.path.join(folder, file)
	try:
		with open(file_name, 'wb') as f:
			pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save data to', file_name, ':', e)

def main():

	folder = "data/Proj1_Train/"
	# list all image files 
	image_files = os.listdir(folder)

 	# check the existance of pickle files 
	pickle_folder = "data/pickle_folder"
	if not os.path.exists(pickle_folder):
		os.makedirs(pickle_folder)
 
	data_set_file = "train" + ".pickle"
	pickle_file = os.path.join(pickle_folder,data_set_file)
	if os.path.exists(pickle_file):
		print 'train file existed. reading...'
		with open(pickle_file, 'rb') as f:
			train = pickle.load(f)
	else: 
		train = np.array([])


	data_set_file = "target" + ".pickle"
	pickle_file = os.path.join(pickle_folder,data_set_file)
	if os.path.exists(pickle_file):
		with open(pickle_file, 'rb') as f:
			target = pickle.load(f)
	else: 
		target = np.array([])

	data_set_file = "ids" + ".pickle"
	pickle_file = os.path.join(pickle_folder,data_set_file)
	if os.path.exists(pickle_file):
		with open(pickle_file, 'rb') as f:
			ids = pickle.load(f)
	else: 
		ids = np.array([])

	num_images = 0 
	for image in image_files:
		image_file = os.path.join(folder,image)
		print "\n %d. Image file: "%(num_images+1), image_file 
		num_images += 1

		# open image 
		img = cv2.imread(image_file)
		img_denoised = cv2.fastNlMeansDenoisingColored(img,None,5,5,3,5)
		# cv2.imshow('original image',img)
		# cv2.imshow('denoised image', img_denoised)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# exit(0)

		img = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2RGB)
		# OpenCV represents RGB images as multi-dimensional Numpy but in reverse order: BGR rather than RGB. 
		# Thus, we have to convert it back to RGB 
		pl.imshow(img, aspect='auto')	
		pl.title("left click: line segment         right click: close region. Let draw barrel RED")

		# let user draw first ROI which is barrel RED 
		print '-- Draw ROI:  '
		ROI = roipoly(roicolor='r') #let user draw first ROI

		color = int(raw_input("---- What color it is? (barrel: 1, floor red: 2, wall red: 3, chair red: 4, yellow: 5, wall gray brick: 6, green grass: 7) : "))

		# show the image with the first ROI
		pl.imshow(img, aspect='auto')
		ROI.displayROI()
		# print np.shape(img)
		grid = makeGrid(ROI, img)
		
		# # Uncomment the following two lines before running test_saved_pickle() 
		# img[grid] = (0,0,0)
		# ROI.append(img)
		# # Comment the following lines before running test_saved_pickle() 

		# append data 
		# train data 
		locate_pixels = np.where(grid)
		num_pixels = np.shape(locate_pixels)[1]
		img_ids = np.array([image for i in range(num_pixels)])
		color_ids = np.array([color for i in range(num_pixels)])

		# train
		try:
			train = np.concatenate((train, np.vstack(([locate_pixels[0].T],[locate_pixels[1].T])).T), axis=0)
		except: 
			train = np.vstack(([locate_pixels[0].T],[locate_pixels[1].T])).T
		# target 
		try: 
			target = np.concatenate((target,color_ids), axis=0)
		except: 
			target = color_ids
		try: 
			ids = np.concatenate((ids,img_ids), axis=0)
		except:
			ids = img_ids
		# print train[0]
		# print train[-1]
		# print np.shape(train)
		# print np.shape(target)
		# print np.shape(ids)
		# print 'grid and image shape'
		# print np.shape(grid)
		# print np.shape(img)
		# print np.shape(img_ids)
		# print grid[0]


		pl.imshow(img, aspect='auto')
		print "-- Sofar, total number of element in the train set: ", np.shape(train)
		

		# add RGB of the pixels to the corresponding data array 


		while True: 
			command = raw_input("-- Draw ROI for another color Y/N ?:  ")
			command = command.lower()
			if command == 'n' or command == 'N':
				break 
			# else, let user draw next ROI
			print "--- Draw next ROI...."
			print "----  barrel: 1, floor red: 2, wall red: 3, chair red: 4, yellow: 5, wall gray brick: 6 , green grass: 7!! "
			ROI = roipoly(roicolor='b') #let user draw ROI
			color = int(raw_input("---- What color it is? (barrel: 1, floor red: 2, wall red: 3, chair red: 4, yellow: 5, wall gray brick: 6, green grass: 7) : "))
			pl.imshow(img, aspect='auto')
			ROI.displayROI()
			print "----- Color has been segmented:", color, " or ", color_dict[color]

			grid = makeGrid(ROI, img)

			# append data 
			# train data 
			locate_pixels = np.where(grid)
			num_pixels = np.shape(locate_pixels)[1]
			img_ids = np.array([image for i in range(num_pixels)])
			color_ids = np.array([color for i in range(num_pixels)])

			# train
			train = np.concatenate((train, np.vstack(([locate_pixels[0].T],[locate_pixels[1].T])).T), axis=0)
			# target 
			target = np.concatenate((target,color_ids), axis=0)
			ids = np.concatenate((ids,img_ids), axis=0)

			
		# cv2.waitKey(27)
		# clear window to avoid inerference 	
		pl.clf()

		if num_images >= 10:
			command = raw_input("-- Continue segmentation?  Y/N ?:  ")
			command = command.lower()
			if command == 'n' or command == 'N':
				break 

	# dump datasets to pickle files 
	if not os.path.exists(pickle_folder):
		os.makedirs(pickle_folder)
	# remove old file 
	data_set_file = "train" + ".pickle"
	pickle_file = os.path.join(pickle_folder,data_set_file)
	if os.path.exists(pickle_file):
		os.remove(pickle_file)
	# create new file 
	dump_pickle(pickle_folder, "train", train)
	# remove old file 
	data_set_file = "target" + ".pickle"
	pickle_file = os.path.join(pickle_folder,data_set_file)
	if os.path.exists(pickle_file):
		os.remove(pickle_file)
	# create new file 
	dump_pickle(pickle_folder, "target", target)

	# remove old file 
	data_set_file = "ids" + ".pickle"
	pickle_file = os.path.join(pickle_folder,data_set_file)
	if os.path.exists(pickle_file):
		os.remove(pickle_file)
	# create new file 
	dump_pickle(pickle_folder, "ids", ids)


def test_saved_pickle():
	"""This funtion is used to make sure that the content that we saved to each dataset is correct! 
		Before running this function, comment and uncomment somelines mentioned in main() function"""
	pickle_folder = "data/pickle_folder"
	files = os.listdir(pickle_folder)
	for file in files:
		file_name = os.path.join(pickle_folder,file)
		print 'file_name:',file_name
		try:
			with open(file_name, 'rb') as f:
				barrel_red_set = pickle.load(f)
				print np.shape(barrel_red_set)
				print barrel_red_set[0]
				print barrel_red_set[-1]
		except Exception as e:
			print('Unable to process data from', file_name, ':', e)
			raise

if __name__=="__main__":
	main()
	test_saved_pickle()


