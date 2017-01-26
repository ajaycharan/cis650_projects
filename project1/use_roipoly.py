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
color_dict = {1:"barrel RED", 2:"floor red: 2", 3: "wall red: 3", 4: "chair red: 4", 5: "yellow: 5"}

def ROI2RGB(ROI, currentImage):
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


	# suppose that each image contains less than 100 000 pixels of each kind of color 
	# barrel_red_set = np.ndarray(shape=(len(image_files), 100000),
 #                         dtype=np.float32)
	# floor_red_set = np.ndarray(shape=(len(image_files), 100000),
 #                         dtype=np.float32) 
	# wall_red_set =  np.ndarray(shape=(len(image_files), 1000000),
 #                         dtype=np.float32) 
	# chair_red_set =  np.ndarray(shape=(len(image_files), 100000),
 #                         dtype=np.float32) 
	# yellow_wall_set =  np.ndarray(shape=(len(image_files), 100000),
 #                         dtype=np.float32) 
	# gray_brick_set =  np.ndarray(shape=(len(image_files), 100000),
 #                         dtype=np.float32) 
	# green_tree_set =  np.ndarray(shape=(len(image_files), 100000),
 #                         dtype=np.float32)  

	barrel_red_set = []
	floor_red_set  = []
	wall_red_set   = []
	chair_red_set  = []
	yellow_wall_set= []
	gray_brick_set = []


	num_images = 0 
	for image in image_files:
		image_file = os.path.join(folder,image)
		print "\n %d. Image file: "%(num_images+1), image_file 
		num_images += 1

		# open image 
		img = cv2.imread(image_file)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# OpenCV represents RGB images as multi-dimensional Numpy but in reverse order: BGR rather than RGB. 
		# Thus, we have to convert it back to RGB 
		pl.imshow(img, aspect='auto')	
		pl.title("left click: line segment         right click: close region. Let draw barrel RED")

		# let user draw first ROI which is barrel RED 
		print '-- Draw ROI of barrel (red) '
		barrel_red_ROI = roipoly(roicolor='r') #let user draw first ROI

		# show the image with the first ROI
		pl.imshow(img, aspect='auto')
		barrel_red_ROI.displayROI()
		# print np.shape(img)
		barrel_grid = ROI2RGB(barrel_red_ROI, img)
		# print 'barrel grid ', len(barrel_grid), barrel_grid[0]
		
		# # Uncomment the following two lines before running test_saved_pickle() 
		# img[barrel_grid] = (0,0,0)
		# barrel_red_set.append(img)
		# # Comment the following line before running test_saved_pickle() 
		barrel_red_set.append(img[barrel_grid])
		pl.imshow(img, aspect='auto')
		print "-- Sofar, total number of element in barrel red set: ", np.shape(barrel_red_set)
		

		# add RGB of the pixels to the corresponding data array 


		while True: 
			command = raw_input("-- Draw ROI for another color YN ?:  ")
			command = command.lower()
			if command == 'n' or command == 'N':
				break 
			# else, let user draw next ROI
			print "--- Draw next ROI...."
			ROI = roipoly(roicolor='b') #let user draw ROI
			color = int(raw_input("---- What color it is? (floor red: 2, wall red: 3, chair red: 4, yellow: 5, gray brick: 6) : "))
			pl.imshow(img, aspect='auto')
			ROI.displayROI()
			print "----- Color has been segmented:", color, " or ", color_dict[color]

			grid = ROI2RGB(ROI, img)
			if color == 2:
				floor_red_set.append(img[grid])
				print "-- Sofar, total number of element in ", color_dict[color], "set:", np.shape(floor_red_set)
			elif color == 3: 
				wall_red_set.append(img[grid])
				print "-- Sofar, total number of element in ", color_dict[color], "set:", np.shape(wall_red_set)
			elif color == 4: 
				chair_red_set.append(img[grid])
				print "-- Sofar, total number of element in ", color_dict[color], "set:", np.shape(chair_red_set)
			elif color == 5: 
				yellow_wall_set.append(img[grid])
				print "-- Sofar, total number of element in ", color_dict[color], "set:", np.shape(yellow_wall_set)
			else 		  : 
				gray_brick_set.append(img[grid])
				print "-- Sofar, total number of element in ", color_dict[color], "set:", np.shape(gray_brick_set)
		# cv2.waitKey(27)
		# clear window to avoid inerference 	
		pl.clf()

	# dump datasets to pickle files 
	pickle_folder = "data/pickle_folder"
	if not os.path.exists(pickle_folder):
		os.makedirs(pickle_folder)
	dump_pickle(pickle_folder, "barrel_red_set", barrel_red_set)
	dump_pickle(pickle_folder, "floor_red_set", floor_red_set)
	dump_pickle(pickle_folder, "wall_red_set", wall_red_set)
	dump_pickle(pickle_folder, "chair_red_set", chair_red_set)
	dump_pickle(pickle_folder, "yellow_wall_set", yellow_wall_set)
	dump_pickle(pickle_folder, "gray_brick_set", gray_brick_set)

# def test_saved_pickle():
# 	"""This funtion is used to make sure that the content that we saved to each dataset is correct! 
# 		Before running this function, comment and uncomment somelines mentioned in main() function"""
# 	pickle_folder = "data/pickle_folder"
# 	files = os.listdir(pickle_folder)
# 	for file in files:
# 		file_name = os.path.join(pickle_folder,file)
# 		print 'file_name:',file_name
# 		try:
# 			with open(file_name, 'rb') as f:
# 				barrel_red_set = pickle.load(f)
# 				print np.shape(barrel_red_set)
# 				pl.imshow(barrel_red_set[1])
# 				pl.show()
# 				time.sleep(100)
# 		except Exception as e:
# 			print('Unable to process data from', file_name, ':', e)
# 			raise

if __name__=="__main__":
	main()
	# test_saved_pickle()


