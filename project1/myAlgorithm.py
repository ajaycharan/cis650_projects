from __init__ import *
import cv2, os 
# from use_GMM import myAlgorithm
log_file = "test_result.csv"
log_array = []
# folder = "Test_Set" 
folder = "data/pickle_folder" 
image_num = 0 

for filename in os.listdir(folder):
    # Read one test image 
    # img = cv2.imread(os.path.join(folder,filename))

    # # My compuatation 
    # blX, bl, trX, trY, d = myAlgorithm(img)

    image_num += 1 

    # row to write to file 
    row = 'ImageNo = [%s], BottomLeftX = %.3f, BottomLeftY = %.3f, TopRightX = %.3f, TopRightY = %.3f, Distance = %.3f\n'%(str(image_num).zfill(2) , 2.01, 3.02, 2.492, 0.2844, 3.58324)
    log_array.append(row)

    # Display results 
    # (1) Segmented image
	# (2) Barrel bounding box
	# (3) Distance of barrel
	# You may also want to plot and display other diagnostic information
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

# Write results 
if os.path.exists(log_file):
	os.remove(log_file)
try:
	with open(log_file,'wb') as f:
		for row in log_array:
			f.write(row)
except Exception as e:
	print('Unable to write to file', log_file, ':', e)
	raise



