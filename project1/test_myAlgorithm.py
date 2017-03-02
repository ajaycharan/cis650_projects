from myAlgorithm import * 
import cv2, os 
# from use_GMM import myAlgorithm
log_file = "test_result.csv"
log_array = []
folder = "Test_Set" 
# Check test folder 
if not os.path.exists(folder):
	print 'Error! Cannot find folder ', folder 
	exit(1)

image_num = 0 
for filename in os.listdir(folder):
	print 'Image: ', filename 
	image_num += 1 
	# Read one test image 
	img = cv2.imread(os.path.join(folder,filename))

	# My compuatation 
	final_result = display_bouncing_box(img,filename)

	if np.shape(final_result)[0] == 0:
		row = 'ImageNo = [%s]. There is no box!'%(str(image_num).zfill(2))
	else: 
		boxs = ['BottomLeftPoint = (%.3f,%.3f), TopLeftPoint = (%.3f,%.3f), TopRightPoint = (%.3f,%.3f), BottomRightPoint = (%.3f,%.3f), Distance = %.3f'%(box[0][0],box[0][1],box[1][0],box[1][1],box[2][0],box[2][1],box[3][0],box[3][1],d) for box, d in final_result]
		# row to write to file 
		# row = 'ImageNo = [%s], BottomLeftX = %.3f, BottomLeftY = %.3f, TopRightX = %.3f, TopRightY = %.3f, Distance = %.3f\n'%(str(image_num).zfill(2) , 2.01, 3.02, 2.492, 0.2844, 3.58324)
		# log_array.append(row)
		row = 'ImageNo = [%s], '%(str(image_num).zfill(2)) + ' '.join(boxs) + '\n'
		log_array.append(row)


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



