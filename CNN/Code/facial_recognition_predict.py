import os
import numpy as np
from skimage import io
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# This script runs the CNN predictions on the images obtained from the experiment results
# (Images are stored in 'Images' folder on the Google Drive)
# 5 categories are evaluated: Happy Max, Happy Average, Happy Variance, Neutral Average, Neutral Variance, Sad Max
# Results are printed to facial_recog_data.csv (5 categories per experiment platform)

# Write data to .csv file
def write_data_to_file(data_labels, happyMax, happyAvg, happyVar, neutralAvg, neutralVar, sadMax):
	import pandas as pd
	# Note: Clear data.csv each time! After clearing, add '0' to make it non-empty
	open('facial_recog_data.csv', 'w+').close() # Clear file before writing to it (and create if nonexistent)
	with open('facial_recog_data.csv', 'w') as f:
		f.write('0') # Add an initial value
	f.close()
	print('Writing data to .csv file...')
	data = pd.read_csv('facial_recog_data.csv', 'w') 
	data.insert(0,"filename", data_labels)
	data.insert(1,"happyMax", happyMax)
	data.insert(2,"happyAvg", happyAvg)
	data.insert(3,"happyVar", happyVar)
	data.insert(4,"neutralAvg", neutralAvg)
	data.insert(5,"neutralVar", neutralVar)
	data.insert(6,"sadMax", sadMax)
	data.to_csv('facial_recog_data.csv')
	print('Finished writing data!')


# Parse the model predictions into happy/neutral/sad array values 
def parse_model_predictions(predictions):
	happy_vals = []
	neutral_vals = []
	sad_vals = []
	print("Printing predictions...")
	for (i,pred) in enumerate(predictions): # Get all probabilities per happy/sad/neutural class
		happy_vals.append(pred[3])
		sad_vals.append(pred[4])
		neutral_vals.append(pred[6])
	return happy_vals, neutral_vals, sad_vals

# Get data labels (filenames)
def get_data_labels(path):
	data_labels = os.listdir(path)
	return data_labels

# ----------------------- Main ----------------------- #
path = 'images/'
model = load_model('facial_recog.h5') # Get model
data_labels = get_data_labels(path)

happyMax = []
happyAvg = []
happyVar = []
neutralAvg = []
neutralVar = []
sadMax = []
new_data_labels=[]

for (j,label) in enumerate(data_labels):
	# Prepare & convert test data
	test_dir = path + label
	print("test_dir: ", test_dir)

	# Get image height and width (can vary in dataset)
	image_list = os.listdir(test_dir) # Get list of images
	img_tmp_path = image_list[0] # Get filname of 1st image
	img_tmp = Image.open(test_dir+'/'+img_tmp_path).convert('L') # Convert image to grayscale
	img_tmp = img_to_array(img_tmp) # Convert image to array
	(height, width, channels)=img_tmp.shape # Finally get height, width, channel parameters
	
	# Get number of images in list 
	num_imgs = len(image_list) 
	print("num_imgs: ", num_imgs)

	test_imgs = np.zeros([num_imgs, height, width, 1]) # Create array of testing images
	# Iterate over images in folder
	for (i,img_path) in enumerate(os.listdir(test_dir)): 
		img = Image.open(test_dir+'/'+img_path).convert('L') # Convert image to grayscale
		img_as_array = img_to_array(img) # Convert image to array
		test_imgs[i,:,:,:] = (img_as_array) # Add image to test_imgs array 
		
	# Get predictions
	print('test_imgs.shape', (test_imgs.shape))
	predictions = model.predict(test_imgs, batch_size=64) # Make predictions
	happy_vals, neutral_vals, sad_vals = parse_model_predictions(predictions) # Get associated arrays 
	
	# Append data to categories
	happyMax.append(np.max(happy_vals))
	happyAvg.append(np.average(happy_vals))
	happyVar.append(np.var(happy_vals))
	neutralAvg.append(np.average(neutral_vals))
	neutralVar.append(np.var(neutral_vals))
	sadMax.append(np.max(sad_vals))
	new_data_labels.append(label)

# Write all data to .csv file 
write_data_to_file(new_data_labels, happyMax, happyAvg, happyVar, neutralAvg, neutralVar, sadMax)
print('Finished writing data to file! All done!') # Done!!!!!

