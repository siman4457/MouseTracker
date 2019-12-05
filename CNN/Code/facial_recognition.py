import matplotlib
matplotlib.use('Agg')
import numpy as np
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras import optimizers

# This script builds & trains the CNN: 65% Training Accuracy, 63% validation Accuracy 
# Reference: https://medium.com/free-code-camp/facial-emotion-recognition-develop-a-c-n-n-and-break-into-kaggle-top-10-f618c024faa7
# (While we used the load_data() function from the article, the actual CNN was a variation from the one in the book Deep Learning with Python.)

# Load data from .csv file
def load_data(dataset_path):
  data = []
  test_data = []
  labels =[]
  test_labels = []
  with open(dataset_path, 'r') as file:
      for line_no, line in enumerate(file.readlines()):
          if 0 < line_no <= 35887:
            curr_class, line, set_type = line.split(',')
            image_data = np.asarray([int(x) for x in line.split()]).reshape(48, 48)
            image_data = image_data.astype(np.uint8)/255.0
            
            # Only get data labeled as happy, sad, neutral (in order)
            if(curr_class == '3' or curr_class == '4' or curr_class == '6'): 
              if (set_type.strip() == 'PrivateTest'):
                test_data.append(image_data)
                test_labels.append(curr_class)
              else:
                data.append(image_data)
                labels.append(curr_class)
      
      test_data = np.expand_dims(test_data, -1)
      test_labels = to_categorical(test_labels, num_classes = 7)
      data = np.expand_dims(data, -1)   
      labels = to_categorical(labels, num_classes = 7)
    
      return np.array(data), np.array(labels), np.array(test_data), np.array(test_labels)

# Print training data
def print_training_data(acc, val_acc, loss, val_loss):
	print('Training Accuracy:\n', acc) 
	print('Validation Accuracy:\n', val_acc) 
	print('Training Loss:\n', loss) 
	print('Validation Loss:\n', val_loss) 

# Write data to .csv file
def write_data_to_file(acc, val_acc, loss, val_loss):
	import pandas as pd
	# Write data to .csv file
	# Note: Clear data.csv each time! After clearing, add '0' to make it non-empty
	open('data.csv', 'w+').close() # Clear file before writing to it (and create if nonexistent)
	with open('data.csv', 'w') as f:
		f.write('0') # Add an initial value
	f.close()
	print('Writing data to .csv file...')
	data = pd.read_csv('data.csv', 'w') 
	data.insert(0,"Training Acc", acc)
	data.insert(1,"Validation Acc", val_acc)
	data.insert(2,"Training Loss", loss)
	data.insert(3,"Validation Loss", val_loss)
	data.to_csv('data.csv')
	print('Finished writing data!')

# Plot training/validation graphs
def plot_graphs(acc, val_acc, loss, val_loss):
	import matplotlib.pyplot as plt

	# Plot accuracy
	print("Starting plotting...") 
	epochs = range(1, len(acc)+1)
	plt.plot(epochs, acc, 'bo', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Training and Validation accuracy')
	plt.legend()
	plt.savefig('Emotions_TrainingAcc.png')
	plt.figure()

	# Plot loss
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and Validation loss')
	plt.legend()
	plt.ylim([0,4]) # Loss should not increase beyond 4! 
	plt.savefig('Emotions_TrainingLoss.png')
	plt.figure()
	plt.show()
	print('Finished plotting!')

def create_model():
	# Create model
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3,3), activation='relu',input_shape=(None, None, 1)))
	model.add(layers.BatchNormalization())
	# model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
	# model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D(2,2))
	model.add(layers.Dropout(0.5))

	model.add(layers.Conv2D(64, (3,3), activation='relu'))
	model.add(layers.BatchNormalization())
	# model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
	# model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D(2,2))
	model.add(layers.Dropout(0.5))

	model.add(layers.Conv2D(128, (3,3), activation='relu'))
	model.add(layers.BatchNormalization())
	# model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
	# model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D(2,2))
	model.add(layers.Dropout(0.5))

	model.add(layers.Conv2D(256, (3,3), activation='relu'))
	model.add(layers.BatchNormalization())
	# model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
	# model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D(2,2))
	model.add(layers.Dropout(0.5))

	# model.add(layers.Flatten()) # This doesn't work for inputs of arbitrary shape 
	model.add(layers.GlobalAveragePooling2D())
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(7, activation='softmax'))
	model.summary()
	print ("Model Outputs: ", model.output)
	return model

# ----------------------- Main ----------------------- #
dataset_path = 'fer2013/fer2013.csv' # Path to training data 
# dataset_path = 'challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013.csv'

# Load data 
train_data, train_labels, test_data, test_labels = load_data(dataset_path)
print("len(train_data): ", len(train_data))
print("len(train_labels): ", len(train_labels))
print("len(test_data): ", len(test_data))
print("len(test_labels): ", len(test_labels))

# Build & train model 
model = create_model() # Create model 
model.compile(optimizer=optimizers.SGD(lr=0.01), loss = 'categorical_crossentropy', metrics=['accuracy']) # Compile model
history = model.fit(train_data, train_labels, epochs=100, batch_size=64, validation_split=0.2, shuffle=True) # Fit model to data 

# Save model 
print("Saving model...")
model.save('facial_recog.h5')
print("Model saved successfully!")

# Test model on test set
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("test_acc: ", test_acc)
print("test_loss: ", test_loss)

# Get names of accuracy/loss values (stored as dictionary)
history_dict = history.history
print("Keys: ", history_dict.keys())

# Get accuracy/loss values 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Save accuracy/loss values
print_training_data(acc, val_acc, loss, val_loss)
write_data_to_file(acc, val_acc, loss, val_loss)
plot_graphs(acc, val_acc, loss, val_loss)

