import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Just a script to modify the Python plots

# Plot training/validation graphs
def plot_graphs(acc, val_acc, loss, val_loss):
	import matplotlib.pyplot as plt

	# Plot results
	print("Starting plotting...") 
	epochs = range(1, len(acc)+1)
	plt.plot(epochs, acc, 'bo', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Training and Validation accuracy')
	plt.legend()
	plt.xlabel('Number of Epochs')
	plt.ylabel('Accuracy')
	plt.savefig('Emotions_Acc.png')
	plt.figure()

	epochs = range(1, 100+1)
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and Validation loss')
	plt.legend()
	plt.ylim([0,1.75]) # Loss should not increase beyond 4! 
	plt.ylabel('Number of Epochs')
	plt.xlabel('Loss')
	plt.savefig('Emotions_Loss.png')
	plt.figure()
	plt.show()
	print('Finished plotting!')

data = pd.read_csv('data.csv') 
# print("len(Data): ", len(data))

acc = []
val_acc = []
loss = []
val_loss = []
for row in range(len(data)):
	acc.append(data.iloc[row][1])
	val_acc.append(data.iloc[row][2])
	loss.append(data.iloc[row][3])
	val_loss.append(data.iloc[row][4])

print("len(acc): ", len(acc))
print("len(val_acc): ", len(val_acc))
print("len(loss): ", len(loss))
print("len(val_loss): ", len(val_loss))
plot_graphs(acc, val_acc, loss, val_loss)


