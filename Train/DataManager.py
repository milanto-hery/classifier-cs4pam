from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn import utils
from tensorflow.keras.utils import to_categorical

class DataManager:

	def __shuffle_data(self, spetrograms, labels, seed):

	    return utils.shuffle(spetrograms, labels, random_state=seed)
	
	def __augment_with_time_shift(self, X_data, absence_class_label, amount_to_create):
		
		''' Augment the spectrograms by shifting them to some random
		time steps to the right.
		'''
		# Lists to store the new augmented data
		augmented_spectrograms = []
		augmented_labels = []

		# Iterate a number of times
		for i in range (0, amount_to_create):

			# Randomly select a presence spectrogram
			random_absence_index = random.randint(0, len(X_data)-1)
			random_absence_spectrogram = X_data[random_absence_index]
			
			# Randomly select amount to shift by
			random_shift = random.randint(1, random_absence_spectrogram.shape[1]-1)

			# Time shift
			shifted_spectrogram = np.roll(random_absence_spectrogram, random_shift, axis=1)

			# Append the augmented spectrogram
			augmented_spectrograms.append(shifted_spectrogram)

			# Append the class label (to the presence class)
			augmented_labels.append(absence_class_label)
			
		# Return the augmented spectograms and labels
		return np.asarray(augmented_spectrograms), np.asarray(augmented_labels)


	def __target_encoding(self, y, call_order):


		# Transform the Y labels into one-hot encoded labels
		for index, call_type in enumerate(call_order):
			y = np.where(y == call_type, index, y)

		# Convert from Integer values to one-hot
		y = to_categorical(y, num_classes = 2)

		return y

	def augment_and_prep_data(self, presence_class_label, X_calls, 
		y, seed, train_size, call_order, verbose):
		# add the channel 1 dimension to confirm the image is grayscale
		#X_calls = np.expand_dims(X_calls, axis=-1)
		X_calls_train, X_calls_test, y_train, y_test = train_test_split(
			X_calls, y, shuffle = True, random_state = seed, train_size = train_size)

		if verbose:
			print ('Original split shapes')
			print('Training__: ', X_calls_train.shape)
			print('Validation: ', X_calls_test.shape)

			print('Train_label: ', y_train.shape)
			print('Val_label__: ',y_test.shape)

		# Determine the presence indices from the training data
		presence_class_label = 'gibbon'
		presence_indices = np.where(y_train == presence_class_label)[0]

		# Find out how many presence examples there are in the training data
		amount_presence = len(presence_indices)

		# Determine the absence indices from the training data
		absence_indices = np.where(y_train != presence_class_label)[0]

		# Find out how many absence examples there are in the training data
		amount_absence = len(absence_indices)

		# Determine the distribution of the training data
		unique, counts = np.unique(y_train, return_counts=True)
		original_distribution = dict(zip(unique, counts))

		if verbose:
			print('Original Data distribution:',original_distribution)

		# Determine the difference between the number of absence and presence
		difference = abs(amount_absence-amount_presence)

		if verbose:
			print ('difference between absence and presence training examples', difference)

		# Assert
		assert difference > 0, f"difference greater than 0 is expected, got: {difference}"

		# Create several new spectrograms (i.e. increase the number of absence examples)
		X_augmented, Y_augmented = self.__augment_with_time_shift(
		    X_calls_train[absence_indices], 
		    'no-gibbon', difference)

		# The new presence examples is now made up of the augmented examples and the original training ones
		X_absence = np.concatenate((X_augmented, X_calls_train[absence_indices]))
		Y_absence = np.concatenate((Y_augmented, y_train[absence_indices]))

		# The absence examples are from the original training data
		X_presence = X_calls_train[list(presence_indices)]
		Y_presence = y_train[list(presence_indices)]

		# The new training data is the concatenation of the presence (with augmentated examples) and absence examples
		X_calls_train = np.concatenate((X_presence, X_absence))
		y_train = np.concatenate((Y_presence, Y_absence))

		X_calls_train, y_train = self.__shuffle_data(X_calls_train, y_train, seed)

		# These aren't needed anymore
		del X_presence, X_absence, Y_presence, Y_absence

		# Target encoding
		y_train = self.__target_encoding(y_train, call_order)
		y_test = self.__target_encoding(y_test, call_order)


		if verbose:
			print ('Shapes (after augmentation):')
			print('X_calls_train', X_calls_train.shape)
			print('X_calls_test',X_calls_test.shape)

			print('y_train',y_train.shape)
			print('y_test',y_test.shape)

			unique, counts = np.unique(y_train, return_counts=True)
			original_distribution = dict(zip(unique, counts))
			print('New Data distribution:',original_distribution)

		return X_calls_train, X_calls_test, y_train, y_test
    
    # only applied for pretrained model: reshape the channel 1 to rgb
	def reshape_to_rgb(self, X_calls):
		X_calls = np.reshape(X_calls, (X_calls.shape[0],
			X_calls.shape[1],
			X_calls.shape[2],
              1))
		X_calls_rgb = []
		for x in X_calls:
			X_calls_rgb.append(np.dstack([x]*3))
		X_calls_rgb = np.asarray(X_calls_rgb)
		return X_calls_rgb
