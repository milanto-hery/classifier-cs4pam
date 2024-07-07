#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
import pickle
import os
import yaml
import argparse
import json

from CNNNetwork import *
from DataManager import * #augment_and_prep_data, reshape_to_rgb

class TrainHelper:
    def __init__(self):
        self.model = None
        self.history = None
        self.save_results_folder = None
        self.data = DataManager()

    def compile_model(self, model_type, input_data):
    
        nx,ny,nz = input_data[0].shape # the input shape of data define the CNN input
        CNN = CNNNetwork()
        if model_type == 'CNN':
            self.model = CNN.custom_cnn((nx,ny,nz))  # Replace with your CNN model
        elif model_type == 'ResNet50':
            self.model = CNN.ResNet50((nx,ny,nz)) 
        elif model_type == 'ResNe50V2':
            self.model = CNN.ResNet50V2((nx,ny,nz))  # Replace with your Resnet 50V2
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        self.model.summary()
    

    def train_model(self, X_train, y_train, X_val, y_val, batch_size, epochs, callbacks):
        # Convert the target variables to categorical if needed
        num_classes = len(np.unique(y_train))
        if num_classes > 2:
            y_train = to_categorical(y_train, num_classes)
            y_val = to_categorical(y_val, num_classes)
            print(f'Number of classes: {num_classes}.')

        # Train the model and store the history
        self.history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, verbose=2, callbacks=callbacks)
        
        # Calculate the training loss
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        print(f'Train Accuracy: {train_acc:.4f}')
        print(f'Train Loss: {train_loss:.4f}')

        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f'Validation Accuracy: {val_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')

    def plot_evaluation(self):
        # Retrieve the training history
        history = self.history
        
        # Plot the training and validation loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot the training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, X_val, y_val):
        # Predict the labels for the validation set
        y_pred = self.model.predict(X_val)
        
        # Convert the multilabel-indicator format to binary labels
        y_pred_binary = np.argmax(y_pred, axis=1)
        y_val_binary = np.argmax(y_val, axis=1)
        
        # Compute the confusion matrix
        cm = confusion_matrix(y_val_binary, y_pred_binary)
        
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
        
    def train_and_evaluate(self, model_type, X_train, y_train, X_val, y_val, batch_size, epochs, callbacks):
        # Compile the model
        self.compile_model(model_type, X_train)

        # Train the model
        self.train_model(X_train, y_train, X_val, y_val,batch_size, epochs, callbacks)
        # Get the current date and time
        
        # Plot the evaluation metrics
        self.plot_evaluation()

        # Plot the confusion matrix
        self.plot_confusion_matrix(X_val, y_val)
        
    #     return config
    def load_json_config(self, file_name):
        try:
            with open(file_name, 'r') as file:
                config = json.load(file)
            return config
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file '{file_name}': {e}")
            return None

    def run_training(self, X, Y, config_species, model_name, method_name, compression_parameter,number_of_run):

        data_m = DataManager()
        X = np.expand_dims(X, axis=-1)
        
        config = self.load_json_config('params.json')
        if config:
            print(config)

        save_weights_folder = 'Saved_weights'

        # Extract parameters from the config
        call_order = config[config_species]['call_order']
        absence_class_label = config[config_species]['absence_class_label']
        batch_size = config[config_species]['batch_size']
        epochs = config[config_species]['epochs']
        train_size = config[config_species]['train_size']
        seed = config[config_species]['seed']
        verbose = config[config_species]['verbose']

        # Change to X_reconstructed_S when using the recontructed data
        X_calls_train, X_calls_val,y_train, y_val = data_m.augment_and_prep_data(
            absence_class_label, X, Y, seed, train_size, call_order, verbose)
        # Create a folder to save weights
        if not os.path.exists(save_weights_folder):
            os.makedirs(save_weights_folder)
        # Train the model 10 times and save weights after each iteration
        for i in range(number_of_run):
            filepath= save_weights_folder+'/'+f"Weights_{method_name}"+'/'+f"Weights_{compression_parameter}"+'/'+"{}_{}.hdf5".format(model_name, i)
            checkpoint_callback = callbacks.ModelCheckpoint(filepath,
                                                            monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            self.train_and_evaluate(model_name, X_calls_train, y_train, X_calls_val, y_val,
                                batch_size=batch_size, epochs=epochs, callbacks=checkpoint_callback)
            print(f"Model weights after iteration {i+1} saved.")

    def run_pretrain(self, X, Y, config_species, model_name, method_name, compression_parameter,number_of_run):

        data_m = DataManager()
        #X = np.expand_dims(X, axis=-1)
        X = data_m.reshape_to_rgb(X)
        
        config = self.load_json_config('params.json')
        if config:
            print(config)

        save_weights_folder = 'Saved_weights'

        # Extract parameters from the config
        call_order = config[config_species]['call_order']
        absence_class_label = config[config_species]['absence_class_label']
        batch_size = config[config_species]['batch_size']
        epochs = config[config_species]['epochs']
        train_size = config[config_species]['train_size']
        seed = config[config_species]['seed']
        verbose = config[config_species]['verbose']

        # Change to X_reconstructed_S when using the recontructed data
        X_calls_train, X_calls_val,y_train, y_val = data_m.augment_and_prep_data(
            absence_class_label, X, Y, seed, train_size, call_order, verbose)
        # Create a folder to save weights
        if not os.path.exists(save_weights_folder):
            os.makedirs(save_weights_folder)
        # Train the model 10 times and save weights after each iteration
        for i in range(number_of_run):
            filepath= save_weights_folder+'/'+f"Weights_{method_name}"+'/'+f"Weights_{compression_parameter}"+'/'+"{}_{}.hdf5".format(model_name, i)
            checkpoint_callback = callbacks.ModelCheckpoint(filepath,
                                                            monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            self.train_and_evaluate(model_name, X_calls_train, y_train, X_calls_val, y_val,
                                batch_size=batch_size, epochs=epochs, callbacks=checkpoint_callback)
            print(f"Model weights after iteration {i+1} saved.")

