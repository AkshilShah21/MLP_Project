# main.py

import os
import numpy as np
from Models import cnn_softmax, cnn_svm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from Data_Preprocessing import preprocess

# train model
def train_model(model, train_data, validation_data, epochs, batch_size):
    history = model.fit(train_data, epochs=epochs, validation_data=validation_data, batch_size=batch_size )
    return model, history

def main():
    print("Choose the dataset:")
    print("1. MINST")
    print("2. Fashion-MINST")
    dataset_option = int(input("Enter your choice (1/2): "))

    if dataset_option == 1 or dataset_option == 2:
        # Load and preprocess dataset
        if dataset_option == 1:
            (ds_train, ds_test), ds_info = tfds.load('mnist',split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True,)
            dataset_name = 'MINST'
        else:
            (ds_train, ds_test), ds_info = tfds.load('fashion_mnist',split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True,)
            dataset_name = 'MINST'
       

        #PREPROCESS
        ds_train = preprocess.preprocess_train(ds_train, ds_info)
        ds_test = preprocess.preprocess_test(ds_test)

        # Build CNN models
        input_shape=(28, 28, 1)  
        num_classes = 10
        cnn_softmax_model = cnn_softmax.compiled_cnn_softmax(cnn_softmax.build_cnn_softmax_model(input_shape, num_classes), 1e-3)
        cnn_svm_model = cnn_svm.compiled_cnn_svm(cnn_svm.build_cnn_svm_model(input_shape, num_classes), 1e-3)

        # Train CNN models
        epochs = 15  # You can adjust the number of epochs
        cnn_softmax_model, softmax_history = train_model(cnn_softmax_model, ds_train, ds_test, epochs, batch_size=128)
        cnn_svm_model, svm_history = train_model(cnn_svm_model, ds_train, ds_test, epochs, batch_size=128)

        # Print comparison between CNN-Softmax and CNN-SVM
        print("Comparison between CNN-Softmax and CNN-SVM:")
        print("CNN-Softmax:", softmax_history.history)
        print("CNN-SVM:", svm_history.history)

        # Plot comparison
        fig = plt.figure() 
        plt.plot(softmax_history.history['accuracy'], label='CNN-Softmax')
        plt.plot(svm_history.history['accuracy'], label='CNN-SVM')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Comparison for {dataset_name}')
        plt.legend()
        plt.savefig(f"figures/{dataset_name}_accuracy.jpg")
        plt.show()
        fig.show()

        fig = plt.figure() 
        plt.plot(softmax_history.history['loss'], label='CNN-Softmax')
        plt.plot(svm_history.history['loss'], label='CNN-SVM')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Loss Comparison for {dataset_name}')
        plt.legend()
        plt.savefig(f"figures/{dataset_name}_loss.jpg")
        plt.show()
        fig.show()


if __name__ == "__main__":
    main()
