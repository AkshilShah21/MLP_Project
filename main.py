import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from Models import cnn_softmax, cnn_svm
from Data_Preprocessing import preprocess

def train_model(model, train_data, epochs, batch_size, validation_data):
    history = model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    return model, history

def main():
    print(f"access of tensorflow for hardware:\n{tf.config.list_physical_devices()}")
    print(f"tensorflow version: {tf.__version__}")
    print("Choose the dataset:")
    print("1. MINST")
    print("2. Fashion-MINST")
    print("3. Dogs vs cat")
    dataset_option = int(input("Enter your choice (1/2/3): "))
    dataset_name = None

    if dataset_option == 1:
        dataset_name = 'MINST'

        #load dataset
        (ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True,)

        #PREPROCESS
        ds_train = preprocess.preprocess_train(ds_train, ds_info)
        ds_test = preprocess.preprocess_test(ds_test)

        input_shape=(28, 28, 1)  
        num_classes = 10

    elif dataset_option == 2:
        dataset_name = 'Fashion-MINST'
        (ds_train, ds_test), ds_info = tfds.load('fashion_mnist',split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True,)

        #PREPROCESS
        ds_train = preprocess.preprocess_train(ds_train, ds_info)
        ds_test = preprocess.preprocess_test(ds_test)

        input_shape=(28, 28, 1)  
        num_classes = 10

    elif dataset_option == 3:
        dataset_name = 'Dogs vs cat'
        ds_train, ds_info = tfds.load('cats_vs_dogs', split='train', shuffle_files=True, as_supervised=True, with_info=True)

        #preprocess
        ds_train = ds_train.map(preprocess.image_resize, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = preprocess.preprocess_train(ds_train, ds_info, True)
        ds_test = None

        input_shape=(32, 32, 3)  
        num_classes = 2


    else:
        print('invalid input')

    # Build CNN models
    cnn_softmax_model = cnn_softmax.compiled_cnn_softmax(cnn_softmax.build_cnn_softmax_model(input_shape, num_classes), 1e-3)
    cnn_svm_model = cnn_svm.compiled_cnn_svm(cnn_svm.build_cnn_svm_model(input_shape, num_classes), 1e-3)

    # Train CNN models
    epochs = 200
    cnn_softmax_model, softmax_history = train_model(cnn_softmax_model, ds_train, epochs, batch_size=128, validation_data = ds_test)
    cnn_svm_model, svm_history = train_model(cnn_svm_model, ds_train, epochs, batch_size=128, validation_data = ds_test)

    # Plot comparison
    plt.plot(softmax_history.history['accuracy'], label='CNN-Softmax')
    plt.plot(svm_history.history['accuracy'], label='CNN-SVM')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Comparison for {dataset_name}')
    plt.legend()
    plt.savefig(f"figures/{dataset_name}_accuracy.jpg")
    plt.show()

    plt.plot(softmax_history.history['loss'], label='CNN-Softmax')
    plt.plot(svm_history.history['loss'], label='CNN-SVM')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Comparison for {dataset_name}')
    plt.legend()
    plt.savefig(f"figures/{dataset_name}_loss.jpg")
    plt.show()


if __name__ == "__main__":
    main()
