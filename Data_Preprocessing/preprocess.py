import tensorflow as tf

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.,tf.one_hot(label, depth=10)

def normalize_img_binary(image, label):
    return tf.cast(image, tf.float32) / 255.,tf.one_hot(label, depth=2)

def image_resize(image, label):
    return tf.image.resize(image, (32, 32)), label

def preprocess_train(ds_train, ds_info, binary = False):
    if binary:
        ds_train = ds_train.map(normalize_img_binary, num_parallel_calls=tf.data.AUTOTUNE)
    else: 
        ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    return ds_train

def preprocess_test(ds_test):
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return ds_test