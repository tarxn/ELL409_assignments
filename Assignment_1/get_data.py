## Handle the necessary imports
import numpy as np 
import gzip

## Function to read the data from the compressed files
def read_data():
    train_images = gzip.open('train-images-idx3-ubyte.gz','r')
    train_labels = gzip.open('train-labels-idx1-ubyte.gz','r')
    image_size = 28
    num_images = 60000
    train_images.read(16)
    buffer = train_images.read(image_size * image_size * num_images)
    data_train_image = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
    data_train_image = data_train_image.reshape(num_images, image_size, image_size, 1)
    y = [] 
    train_labels.read(8)
    for i in range(num_images):   
        buf = train_labels.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        y.append(labels[0])
    y = np.array(y)
    X = []
    for i in data_train_image:
        xi = np.asarray(i).squeeze()
        X.append(xi.flatten())
    X = np.array(X)
    return X, y

## Function to get subset of data with labels 2 and 9
def get_subset(X, y):
    indices = np.where((y == 2) | (y == 9))
    X_subset = X[indices]
    y_subset = y[indices]
    return X_subset, y_subset

## In case you need the whole data, use this:
X, y = read_data()

## In case you need the subset of data with classes 2 and 9 use this:
X_subset, y_subset = get_subset(X, y)

np.save("train_images_subset.npy", X_subset)
np.save("train_labels_subset.npy", y_subset)

np.save("train_images.npy", X)
np.save("train_labels.npy", y)