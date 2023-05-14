import tensorflow
# create layer by layer CNN model
from keras.models import Sequential
from keras.layers import Dense
# spatial convolution over images
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout

# create CNN model
# multiple classes involved => use softmax for probability distribution
class SudokuModel:
    # method that doesn't require an instance of SudokuBoard to be accessd
    # no "self" parameter required
    @staticmethod
    # all dimensions are of MNIST digits (pixels and grayscale channels)
    # classes = number of digits to recognize (0-9)
    def build_model(width, height, depth, classes):
        # initilize sequential model (stack of layers)
        model = Sequential()
        # input shape for first layer of OCR model
        inputShape = (height, width, depth)

        # incrementally building the rest of the CNN model

        # first stack:

        # convolution layer (set input shape)
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same", input_shape=inputShape))
        # RELU layer
        model.add(Activation('relu'))
        # max pooling layer
        # sharpen the feature map
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        # second layer: RELU and Max pooling
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        # build FC layers
        # flatten to one-dimensional layer
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        # 50% dropout
        model.add(Dropout(0.5))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        # softmax classifier
        model.add(Activation('softmax'))

        return model



