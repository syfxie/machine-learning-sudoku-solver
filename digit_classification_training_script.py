# import model
from model import SudokuModel

# digits database
from keras.datasets import mnist    
from keras import callbacks
# using the Adam optimizer for computational efficiency and saving memory
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

# this programs will take one argument: the path to the trined digit classification model
# initialize parser
parser = argparse.ArgumentParser()
# add the "model" argument
parser.add_argument("-m", "--model", required=True, help="input path to trained digit classification model")
# turn parsed cmd line argument into a Python dictionary
args = vars(parser.parse_args())
print(args)

# INITIALIZE HYPERPARAMETERS
# learning rate
LEARNING_RATE = 0.01
# training epochs
EPOCHS = 10
# batch size
BATCH_SIZE = 100

# https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras/
# earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)

# load mnist dataset
((X_train, y_train), (X_test, y_test)) = mnist.load_data()

# reshape data
# each image is reshaped into a 28x28x1 shape (28x28 & grayscale)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# scale input features by converting to floats in range [0, 1]
X_train = X_train.astype("float32") / 255.0     
X_test = X_test.astype("float32") / 255.0

# multiple classes classification => use label binarizer to convert input labels into binary labels
y_train = LabelBinarizer.fit_transform(y_train) 
y_test = LabelBinarizer.fit_transform(y_test)

# initialize optimizer
optimizer = Adam(lr=LEARNING_RATE)

# load model to classify digits from 0-9
model = SudokuModel.build(width=28, height=28, depth=1, classes=10)

# compile model for multi-class classification
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# fit model on training data
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

# evaluate model
print("evaluate model on test data")
results = model.evaluate(X_test, y_test, batch_size=128)

# generate predictions
print("generate predictions")
predictions = model.predict(X_test)
# print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(s) for s in LabelBinarizer.classes_]))

# serialize
model.save(args["sudoku_model"], save_format="h5")

# TO-DO: 
# write tensorboard logs and/or for earlystopping during training
# print classification report
# research other optimizers
# adjust model with different hyperparameters
    # configure optimal learning rate
    # option 1: add diagnostic plot of loss over training epochs
    # option 2: sensitivity analysis/grid search
