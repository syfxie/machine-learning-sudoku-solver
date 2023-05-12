from cnn_model import SudokuModel
from tensorflow import keras
# digits database
from keras.datasets import mnist     
from keras import callbacks    
from keras import optimizers 
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import argparse

# look up optimizers

# create parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="input path to trained model")
# parse arguments
args = vars(parser.parse_args())

# initialize hyperparameters
# learning rate
LEARNING_RATE = 0.01       # note: try different learning rates
# training epochs
EPOCHS = 10
# batch size
BATCH_SIZE = 64

# configure optimal learning rate
# option 1: add diagnostic plot of loss over training epochs
# option 2: sensitivity analysis/grid search

# https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras/
# earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)

# RETRIEVE MNIST DATASET
((X_train, y_train), (X_test, y_test)) = mnist.load_data()

# reshape data
# each image is reshaped into a 28x28x1 shape (28x28 and grayscale)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shapd[0], 28, 28, 1))

# scale input features
X_train = X_train.astype("float32") / 255.0     
X_test = X_test.astype("float32") / 255.0

# multiple classes classification => use label binarizer to convert input labels into binary labels
y_train = LabelBinarizer.fit_transform(y_train) 
y_test = LabelBinarizer.fit_transform(y_test)

# initialize optimizer and model
optimizer = optimizers.Adam(lr=LEARNING_RATE)

# Initilize CNN model
model = SudokuModel.build(width=28, height=28, depth=1, classes=10)
# compile for multi-class classification
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# optimizer = optimizers.SGD(lr=LEARNING_RATE)

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

