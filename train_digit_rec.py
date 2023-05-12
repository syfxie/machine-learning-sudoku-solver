from cnn_model import SudokuNet
from tensorflow import keraa
# digits database
from keras.datasets import mnist                
from sklearn.preprocessing import LabelEncoder
import argparse

# look up optimizers

# create parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="input path to trained model")
# parse arguments
args = vars(parser.parse_args())

# initial learning rate

