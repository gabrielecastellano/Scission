# PLOT MODELS
# Plot keras DNNs to layer by layer as Directed Acyclic Graphs.
# Author: Gabriele Castellano

import argparse
import os
import pickle
import time
import csv
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inceptionresnetv2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnetlarge
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnetmobile
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

from flopco_keras.flopco_keras import FlopCoKeras

model_dict = {
    "vgg19": VGG19,
    "xception": Xception,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "resnet50v2": ResNet50V2,
    "resnet101v2": ResNet101V2,
    "resnet152v2": ResNet152V2,
    "inception_v3": InceptionV3,
    "inceptionresnet_v2": InceptionResNetV2,
    "mobilenet": MobileNet,
    "vgg16": VGG16,
    "mobilenetv2": MobileNetV2,
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
    "nasnetlarge": NASNetLarge,
    "nasnetmobile": NASNetMobile
}

# Script Start

# Parse Args

parser = argparse.ArgumentParser(description="Viewer for Keras Models")

parser.add_argument('-m', '-model', dest='model', action='store', type=str, required=False,
                    help="Model Name (e.g. highPerformanceEdge)")


args = parser.parse_args()


if args.model is not None:
    tmp = model_dict[args.model]
    model_dict.clear()
    model_dict[args.model] = tmp
    

# End Parse Args

start_entire = time.time()

individual_results = []
normal_results = []
result_dict = {}

reference_prediction = None

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

for application in model_dict:

    model = model_dict[application]()

    Path(Path(dname) / "models_profiling" / application).mkdir(parents=True, exist_ok=True)
    file_to_open = Path(dname) / "models_profiling" / application / f"{application}-graph.png"
    tf.keras.utils.plot_model(model, to_file=file_to_open, show_shapes=True)

# Script End
