# Scission Benchmark
# Benchmarks keras DNNs to create layer profiles, output to benchmark data file to be used with Scission Predict
# Author: Luke Lockhart

import argparse
import os
import pickle
import shutil
import time
import csv
from collections.abc import Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from matplotlib.ticker import ScalarFormatter
from pathlib import Path

from flopco_keras.flopco_keras import FlopCoKeras

from multiprocessing import Process, Manager


class ModelResult:
    def __init__(self):
        self.model = ""
        self.layer_count = 0
        self.load_time = 0
        self.preprocess_time = 0
        self.first_prediction = 0
        self.second_prediction = 0
        self.flops_per_layer = []
        self.macs_per_layer = []
        self.params_per_layer = []
        self.layers = []


class LayerBenchmark:
    def __init__(self):
        self.model = ""
        self.device = ""
        self.input_layer = 0
        self.output_layer = 0
        self.second_prediction = 0
        self.input_size = 0
        self.output_size = 0
        self.params = 0
        self.flops = 0
        self.macs = 0
        self.layers = []


def get_per_layer_stats(model):
    stats = FlopCoKeras(model)
    return stats


'''
def estimate_flops(model):
    # use the tensorflow library to calculate the number of flops (slow).
    return get_flops(model, batch_size=1)
'''


# Recursively gets the output of a layer, used to build up a submodel
def get_output_of_layer(layer, new_input, starting_layer_name):
    global layer_outputs
    if layer.name in layer_outputs:
        return layer_outputs[layer.name]

    if layer.name == starting_layer_name:
        out = layer(new_input)
        layer_outputs[layer.name] = out
        return out

    prev_layers = []
    for node in layer._inbound_nodes:
        if isinstance(node.inbound_layers, Iterable):
            prev_layers.extend(node.inbound_layers)
        else:
            prev_layers.append(node.inbound_layers)

    pl_outs = []
    for pl in prev_layers:
        pl_outs.extend([get_output_of_layer(pl, new_input, starting_layer_name)])

    out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
    layer_outputs[layer.name] = out
    return out


# Returns a submodel for a specified input and output layer
def get_model(selected_model, input_layer: int, output_layer: int):

    layer_number = input_layer
    starting_layer_name = selected_model.layers[layer_number].name

    if input_layer == 0:
        new_input = selected_model.input

        return models.Model(new_input, selected_model.layers[output_layer].output)
    else:
        new_input = layers.Input(batch_shape=selected_model.get_layer(starting_layer_name).get_input_shape_at(0))

    new_output = get_output_of_layer(selected_model.layers[output_layer], new_input, starting_layer_name)
    model = models.Model(new_input, new_output)

    return model


# Navigates the model structure to find regions without parallel paths, returns valid split locations
def create_valid_splits(model):

    layer_index = 1
    multi_output_count = 0

    valid_splits = [0]
    for layer in model.layers[1:]:

        if len(layer._outbound_nodes) > 1:
            multi_output_count += len(layer._outbound_nodes) - 1

        if type(layer._inbound_nodes[0].inbound_layers) == list:
            if len(layer._inbound_nodes[0].inbound_layers) > 1:
                multi_output_count -= (
                        len(layer._inbound_nodes[0].inbound_layers) - 1)

        if multi_output_count == 0:
            valid_splits.append(layer_index)

        layer_index += 1

    return valid_splits


# Pre-processes the input image for a specific model
def preprocess_input(application, batch_size=1):
    global input_image
    image = None
    file_name = input_image
    data_folder = Path("")
    file_to_open = data_folder / file_name

    if application == "xception":
        image = load_img(file_to_open, target_size=(299, 299))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_xception(image)
    elif application == "vgg16":
        image = load_img(file_to_open, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_vgg16(image)
    elif application == "vgg19":
        image = load_img(file_to_open, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_vgg19(image)
    elif application == "resnet50" or application == "resnet101" or application == "resnet152":
        image = load_img(file_to_open, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_resnet(image)
    elif application == "resnet50v2" or application == "resnet101v2" or application == "resnet152v2":
        image = load_img(file_to_open, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_resnetV2(image)
    elif application == "inception_v3":
        image = load_img(file_to_open, target_size=(299, 299))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_inceptionv3(image)
    elif application == "inceptionresnet_v2":
        image = load_img(file_to_open, target_size=(299, 299))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_inceptionresnetv2(image)
    elif application == "mobilenet":
        image = load_img(file_to_open, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_mobilenet(image)
    elif application == "mobilenetv2":
        image = load_img(file_to_open, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_vgg16(image)
    elif application == "densenet121" or application == "densenet169" or application == "densenet201":
        image = load_img(file_to_open, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_densenet(image)
    elif application == "nasnetlarge":
        image = load_img(file_to_open, target_size=(331, 331))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_nasnetlarge(image)
    elif application == "nasnetmobile":
        image = load_img(file_to_open, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = np.tile(image, (batch_size, 1, 1, 1))
        image = preprocess_input_nasnetmobile(image)
    else:
        print("[+] Application not found.")

    return image


'''
def predict(model, _input, batch_size=1):
    if batch_size == 1:
        return model(_input, training=False)
    else:
        return model.predict(input, batch_size=batch_size)
'''


def benchmark_application(application, result_dict, batch_size=1):
    global reference_prediction

    start_application = time.time()

    # Loading Model Start
    start_load = time.time()
    selected_model = model_dict[application]()
    total_load = time.time() - start_load

    # Load Model End

    layer_outputs = {}

    print("[+] " + application + " - " + " Layers: " + str(len(selected_model.layers)) + " - Split Points: " + str(
        len(create_valid_splits(selected_model)) - 1) + " - Loading took: " + str(total_load))

    # Normal Execution Benchmark Start
    start_normal = time.time()
    normal_result = benchmark_normal_execution(selected_model, application, batch_size)
    total_normal = time.time() - start_normal
    print(f" - {total_normal}")

    normal_result.load_time = total_load
    # Normal Execution Benchmark End

    # Individual Layer Benchmark Start
    start_individual = time.time()
    individual_results = benchmark_individual_execution(selected_model, application, batch_size)
    total_individual = time.time() - start_individual

    print(f" - {total_individual}")
    # Individual Layer Benchmark End

    # Calculate difference between normal execution and the summation of individual layers
    total_exec = 0
    total_flops = 0
    total_macs = 0
    for result in individual_results:
        total_exec += result.second_prediction
        # add layer stats to individual block stats
        for flops in normal_result.flops_per_layer[result.input_layer:result.output_layer+1]:
            result.flops += flops
        for macs in normal_result.macs_per_layer[result.input_layer:result.output_layer+1]:
            result.macs += macs
        for params in normal_result.params_per_layer[result.input_layer:result.output_layer + 1]:
            result.params += params
        for layer in normal_result.layers[result.input_layer:result.output_layer + 1]:
            result.layers.append(layer)

    # execution time
    extra = ((total_exec / normal_result.second_prediction) * 100) - 100
    print(f"[-] NE: {normal_result.second_prediction} - SUM: {total_exec} - % Change: {extra}")

    total_application = time.time() - start_application
    print(f"[+] Benchmarking {application} took: {total_application} \n")

    result_dict[application] = individual_results

    # create_individual_graphs(selected_model, application, individual_results)

    reference_prediction = None
    K.clear_session()
    if not store_models:
        model_dir = Path(os.path.expanduser("~/.keras/models"))
        shutil.rmtree(model_dir)


# Benchmarks the normal execution of a DNN - entire model unmodified
def benchmark_normal_execution(selected_model, application, batch_size=1):
    global number_of_repeats
    global reference_prediction

    normal_result = ModelResult()

    normal_result.model = application
    normal_result.layer_count = len(selected_model.layers)

    print("[-] Benchmarking normal execution", end='', flush=True)

    start_preprocess = time.time()
    image = preprocess_input(application, batch_size=batch_size)
    total_preprocess = time.time() - start_preprocess
    normal_result.preprocess_time = total_preprocess

    start_first = time.time()
    selected_model.predict(image, batch_size=batch_size)
    # selected_model(image, training=False)
    total_first = time.time() - start_first
    normal_result.first_prediction = total_first

    # perform some runs to warm up the model
    for _ in range(10):
        selected_model.predict(image, batch_size=batch_size)

    t = []
    for _ in range(number_of_repeats):
        start_second = time.time()
        reference_prediction = selected_model.predict(image, batch_size=batch_size)
        t.append(time.time() - start_second)

    normal_result.second_prediction = np.percentile(t, 50)
    per_layer_stats = get_per_layer_stats(selected_model)
    normal_result.flops_per_layer = per_layer_stats.flops
    normal_result.macs_per_layer = per_layer_stats.macs
    normal_result.params_per_layer = per_layer_stats.params
    normal_result.layers = per_layer_stats.layers

    return normal_result


# Benchmarks the indiviual layers/blocks of a DNN
def benchmark_individual_execution(selected_model, application, batch_size=1):
    global number_of_repeats
    global reference_prediction

    print("[-] Benchmarking individual layers", end='', flush=True)

    individual_results = []

    image = preprocess_input(application, batch_size=batch_size)
    np.save("fsize", image)
    input_size = os.stat("fsize.npy").st_size
    os.remove("fsize.npy")

    input_tensor = image

    split_points = create_valid_splits(selected_model)

    for index, split_point in enumerate(split_points):

        result = LayerBenchmark()
        result.model = application

        if index == 0:
            first_point = 0
            result.input_size = input_size
        else:
            first_point = split_points[index - 1] + 1
            result.input_size = individual_results[-1].output_size

        result.input_layer = first_point
        result.output_layer = split_point

        new_model = get_model(selected_model, first_point, split_point)

        output_tensor = new_model.predict(input_tensor, batch_size=batch_size)

        # perform some runs to warm up the model
        for _ in range(10):
            new_model.predict(input_tensor, batch_size=batch_size)

        t = []
        for x in range(number_of_repeats):
            start_second = time.time()
            new_model.predict(input_tensor, batch_size=batch_size)
            t.append(time.time() - start_second)

        result.second_prediction = np.percentile(t, 50)

        np.save("fsize", output_tensor)
        result.output_size = os.stat("fsize.npy").st_size
        os.remove("fsize.npy")

        individual_results.append(result)
        del result

        input_tensor = output_tensor

    if reference_prediction.all() != output_tensor.all():
        print("[WARNING] PREDICATION ACCURACY DIFFERS FROM NORMAL EXECUTION [WARNING]")

    return individual_results


# Creates graphs showing the per layer execution time and output size of a DNN
def create_individual_graphs(selected_model, application, individual_results):
    global device_type

    plt.rcParams.update({'font.size': 35})

    file_path = "results/individual graphs/"

    Path(file_path).mkdir(parents=True, exist_ok=True)

    bars = []
    execution_times = []
    output_sizes = []

    result: LayerBenchmark
    for result in individual_results:

        if result.input_layer is result.output_layer:
            label = str(result.input_layer)
            bars.append(label)
        else:
            label = str(result.input_layer) + "-" + str(result.output_layer)
            bars.append(label)

        execution_times.append(result.second_prediction * 1000)
        output_sizes.append(result.output_size / 1024 / 1024)

    width = 0.8
    ind = np.arange(len(bars))

    fig, ax1 = plt.subplots(figsize=(20, 10))

    color1 = "tab:blue"
    ax1.set_xlabel("Layer no.", labelpad=20)
    ax1.set_ylabel("Execution time (ms)", color=color1, labelpad=10)
    ax1.bar(ind, execution_times, width, color=color1)

    color2 = "tab:red"
    ax2 = ax1.twinx()
    ax2.set_ylabel("Output size (MB)", color=color2, labelpad=10)
    ax2.plot(ind, output_sizes, width, color=color2, linewidth=4)
    ax2.set_xticks(ind)
    ax2.set_xticklabels(bars)

    from tensorflow.keras.utils import plot_model

    file_path = "results/model graphs/"
    Path(file_path).mkdir(parents=True, exist_ok=True)
    file_to_open = Path("results/model graphs/") / (application + "-" + "Graphs" + ".png")
    plot_model(selected_model, to_file=file_to_open)

    file_to_open = Path("results/individual graphs/") / (application + "-" + device_type + ".png")
    plt.setp(ax1.get_xticklabels(), rotation=270, horizontalalignment='center')
    plt.savefig(file_to_open, bbox_inches='tight')


# Pickles and saves the benchmark data
def save_data(folder, data, device_type, device_name, batch_size):
    file_name = f"{folder}/benchmark_data/{device_type}-{device_name}-b{batch_size}.dat"
    print(f"Saving benchmark data at '{file_name}'...")
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
if __name__ == "__main__":
    # Parse Args

    parser = argparse.ArgumentParser(description="Scission benchmarking for Keras Models")

    parser.add_argument('platform', action='store', type=str, help="Platform Type (e.g. cloud/edge/device)")
    parser.add_argument('name', action='store', type=str, help="Platform Name (e.g. highPerformanceEdge)")
    parser.add_argument('input', action='store', type=str, help="Input image filename (e.g. cat.jpeg)")
    parser.add_argument('-r', '-repeats', dest='repeats', action='store', type=int, required=False,
                        help="Number of repeats for averaging (default: 10)")
    parser.add_argument('-dc', '--disablecuda', dest='cuda', action='store', type=str, required=False,
                        help="Disable cuda (default: False)")
    parser.add_argument('-s', '--store-models', dest='store_models', action='store', type=str2bool, required=False,
                        default=True, help="Keep keras models on disk after execution (default: True)")
    parser.add_argument('-o', '--output-directory', dest='output_dir', action='store', type=str, required=False,
                        help="If given, uses it as alternative base output folder (default: the project root folder)")
    parser.add_argument('-b', '--batch-size', dest='batch_size', action='store', type=int, required=False, default=1,
                        help="inference batch size (default: 1)")

    args = parser.parse_args()

    if args.repeats is not None:
        number_of_repeats = args.repeats
    else:
        number_of_repeats = 10

    if args.cuda is not None:
        disable_cuda = str2bool(args.cuda)
    else:
        disable_cuda = False

    if args.store_models is not None and args.store_models is False:
        store_models = False
    else:
        store_models = True

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = None

    batch_size = args.batch_size

    device_type = args.platform
    device_name = args.name
    input_image = args.input

    if disable_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # End Parse Args

    start_entire = time.time()

    manager = Manager()
    result_dict = manager.dict()

    reference_prediction = None

    for application in model_dict:
        while application not in result_dict:
            layer_outputs = {}

            p = Process(target=benchmark_application, args=(application, result_dict, batch_size))
            p.start()
            p.join()

        '''
        start_application = time.time()

        normal_result = ModelResult()

        # Loading Model Start
        start_load = time.time()
        selected_model = model_dict[application]()
        total_load = time.time() - start_load
        normal_result.load_time = total_load
        # Load Model End

        layer_outputs = {}

        print("[+] " + application + " - " + " Layers: " + str(len(selected_model.layers)) + " - Split Points: " + str(
            len(create_valid_splits()) - 1) + " - Loading took: " + str(total_load))

        # Normal Execution Benchmark Start
        start_normal = time.time()
        benchmark_normal_execution(selected_model, application)
        total_normal = time.time() - start_normal

        print(f" - {total_normal}")
        # Normal Execution Benchmark End

        # Individual Layer Benchmark Start
        start_individual = time.time()
        benchmark_individual_execution(application)
        total_individual = time.time() - start_individual

        result_dict[application] = individual_results
        print(f" - {total_individual}")
        # Individual Layer Benchmark End

        # Calculate difference between normal execution and the summation of individual layers
        total_exec = 0
        total_flops = 0
        total_macs = 0
        for result in individual_results:
            total_exec += result.second_prediction
            # add layer stats to individual block stats
            for flops in normal_result.flops_per_layer[result.input_layer:result.output_layer+1]:
                result.flops += flops
            for macs in normal_result.macs_per_layer[result.input_layer:result.output_layer+1]:
                result.macs += macs
            for params in normal_result.params_per_layer[result.input_layer:result.output_layer + 1]:
                result.params += params
            for layer in normal_result.layers[result.input_layer:result.output_layer + 1]:
                result.layers.append(layer)

        # execution time
        extra = ((total_exec / normal_result.second_prediction) * 100) - 100
        print(f"[-] NE: {normal_result.second_prediction} - SUM: {total_exec} - % Change: {extra}")

        total_application = time.time() - start_application
        print(f"[+] Benchmarking {application} took: {total_application} \n")

        # create_individual_graphs(selected_model, application, individual_results)

        selected_model = None
        layer_outputs = {}
        individual_results = []
        reference_prediction = None
        K.clear_session()
        '''

    # plot results
    if output_dir is not None:
        pass
        dname = os.path.abspath(output_dir)
    else:
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
    Path(Path(dname) / "models_profiling").mkdir(parents=True, exist_ok=True)
    file_to_open = Path(dname) / "models_profiling" / f"{device_name}-stats-b{batch_size}.csv"
    with open(file_to_open, mode='w') as stats_file:
        stats_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for application, results in result_dict.items():
            output_sizes = [result.output_size/1024/1024 for result in results]
            flops = [result.flops for result in results]
            macs = [result.macs for result in results]
            params = [result.params for result in results]
            times = [result.second_prediction*1000 for result in results]
            layers = [result.layers for result in results]
            bars = [f"{result.input_layer}-{result.output_layer}" if result.input_layer != result.output_layer else f"{result.input_layer}" for result in results]

            stats_writer.writerow([application])
            stats_writer.writerow(["layers"] + bars)
            stats_writer.writerow(["flops"] + flops)
            stats_writer.writerow(["params"] + params)
            stats_writer.writerow(["output"] + output_sizes)
            stats_writer.writerow(["times"] + times)
            stats_writer.writerow([])
            stats_writer.writerow(["block details:"])
            for i in range(len(layers)):
                stats_writer.writerow([bars[i]] + layers[i])
            stats_writer.writerow([])
            stats_writer.writerow([])

            plt.rcParams.update({'font.size': 35})
            width = 0.8

            # times/output_size plot
            ind = np.arange(len(bars))
            ratio = max(len(ind)/25, 1)
            fig, ax1 = plt.subplots(figsize=(20*ratio, 10))
            ax2 = ax1.twinx()
            ax1.set_ylabel("Execution time (ms)", labelpad=10, color='tab:blue')
            ax1.bar(ind, times, width, align='center', color='tab:blue')
            ax1.set_xticks(ind)
            ax1.set_xticklabels(bars)
            ax2.set_ylabel("Output size (MB)", labelpad=10, color='tab:red')
            ax2.plot(ind, output_sizes, width, color='tab:red', linewidth=3)
            ax2.set_xticks(ind)
            ax2.set_xticklabels(bars)

            #plt.yscale("log")
            plt.setp(ax1.get_xticklabels(), rotation=270, horizontalalignment='center')
            ax1.margins(x=1/len(ind))
            ax2.margins(x=1/len(ind))

            for axis in [ax1.yaxis]:
                axis.set_major_formatter(ScalarFormatter())

            Path(Path(dname) / "models_profiling" / application).mkdir(parents=True, exist_ok=True)
            file_to_open = Path(dname) / "models_profiling" / application / f"{device_name}-{application}-performance-b{batch_size}.png"
            #plt.tight_layout()
            #ratio = len(ind)/25
            #plt.figure(figsize=(1*ratio, 1))
            plt.savefig(file_to_open, bbox_inches='tight')
            plt.close(fig)
            plt.cla()

            # flops/params plot
            width = 0.4
            ind = np.arange(len(bars))
            _, ax1 = plt.subplots(figsize=(20*ratio, 10))
            ax2 = ax1.twinx()
            ax1.set_ylabel("FLOPs", labelpad=10, color='tab:blue')
            ax1.bar(ind, flops, width, align='center', color='tab:blue')
            ax1.set_xticks(ind)
            ax1.set_xticklabels(bars)
            ax2.set_ylabel("Parameters", labelpad=10, color='tab:orange')
            ax2.bar(ind+width, params, width, align='center', color='tab:orange')
            ax2.set_xticks(ind+width/2)
            ax2.set_xticklabels(bars)

            # plt.yscale("log")
            plt.setp(ax1.get_xticklabels(), rotation=270, horizontalalignment='center')
            ax1.margins(x=1/len(ind))
            ax2.margins(x=1/len(ind))

            for axis in [ax1.yaxis]:
                axis.set_major_formatter(ScalarFormatter())

            file_to_open = Path(dname) / "models_profiling" / application / f"{application}-stats.png"
            plt.autoscale(enable=True)
            plt.savefig(file_to_open, bbox_inches='tight')
            plt.close(fig)
            plt.cla()

    save_data(dname, result_dict, device_type, device_name, batch_size)

    total_entire = time.time() - start_entire
    print(f"[+] Benchmarking took: {total_entire}")

# Script End
