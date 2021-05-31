import numpy as np
from collections import defaultdict
from functools import partial
import copy

# from flopco_keras.compute_layer_flops import *
from flopco_keras.compute_layer_flops import *

class FlopCoKeras():

    def __init__(self, model):
        '''
        instances: list of layer types,
            supported types are [nn.Conv2d, nn.Linear,
            nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.Softmax]
        '''
        self.model = model

        self.flops = []
        self.macs = []
        self.params = []
        self.layers = []
        self.get_flops = {
            'ReLU': compute_relu_flops,
            'InputLayer': compute_input_flops,
            'Conv2D': compute_conv2d_flops,
            'ZeroPadding2D': compute_padding_flops,
            'DepthwiseConv2D': compute_depthwiseconv2d_flops,
            'SeparableConv2D': compute_separableconv2d_flops,
            'Activation': compute_activation_flops,
            'Dense': compute_fc_flops,
            'BatchNormalization': compute_bn2d_flops,
            'TensorFlowOpLayer': compute_tfop_flops,
            'MaxPooling2D': compute_pool2d_flops,
            'AveragePooling2D': compute_pool2d_flops,
            'GlobalAveragePooling2D': compute_globalavgpool2d_flops,
            'Add': compute_add_flops,
            'Flatten': compute_flatten_flops,
            'Reshape': compute_reshape_flops,
            'Cropping2D': compute_reshape_flops,
            'Dropout': compute_dropout_flops,
            'Concatenate': compute_concatenate_flops,
            'Lambda': compute_lambda_flops}

        self.get_stats(flops=True, macs=True, params=True)

        self.total_flops = sum(self.flops)
        self.total_macs = sum(self.macs)
        self.total_params = sum(self.params)

        #self.relative_flops = [k / self.total_flops for k in self.flops]

        #self.relative_macs = [k / self.total_macs for k in self.macs]

        # self.relative_params = [k/self.total_params for k in self.params] #TO DO

        del self.model

    def __str__(self):
        print_info = "\n".join([str({k: v}) for k, v in self.__dict__.items()])

        return str(self.__class__) + ": \n" + print_info

        # def count_params(self):

    #     self.params = [0]
    # self.params = defaultdict(int)

    # for mname, m in self.model.named_modules():
    #     if m.__class__ in self.instances:

    #         self.params[mname] = 0

    #         for p in m.parameters():
    #             self.params[mname] += p.numel()

    def _save_layer_name(self, layer):
        self.layers.append(layer.__class__.__name__)

    def _save_params(self, layer):
        params = 0
        for weights in layer.weights:
            params += numel(weights.shape)
        self.params.append(params)
        # self.params.append(layer.count_params())

    def _save_flops(self, layer, macs=False):
        flops = self.get_flops[layer.__class__.__name__](layer, macs)
        if macs:
            self.macs.append(flops)
        else:
            #("{} {}".format(layer.__class__.__name__, flops))
            self.flops.append(flops)

    def get_stats(self, flops=False, macs=False, params=False):

        #i = 0
        self.layers = []
        if flops:
            self.flops = []
        if macs:
            self.flops = []
        if params:
            self.flops = []
        for layer in self.model.layers:
            self._save_layer_name(layer)
            if flops:
                #print("layer: " + str(i))
                self._save_flops(layer)
            if macs:
                self._save_flops(layer, macs=True)
            if params:
                self._save_params(layer)
            #i += 1
