def numel(w: list):
    out = 1
    for k in w:
        out *= k
    return out


def compute_input_flops(layer, macs=False):
    return 0


def compute_padding_flops(layer, macs=False):
    return 0


def compute_activation_flops(layer, macs=False):
    return 0


def compute_tfop_flops(layer, macs=False):
    return 0


def compute_add_flops(layer, macs=False):
    return 0


def compute_conv2d_flops(layer, macs=False):
    #     _, cin, h, w = input_shape
    if layer.data_format == "channels_first":
        _, input_channels, _, _ = layer.input_shape
        _, output_channels, h, w, = layer.output_shape
    elif layer.data_format == "channels_last":
        _, _, _, input_channels = layer.input_shape
        _, h, w, output_channels = layer.output_shape

    w_h, w_w = layer.kernel_size

    #     flops = h * w * output_channels * input_channels * w_h * w_w / (stride**2)
    flops = h * w * output_channels * input_channels * w_h * w_w

    if not macs:
        #print("conv2d: H={} W={} C_out={} C_in={} K_h={} K_w={} bias={}".format(h, w, output_channels, input_channels, w_h, w_w, layer.use_bias))
        #print(flops)
        # flops_bias = -input_channels*numel(layer.output_shape[1:]) if not layer.use_bias else 0
        flops_bias = 0  # ts profiler seems to ignore them
        flops = 2 * flops + flops_bias
        #print(flops)
    return int(flops)

def compute_depthwiseconv2d_flops(layer, macs=False):
    #     _, cin, h, w = input_shape

    if layer.data_format == "channels_first":
        _, input_channels, _, _ = layer.input_shape
        _, output_channels, h, w, = layer.output_shape
    elif layer.data_format == "channels_last":
        _, _, _, input_channels = layer.input_shape
        _, h, w, output_channels = layer.output_shape

    w_h, w_w = layer.kernel_size

    flops = w_h * w_w * input_channels * h * w

    if not macs:
        #print("depthwise: H={} W={} C_out={} C_in={} K_h={} K_w={} bias={}".format(h, w, output_channels, input_channels, w_h, w_w, layer.use_bias))
        #print(flops)
        # flops_bias = -numel(layer.output_shape[1:]) if not layer.use_bias else 0
        flops_bias = 0  # ts profiler seems to ignore them
        flops = 2 * flops + flops_bias
        #print(flops)

    return flops

def compute_separableconv2d_flops(layer, macs=False):
    #     _, cin, h, w = input_shape
    if layer.data_format == "channels_first":
        _, input_channels, _, _ = layer.input_shape
        _, output_channels, h, w, = layer.output_shape
    elif layer.data_format == "channels_last":
        _, _, _, input_channels = layer.input_shape
        _, h, w, output_channels = layer.output_shape

    w_h, w_w = layer.kernel_size

    flops = output_channels * h * w * (w_h * w_w + input_channels)

    if not macs:
        #print("separable: H={} W={} C_out={} C_in={} K_h={} K_w={} bias={}".format(h, w, output_channels, input_channels, w_h, w_w, layer.use_bias))
        #print(flops)
        # flops_bias = -2*numel(layer.output_shape[1:]) if not layer.use_bias else 0
        flops_bias = 0  # ts profiler seems to ignore them
        flops = 2 * flops + flops_bias
        #print(flops)

    return flops

def compute_fc_flops(layer, macs=False):
    ft_in, ft_out = layer.input_shape[-1], layer.output_shape[-1]
    flops = ft_in * ft_out

    if not macs:
        # flops_bias = ft_out if layer.use_bias else 0
        flops_bias = 0  # ts profiler seems to ignore them
        flops = 2 * flops + flops_bias

    return int(flops)


def compute_bn2d_flops(layer, macs=False):
    # subtract, divide, gamma, beta
    flops = 2 * numel(layer.input_shape[1:])

    if not macs:
        flops *= 2

    return int(flops)


def compute_relu_flops(layer, macs=False):
    flops = 0
    if not macs:
        flops = numel(layer.input_shape[1:])

    return int(flops)


def compute_maxpool2d_flops(layer, macs=False):
    flops = 0
    if not macs:
        flops = layer.pool_size[0] ** 2 * numel(layer.output_shape[1:])

    return flops


def compute_pool2d_flops(layer, macs=False):
    flops = 0
    if not macs:
        flops = layer.pool_size[0] ** 2 * numel(layer.output_shape[1:])

    return flops


def compute_globalavgpool2d_flops(layer, macs=False):
    if layer.data_format == "channels_first":
        _, input_channels, h, w = layer.input_shape
        _, output_channels = layer.output_shape
    elif layer.data_format == "channels_last":
        _, h, w, input_channels = layer.input_shape
        _, output_channels = layer.output_shape

    return h * w


def compute_softmax_flops(layer, macs=False):
    nfeatures = numel(layer.input_shape[1:])

    total_exp = nfeatures  # https://stackoverflow.com/questions/3979942/what-is-the-complexity-real-cost-of-exp-in-cmath-compared-to-a-flop
    total_add = nfeatures - 1
    total_div = nfeatures

    flops = total_div + total_exp

    if not macs:
        flops += total_add

    return flops

def compute_lambda_flops(layer, macs=False):
    # depends from the actual lambda function, we assume 1 operation per output value
    flops = numel(layer.output_shape[1:])
    if not macs:
        flops = 2*flops

    return int(flops)

def compute_flatten_flops(layer, macs=False):
    return 0

def compute_reshape_flops(layer, macs=False):
    return 0

def compute_dropout_flops(layer, macs=False):
    return 0

def compute_concatenate_flops(layer, macs=False):
    return 0
