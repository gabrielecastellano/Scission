# Scission: A Tool for Maximising Performance of Deep Neural Networks in Edge Computing [Fork]

## About the research
Scission is a tool for automated benchmarking of distributed DNNs on a given set of target device, edge and cloud resources for identifying the execution approach and determining the optimal partition for maximising the performance of DNN inference. In doing so, it addresses the following questions:

1. Which combination of potential target hardware resources maximises performance?
2. Which sequence of layers should be distributed across the target resource(s) for maximising DNN performance?
3. How can the performance of DNNs be optimised given user-defined objectives or constraints?

Scission is underpinned by a six-step benchmarking approach (as shown below) that collects benchmark data by executing the DNNs on all target resources and subsequently identifies whether a native or distributed execution approach is most suited for the DNN. 

For distributed execution, it identifies the optimal resource pipeline and partitions measured by the lowest end-to-end latency (compute time on resources and the communication time between resources) of the DNN by: (i) pairing the most computationally intensive layers with capable resources to minimize compute latencies, and at the same time (ii) selecting layers with the least amount of output data as potential end layers of a partition to minimise communication latencies. Thus the decision-making approach in Scission is context-aware by capitalising on the hardware capabilities of the target resources, their locality, the characteristics of DNN layers, and network condition. Scission relies on empirical data and does not estimate performance by making assumptions of the target hardware.

#### Fork
This fork extends Scission with the following features:

- The benchmark script is extended with a modified version of the flopco_keras library to compute per-layer statistics for each model (flops, parameters, output size, layers in a block).
- The benchmark script produces as output plots with the results of the execution of each model on a single device (under the foler [](models_profiling); the plots provide per-layer stats.
- The benchmark script also produces as output a csv summary of the execution of every model on the target device, with information about the layers of each block and their execution time.
- Added a new [](plot_models.py) script that generates a png with the graph representation of each model. Each node of the graph represents a layer of the neural network.
- Each model is now benchmarked on a separate process to avoid keras memory leakage.
- Removed the categories of hosts (device, edge, cloud) and the constraint of two split points: a model can now be split across any number of devices and without any predefined order. **[FEATURE NOT YET IMPLEMENTED]**

### Original research
L. Lockhart, P. Harvey, P. Imai, P. Willis, and B. Varghese, "[Scission: Performance-driven and Context-aware Cloud-Edge Distribution of Deep Neural Networks](https://arxiv.org/pdf/2008.03523.pdf)," Proceedings of the 13th IEEE/ACM International Conference on Utility and Cloud Computing (UCC), 2020, pp. 257-268. . 

## About the software

### Viewing Help
```python3 ./scission_benchmark.py --help```  
```python3 ./scission_predict.py --help```  
```python3 ./plot_models.py --help```  

### Scission benchmarking

#### Dependencies
* Tensorflow 2.1
* NumPy
* Matplotlib
* Pillow

#### Benchmarking

```python3 ./scission_benchmark.py cloud 6700k cat.jpg```

An output file "cloud-6700k(GPU).dat" will be saved which contains the benchmark data for each benchmarked model.

Additionally, two plots for each model will be saved under the [](models_profiling) folder: 

- {MODEL}/cloud-6700k(GPU)-{MODEL}-performance.png with FLOPS and parameters per layer for the given model.
- {MODEL}/{MODEL}-stats.png with FLOPS and parameters per layer for the given model.

Finally, a cloud-6700k(GPU)-stats.csv file is generated with information about all the measured stats for each block of every model, and layer types of each block.

Benchmark plots and a csv report will be saved under the model_profiling folder.

#### Docker

The benchmark can also be run as a docker container:

    docker build -t scission/benchmark .

If you need to fetch the updated code form git, you should specify the CACHEBUST argument as follows:

    docker build -t scission --build-arg CACHEBUST=$(date +%s) .

Run the docker container with:

    docker run -it --rm -v $(pwd):/opt/output scission/benchmark cloud 6700k cat.jpg -s False -o /opt/output

Due to the volume bound at /opt/output, all the output files can be found on the usual locations after the run.

### Scission prediction

#### Dependencies
* NumPy
* Matplotlib

#### Network Statistics 

Predictions use externally provided network statistics in a csv file formatted as below:

```
source,     destination,     ping(ms),      bandwidth(mbps)
device,     edge,            66.7,          1.6
device,     cloud,           100,           1.6
edge,       cloud,           20,            50
```

#### Querying - Criteria Format

* A number to indicate that the layer with the corresponding number must be executed on that platform.
* A number preceded by a "!" to indicate that the layer with corresponding number must not be executed on that platform.
* A string to indicate that only the system with the specified name is used for that platform.
* A string predceded by a "!" to indicate that the specified system must not be used. 

Criteria | Explanation 
-------- | ---------- 
-d “3,!10,device1” | Layer 3 must be executed on device, layer 10 must not. The device with name "device1" must be used.
-e “-1” | The edge must not be used.
-c “30,!cloud1” | Layer 30 must be executed on the cloud, the device with name “cloud1” must not be used.

#### Prediction

```python3 ./scission_predict.py benchmark_data network_stats.csv vgg19 -d "!-1" -e 10 -c 16```

The specified benchmark_data folder must contain data files produced by "scission_benchmark" and a network statistics file which contains network connection information for each of the systems benchmarked. An example folder containing benchmark files and network statistic files has been provided ("benchmark_data").

The fastest configurations that fit the specified criteria will be printed, additionally a graph of the fastest configuration showing per layer latencies and network overheads will be saved to the working directory.

The configurations are printed in the format: 
```
End-to-End latency - Total bandwidth used across the edge pipeline - Layer distribution
```

The output from the above prediction is displayed below:

```
[1] 9.0106s - 2.4087MB - Device(raspi4) = 0 - 6, Edge(vm2Core) = 7 - 11, Cloud(6700k(GPU)) = 12 - 25
[2] 9.0594s - 2.4087MB - Device(raspi4) = 0 - 6, Edge(2500) = 7 - 11, Cloud(6700k(GPU)) = 12 - 25
[3] 9.0721s - 2.4087MB - Device(raspi4) = 0 - 6, Edge(vm2Core) = 7 - 11, Cloud(6700k(NoGPU)) = 12 - 25
[4] 9.1209s - 2.4087MB - Device(raspi4) = 0 - 6, Edge(2500) = 7 - 11, Cloud(6700k(NoGPU)) = 12 - 25
[5] 9.1511s - 3.2115MB - Device(raspi4) = 0 - 6, Edge(vm2Core) = 7 - 12, Cloud(6700k(GPU)) = 13 - 25
```

### Model Graphs

#### Dependencies
* Tensorflow 2.1
* NumPy

#### Usage

    python3 ./plot_models.py [-m MODEL]

Output files are generated under the [](models_profiling) folder.

