# vmp-for-svae
Variational Message Passing for Structured VAE (Code for the [ICLR 2018 paper](https://openreview.net/pdf?id=HyH9lbZAW) by Wu Lin, Nicolas Hubacher and Mohammad Emtiyaz Khan)


## Getting Started
Before running our code, create a [conda](https://conda.io/docs/user-guide/getting-started.html "Getting started with conda") environment using the file `environment.yml`. To do so, open a terminal and run:
```conda env create -f environment.yml```

Then, activate the created environment:
```source activate san-cpu-env```

If you don't want to use conda, just make sure to use the libraries listed in `environment.yml` in their specified version (most importantly use TensorFlow version 1.3). 

Please note that for simplicity, `environment.yml` only contains TensorFlow with _CPU_ support. Follow [this installation guide](https://www.tensorflow.org/install/ "tf Installation Guide") if you want to use a _GPU_-enabled version of TensorFlow.


## Running the Code
Execute `experiments_multi_gpu.py` to run our algorithm. Several options can be set at the beginning of this script. For instance it is possible to use multiple GPUs for training.

Then, the experimental setup can be defined: dataset, stepsize, neural network architecture, etc. One or multiple experiment configurations can be listed in the variable `schedule` and are executed consecutively. 

The performance measured during these experiments is saved in a log directory (specified in variable `log_dir`). The training progress can be monitored using _Tensorboard_. In a terminal, run `tensorboard --logdir=<path/to/log_dir>` and open the returned link in a browser.


## Plots
The plots in Figure 2 in the paper have been generated with the script `visualisation/plots.py`, the plots in Figure 3 with the script `visualisation/visualise_sampled_distr.py`. These plots can only be generated after the log files mentioned above have been generated.


## Acknowledgements
- Our code builds on the [SVAE implementation](https://github.com/mattjj/svae) by Johnson et. al. which is written in numpy and autograd. We have 'translated' parts of this code to Tensorflow.
- To allow for multi-GPU training, we used the _model replica approach_ explained [here](https://github.com/normanheckscher/mnist-multi-gpu/blob/master/README.md#training-a-model-using-multiple-gpu-cards "Model Replica Approach") and implemented [here](https://github.com/normanheckscher/mnist-multi-gpu/blob/master/mnist_multi_gpu_batching_train.py "Model Replica Approach") by Norman Heckscher. 
- We tried to make our plots look nicer using [this script](http://bkanuka.com/articles/native-latex-plots/) by Bennett Kanuka.
