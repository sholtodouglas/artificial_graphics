# -*- coding: utf-8 -*-
"""PyTorch on Cloud TPUs: MultiCore Training AlexNet on Fashion MNIST

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/multi-core-alexnet-fashion-mnist.ipynb

##PyTorch on Cloud TPUs: MultiCore Training AlexNet on Fashion MNIST 

This notebook will show you how to train [AlexNet](https://arxiv.org/abs/1404.5997) on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) using a Cloud TPU and all eight of its cores. It's a follow-up to [this notebook](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/single-core-alexnet-fashion-mnist.ipynb), which trains the same network on the same dataset the using a single Cloud TPU core. This will show you how to train your own networks on a Cloud TPU and highlight the difference between single and multicore training.

This notebook is part of a series of tutorials on using PyTorch on Cloud TPUs. PyTorch can use Cloud TPU cores as devices with the PyTorch/XLA package. For more on PyTorch/XLA see its [Github](https://github.com/pytorch/xla) or its [documentation](http://pytorch.org/xla/). We also have a ["Getting Started"](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb) Colab notebook. Additional Colab notebooks, like this one, are available on the PyTorch/XLA Github linked above.
"""

# import os
# assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'

# """### Installing PyTorch/XLA

# Run the following cell (or copy it into your own notebook!) to install PyTorch, Torchvision, and PyTorch/XLA. It will take a couple minutes to run.
# """

# # Installs PyTorch, PyTorch/XLA, and Torchvision
# # Copy this cell into your own notebooks to use PyTorch on Cloud TPUs 
# # Warning: this may take a couple minutes to run
# !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

"""Only run the below commented cell if you would like a nightly release"""

# VERSION = "20200325"  #@param ["1.5" , "20200325", "nightly"]
# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --version $VERSION

"""### Dataset & Network

In this notebook we'll train AlexNet on the Fashion MNIST dataset. Both are provided by the [Torchvision package](https://pytorch.org/docs/stable/torchvision/index.html).

The [previous notebook](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/single-core-alexnet-fashion-mnist.ipynb) trained this combination on a single TPU core and took time to visualize and describe the dataset. To avoid redundancy, please refer to it to learn more about the Fashion MNIST dataset.

### Using Multiple Cloud TPU Cores

Working with multiple Cloud TPU cores is different than training on a single Cloud TPU core. With a single Cloud TPU core we simply acquired the device and ran the operations using it directly. To use multiple Cloud TPU cores we must use other processes, one per Cloud TPU core. This indirection and multiplicity makes multicore training a little more complex than training on a single core, but it's necessary to maximize performance.

The following cell shows how to launch one process per Cloud TPU core.
"""

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# "Map function": acquires a corresponding Cloud TPU core, creates a tensor on it,
# and prints its core
def simple_map_fn(index, flags):
  # Sets a common random seed - both for initialization and ensuring graph is the same
  torch.manual_seed(1234)

  # Acquires the (unique) Cloud TPU core corresponding to this process's index
  device = xm.xla_device()  

  # Creates a tensor on this process's device
  t = torch.randn((2, 2), device=device)

  print("Process", index ,"is using", xm.xla_real_devices([str(device)])[0])

  # Barrier to prevent master from exiting before workers connect.
  xm.rendezvous('init')

# Spawns eight of the map functions, one for each of the eight cores on
# the Cloud TPU
flags = {}
# Note: Colab only supports start_method='fork'
xmp.spawn(simple_map_fn, args=(flags,), nprocs=8, start_method='fork')

"""Let's start looking at the above cell with the call to `spawn(),` which is documented [here](http://pytorch.org/xla/#torch_xla.distributed.xla_multiprocessing.spawn). `spawn()` takes a function (the "map function"), a tuple of arguments (the placeholder `flags` dict), the number of processes to create, and whether to create these new processes by "forking" or "spawning." While spawning new processes is generally recommended, Colab only supports forking.

`spawn()` will create eight processes, one for each Cloud TPU core, and call `simple_map_fn()` -- the map function -- on each process. The inputs to `simple_map_fn()` are an index (zero through seven) and the placeholder `flags.` When the proccesses acquire their device they actually acquire their corresponding Cloud TPU core automatically, as the above cell demonstrates.

### An Aside on Context

How did each process in the above cell know to acquire its own Cloud TPU core?

The answer is context. Accelerators, like Cloud TPUs, manage their operations using an implicit stateful context. In the cell above, the `spawn()` function creates a multiprocessing context and gives it to each new, forked process, allowing them to coordinate. We'll see another example of this coordination below.

Two warnings before we continue! First, you can't mix single process and multiprocess contexts when forking! Practically, this means that all our Cloud TPU-related calls will be done in processes created by spawn.
"""

# Don't mix these!
# Only one type of context per Colab!
# Warning: uncommenting the below and running this cell will cause a runtime error!

#device = xm.xla_device()  # Requires a single process context

#xmp.spawn(simple_map_fn, args=(flags,), nprocs=8, start_method='fork')  # Requires a multiprocess context

"""The second warning is: each process should perform the same Cloud TPU computations!"""

# Don't perform different computations on different processes!
# Warning: uncommenting the below and running this cell will likely hang your Colab!
# def simple_map_fn(index, flags):
#   torch.manual_seed(1234)
#   device = xm.xla_device()  

#   if xm.is_master_ordinal():
#     t = torch.randn((2, 2), device=device)  # Divergent Cloud TPU computation!


# xmp.spawn(simple_map_fn, args=(flags,), nprocs=8, start_method='fork')

"""Performing the same Cloud TPU computations lets the context coordinate the processes correctly. Again, we'll see this better below.

Note, however, you CAN perform different CPU computations in each process, as the next cell demonstrates.
"""

# Common Cloud TPU computation but different CPU computation is OK
def simple_map_fn(index, flags):
  torch.manual_seed(1234)
  device = xm.xla_device()  

  t = torch.randn((2, 2), device=device)  # Common Cloud TPU computation
  out = str(t)  # Each process uses the XLA tensors the same way

  if xm.is_master_ordinal():  # Divergent CPU-only computation (no XLA tensors beyond this point!)
    print(out)

  # Barrier to prevent master from exiting before workers connect.
  xm.rendezvous('init')


xmp.spawn(simple_map_fn, args=(flags,), nprocs=8, start_method='fork')

"""### Multicore Training

The following cell defines a map function that trains AlexNet on the Fashion MNIST dataset on all eight available Cloud TPU cores. The function is long and contained in a single cell, so it includes lengthy comments. It does the following:

- **Setup**: every process sets the same random seed and acquires the device assigned to it (via the accelerator context, see above)
- **Dataloading**: each process acquires its own copy of the dataset, but their sampling from it is coordinated to not overlap.
- **Network creation**: each process has its own copy of the network, but these copies are identical since each process's random seed is the same.
- **Training** and **Evaluation**: Training and evaluation occur as usual but use a ParallelLoader.

Aside from a couple different classes, like the DistributedSampler and the ParallelLoader, the big difference between single core and multicore training is behind the scenes. The `step()` function now not only propagates gradients, but uses the Cloud TPU context to synchronize gradient updates across each processes' copy of the network. This ensures that each processes' network copy stays "in sync" (they are all identical). 
"""

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch_xla.distributed.parallel_loader as pl
import time

def map_fn(index, flags):
  ## Setup 

  # Sets a common random seed - both for initialization and ensuring graph is the same
  torch.manual_seed(flags['seed'])

  # Acquires the (unique) Cloud TPU core corresponding to this process's index
  device = xm.xla_device()  


  ## Dataloader construction

  # Creates the transform for the raw Torchvision data
  # See https://pytorch.org/docs/stable/torchvision/models.html for normalization
  # Pre-trained TorchVision models expect RGB (3 x H x W) images
  # H and W should be >= 224
  # Loaded into [0, 1] and normalized as follows:
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
  to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
  resize = transforms.Resize((224, 224))
  my_transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

  # Downloads train and test datasets
  # Note: master goes first and downloads the dataset only once (xm.rendezvous)
  #   all the other workers wait for the master to be done downloading.

  if not xm.is_master_ordinal():
    xm.rendezvous('download_only_once')

  train_dataset = datasets.FashionMNIST(
    "/tmp/fashionmnist",
    train=True,
    download=True,
    transform=my_transform)

  test_dataset = datasets.FashionMNIST(
    "/tmp/fashionmnist",
    train=False,
    download=True,
    transform=my_transform)
  
  if xm.is_master_ordinal():
    xm.rendezvous('download_only_once')
  
  # Creates the (distributed) train sampler, which let this process only access
  # its portion of the training dataset.
  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=True)
  
  test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset,
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=False)
  
  # Creates dataloaders, which load data in batches
  # Note: test loader is not shuffled or sampled
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=flags['batch_size'],
      sampler=train_sampler,
      num_workers=flags['num_workers'],
      drop_last=True)

  test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=flags['batch_size'],
      sampler=test_sampler,
      shuffle=False,
      num_workers=flags['num_workers'],
      drop_last=True)
  

  ## Network, optimizer, and loss function creation

  # Creates AlexNet for 10 classes
  # Note: each process has its own identical copy of the model
  #  Even though each model is created independently, they're also
  #  created in the same way.
  net = torchvision.models.alexnet(num_classes=10).to(device).train()

  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters())


  ## Trains
  train_start = time.time()
  for epoch in range(flags['num_epochs']):
    para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    for batch_num, batch in enumerate(para_train_loader):
      data, targets = batch 

      # Acquires the network's best guesses at each class
      output = net(data)

      # Computes loss
      loss = loss_fn(output, targets)

      # Updates model
      optimizer.zero_grad()
      loss.backward()

      # Note: optimizer_step uses the implicit Cloud TPU context to
      #  coordinate and synchronize gradient updates across processes.
      #  This means that each process's network has the same weights after
      #  this is called.
      # Warning: this coordination requires the actions performed in each 
      #  process are the same. In more technical terms, the graph that
      #  PyTorch/XLA generates must be the same across processes. 
      xm.optimizer_step(optimizer)  # Note: barrier=True not needed when using ParallelLoader 

  elapsed_train_time = time.time() - train_start
  print("Process", index, "finished training. Train time was:", elapsed_train_time) 


  ## Evaluation
  # Sets net to eval and no grad context 
  net.eval()
  eval_start = time.time()
  with torch.no_grad():
    num_correct = 0
    total_guesses = 0

    para_train_loader = pl.ParallelLoader(test_loader, [device]).per_device_loader(device)
    for batch_num, batch in enumerate(para_train_loader):
      data, targets = batch

      # Acquires the network's best guesses at each class
      output = net(data)
      best_guesses = torch.argmax(output, 1)

      # Updates running statistics
      num_correct += torch.eq(targets, best_guesses).sum().item()
      total_guesses += flags['batch_size']
  
  elapsed_eval_time = time.time() - eval_start
  print("Process", index, "finished evaluation. Evaluation time was:", elapsed_eval_time)
  print("Process", index, "guessed", num_correct, "of", total_guesses, "correctly for", num_correct/total_guesses * 100, "% accuracy.")

# Configures training (and evaluation) parameters
flags['batch_size'] = 32
flags['num_workers'] = 8
flags['num_epochs'] = 1
flags['seed'] = 1234

xmp.spawn(map_fn, args=(flags,), nprocs=8, start_method='fork')

"""The network should take about 30 seconds to train and about 10 seconds to evaluate on each process. Using an entire Cloud TPU is, as expected, dramatically faster than training and evaluating on a single Cloud TPU core.

##What's Next?

This notebook broke down training AlexNet on the Fashion MNIST dataset using an entire Cloud TPU. A [previous notebook](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/single-core-alexnet-fashion-mnist.ipynb) showed how to train AlexNet on Fashion MNIST using only a single Cloud TPU core, and can be a helpful point of comparison. 

In particular, this notebook showed us how to:

- Define a "map function" that runs in parallel on one process per Cloud TPU core. 
- Run the map function using `spawn`.
- Understand the Cloud TPU context, its benefits, like automatic cross-process coordination, and its limits, like needing each process to perform the same Cloud TPU operations.
- Load and sample the datasets.
- Train and evaluate the network.

Additional notebooks demonstrating how to run PyTorch on Cloud TPUs can be found [here](https://github.com/pytorch/xla). While Colab provides a free Cloud TPU, training is even faster on [Google Cloud Platform](https://github.com/pytorch/xla#Cloud), especially when using multiple Cloud TPUs in a Cloud TPU pod. Scaling from a single Cloud TPU, like in this notebook, to many Cloud TPUs in a pod is easy, too. You use the same code as this notebook and just spawn more processes.
"""