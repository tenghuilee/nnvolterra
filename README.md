# nnvolterra

## Run Code

Compile first: ```make compile```

Run all codes: ```make all```


Test xconv: ```make npxconv_test```

MNIST dataset needs to be downloaded, converted to numpy format, and placed to `mnist/` folder.

Train MNIST network: ```make mnist_train```

Test MNIST network: ```make mnist_try```

Hack MNIST network: ```make mnist_hack```

## Required

- python
- numpy
- pytorch
- g++ 9.3.0
- xelatex [optional]: for mnist_draw.py


## About Files

- xconvlibrary
  - xconvolution.hpp: core lib
  - npxconv.cpp: cpp to python
  - npxconv.py: interface for python
  - npxconv_setup.py: setup
  - npxconv_test.py: test file
- outer convolution
  - oconv_draw.py: nnfragile and three example images of outer convolution
  - oconv_rank.py: compute rank for outer convolution and neural network to Volterra Convolution
  - oconv_rank_draw.py: draw images from `oconv_rank.py`
- mnist hack
  - mnist_module.py: the module
  - mnist_train.py: train this module in training set
  - mnist_try.py: check module in test set
  - mnist_hack.py: try to hack the module
  - mnist_draw.py: slightly draw the hack result
- other
  - tensordec.py: about tensor decomposition
  - shape_check.py: check shapes

## Please Cite

```
@misc{li2021understanding,
      title={Understanding Convolutional Neural Networks from Theoretical Perspective via Volterra Convolution}, 
      author={Tenghui Li and Guoxu Zhou and Yuning Qiu and Qibin Zhao},
      year={2021},
      eprint={2110.09902},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```