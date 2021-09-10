fABBA
======================================

> An efficient aggregation based symbolic representation for temporal data


[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![!pypi](https://img.shields.io/pypi/v/fABBA?color=orange)](https://pypi.org/project/fABBA/)


fABBA is a fast and accurate symbolic representation methods, which allows for data compression and mining. 
By replacing the k-means clustering used in ABBA with a sorting-based aggregation technique, fABBA thereby
avoid repeated within-cluster-sum-of-squares computations, and the computational complexity is significantly reduced.
Also, in contrast to the ABBA, fABBA does not require the number of time series symbols to be specified in
advance while achieves competing performance against ABBA and other symbolic methods. 



## Install
To install the current release
```
pip install fABBA
```


#### *Apply series compression*

```python
>>> import numpy as np
>>> from fABBA.symbolic_representation import fabba_model
>>> np.random.seed(1)
>>> N = 100
>>> ts = [np.sin(i) for i in range(N)]
>>> fabba = fabba_model(tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=1, max_len=np.inf, string_form=True)
>>> print(fabba)
fABBA({'_alpha': 0.5, '_sorting': '2-norm', '_tol': 0.1, '_scl': 1, '_verbose': 1, '_max_len': inf, '_string_form': True, '_n_jobs': 1})

>>> string = fabba.fit_transform(ts)
>>> print(string)
(!"#"#"'$!%!$'"#"#&%!$!%&#"#"'$!%!$!"#"#&%!$!%&#"

>>> inverse_ts = fabba.inverse_transform(string, ts[0]) # reconstructed time series

```

Plot the image
```python
>>> plt.plot(ts, label='time series', c='olive')
>>> plt.plot(inverse_ts, label='reconstruction', c='darkblue')
>>> plt.legend()
>>> plt.grid(True, axis='y')
>>> plt.show()
```

![reconstruction](https://raw.githubusercontent.com/nla-group/fABBA/master/fig/demo.png?token=AKE3UMQFBJ7W4ML3N4LQ3KDBHOFE6)


#### *Apply adaptively polygonal chian approximation*

```python
>>> from fABBA.chainApproximation import compress
>>> from fABBA.chainApproximation import inverse_compress
>>> np.random.seed(1)
>>> N = 100
>>> ts = [np.sin(i) for i in range(N)]
>>> pieces = compress(ts, tol=0.1)
>>> inverse_ts = inverse_compress(pieces, ts[0])
```


#### *Apply aggregated digitization*

```python
>>> from fABBA.digitization import digitize
>>> from fABBA.digitization import inverse_digitize
>>> string, parameters = digitize(pieces, alpha=0.1, sorting='2-norm', scl=1) # pieces from aforementioned compression
>>> print(''.join(string))
(!"#"#"'$!%!$'"#"#&%!$!%&#"#"'$!%!$!"#"#&%!$!%&#"

>>> inverse_pieces = inverse_digitize(string, parameters)
>>> inverse_ts = inverse_compress(inverse_pieces, ts[0])
```


#### *Image compression*
```python
>>> import matplotlib.pyplot as plt
>>> from fABBA.load_datasets import load_images
>>> from fABBA.symbolic_representation import image_compress
>>> from fABBA.symbolic_representation import image_decompress
>>> from fABBA.symbolic_representation import fabba_model
>>> from cv2 import resize
>>> img_samples = load_images(shape=(100,100)) # load fABBA image test samples
>>> img = resize(img_samples[0], (100, 100)) # select the first image for test
>>> fabba = fabba_model(tol=0.1, alpha=0.01, sorting='2-norm', scl=1, verbose=1, max_len=np.inf, string_form=True)
>>> strings = image_compress(fabba, img)
>>> inverse_img = image_decompress(fabba, strings)
```

Plot the original image
```python
>>> plt.imshow(img)
>>> plt.show()
```

![original image](https://raw.githubusercontent.com/nla-group/fABBA/master/fig/img.png?token=AKE3UMVCUIERBFTZJDJKZCTBHOE6K)


Plot the reconstructed image
```python
>>> plt.imshow(inverse_img)
>>> plt.show()
```

![reconstruction](https://raw.githubusercontent.com/nla-group/fABBA/master/fig/inverse_img.png?token=AKE3UMS7KL6DK6CK4X5A2QLBHOE7E)

## Experiment

This repository named "experiments" contains all code required to reproduce the experiments in the manuscript 
"An efficient aggregation method for the symbolic representation of temporal data".

#### Overview and dependencies

The "experiments" folder is self-contained, covering all scripts to reproduce the experimental data except UCRArchive2018 in the paper.

The UCRArchive2018 datasets can be downloaded from [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

There are a number of dependencies listed below. Most of these modules, except perhaps the final ones, are part of any standard Python installation. We list them for completeness:

`os,  csv, time, pickle, numpy, warnings, matplotlib, math, collections, copy, sklearn, pandas, tqdm, tslearn`

Please ensure that these modules are available before running the codes.


## Software Contributor

Stefan Guettel <stefan.guettel@manchester.ac.uk>

Xinye Chen <xinye.chen@manchester.ac.uk>



## Reference


If you have used this software in a scientific publication and wish to cite it, 
please use the following citation.

```bibtex
@article{fABBAarticle,
  title={An efficient aggregation method for the symbolic representation of temporal data},
  author={Xinye, Chen and Guettel, Stefan},
  journal={},
  volume={},
  number={},
  pages={},
  year={2021}
}
```

