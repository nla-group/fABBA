# fABBA

### An efficient aggregation method for the symbolicrepresentation of temporal data

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![!pypi](https://img.shields.io/pypi/v/fABBA?color=orange)](https://pypi.org/project/fABBA/)

fABBA is a fast and accurate symbolic representation method for temporal data. 
It is based on a polygonal chain approximation of the time series followed by an aggregation of the polygonal pieces into groups. 
The aggregation process is sped up by sorting the polygonal pieces and exploiting early termination conditions. 
In contrast to the ABBA method [S. Elsworth and S. Güttel, Data Mining and Knowledge Discovery, 34:1175-1200, 2020], fABBA avoids repeated within-cluster-sum-of-squares computations which reduces its computational complexity significantly.
Furthermore, fABBA is fully tolerance-driven and does not require the number of time series symbols to be specified by the user. 

## Install
To install the current release
```
pip install fABBA
```

## Examples 

#### *Compress and reconstruct a time series*

The following example approximately transforms a time series into a symbolic string representation (`transform`) and then converts the string back into a numerical format (`inverse_transform`). In this example the time series is a sine wave and its symbolic representation is `#$!"!"!"!"!"!"!"%`. Note how the periodicity in the time series is reflected in repetitions in its string representation.

```python
import numpy as np
import matplotlib.pyplot as plt
from fABBA.symbolic_representation import fabba_model

ts = [np.sin(0.05*i) for i in range(1000)]          # original time series
fabba = fabba_model(tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=0, max_len=np.inf, string_form=True)

string = fabba.fit_transform(ts)                    # string representation of the time series
print(string)                                       # prints #$!"!"!"!"!"!"!"%

inverse_ts = fabba.inverse_transform(string, ts[0]) # numerical time series reconstruction
```

Plot the time series and its polygonal chain reconstruction:
```python
plt.plot(ts, label='time series', c='olive')
plt.plot(inverse_ts, label='reconstruction', c='darkblue')
plt.legend()
plt.grid(True, axis='y')
plt.show()
```

![reconstruction](https://raw.githubusercontent.com/umtsd/C_temp_img/main/fABBAdemo/demo.png)


#### *Adaptive polygonal chain approximation*

Instead of using `transform` which combines the polygonal chain approximation of the time series and the symbolic conversion into one, both steps of fABBA can be performed independently. This is the first step.

```python
from fABBA.chainApproximation import compress
from fABBA.chainApproximation import inverse_compress
ts = [np.sin(0.05*i) for i in range(1000)]
pieces = compress(ts, tol=0.1)                      # pieces is a list of the polygonal chain pieces
inverse_ts = inverse_compress(pieces, ts[0])        # reconstruct polygonal chain from pieces
```

And this is the second.

```python
from fABBA.digitization import digitize
from fABBA.digitization import inverse_digitize
string, parameters = digitize(pieces, alpha=0.1, sorting='2-norm', scl=1) # compression of the polygon into few symbols
print(''.join(string))                              # string representation 

inverse_pieces = inverse_digitize(string, parameters)
inverse_ts = inverse_compress(inverse_pieces, ts[0])
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

![original image](https://github.com/umtsd/C_temp_img/raw/main/fABBAdemo/img.png)


Plot the reconstructed image
```python
>>> plt.imshow(inverse_img)
>>> plt.show()
```


![reconstruction](https://github.com/umtsd/C_temp_img/raw/main/fABBAdemo/inverse_img.png)

## Experiment

The folder named "experiments" contains all code required to reproduce the experiments in the manuscript 
"An efficient aggregation method for the symbolic representation of temporal data".

#### Overview and dependencies

The "experiments" folder is self-contained, covering all scripts to reproduce the experimental data except UCRArchive2018 datasets in the paper.

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

