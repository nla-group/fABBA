---
title: '*fABBA*: A Python library for the fast symbolic approximation~of time series'
tags:
  - Python
  - time series
  - dimensionality reduction
  - symbolic representation
  - data science
authors:
  - name: Xinye Chen
    orcid: 0000-0003-1778-393X
    affiliation: 1
  - name: Stefan Güttel
    orcid: 0000-0003-1494-4478
    affiliation: 2
    
affiliations:
  - name: Department of Numerical Mathematics, Charles University Prague, Czech Republic
    index: 1
  - name: Department of Mathematics, The University of Manchester, United Kingdom
    index: 2
    
date: 06 December 2023
bibliography: paper.bib
---



# Summary

Adaptive Brownian bridge-based aggregation (ABBA) [@EG19b] is a symbolic time series representation approach that is applicable to general time series. It is based on a tolerance-controlled polygonal chain approximation of the time series, followed by a mean-based clustering of the polygonal pieces into groups.  With the increasing need for faster time series processing, lots of efforts have been put into deriving new time series representations in order to reduce the time complexity of similarity search or enhance forecasting performance of machine learning models. Compared to working on the raw time series data, symbolizing time series with ABBA provides numerous benefits including but not limited to (1) dimensionality reduction, (2) smoothing and noise reduction, and (3) explainable feature discretization. The time series features extracted by ABBA enable fast time series forecasting [@EG20b], anomaly detection [@EG19b; @CG22a], event prediction [@9935005], classification [@TAKTAK2024102294; @10.1007/978-3-031-24378-3_4], and other data-driven tasks in time series analysis [@WANG2023109123; @10.1145/3448672]. An example illustration of an ABBA symbolization is shown in \autoref{fig:enter-label}.


![ABBA symbolization with 4 symbols.\label{fig:enter-label}](abba.png)

ABBA follows a two-phase approach to symbolize time series, namely compression and digitization. The first phase aims to reduce the time series dimension by polygonal chain approximation, and the second phase assigns symbols to the polygonal pieces. Both phases operate together to ensure that the essential time series features are best reflected by the symbols, controlled by a user-chosen error tolerance. The advantages of the ABBA representation against other symbolic representations include (1) better preservation of essential shape features, e.g., when compared against the popular SAX representation~[@SAX03; @EG19b]; (2) effective representation of local up and down trends in the time series which supports motif detection; (3) demonstrably reduced sensitivity to hyperparameters of neural network models and the initialization of random weights in forecasting applications~[@EG20b].  


*fABBA* is a Python library to compute ABBA symbolic time series representations on Linux, Windows, and MacOS systems. With Cython compilation and typed memoryviews, it significantly outperforms existing ABBA implementations. The *fABBA* library also includes a new ABBA variant, fABBA [@CG22a], which uses an alternative digitization method (``greedy aggregation'') instead of k-means clustering~[@1056489], providing significant speedup and improved tolerance-based digitization (without the need to specify the number $k$ of symbols a priori). The experiments in [@CG22a] demonstrate that fABBA compares favorably to the original ABBA module\footnote{https://github.com/nla-group/ABBA} in terms of runtime. *fABBA* is an open-source library and licensed under the 3-Clause BSD License. Its redistribution and use, with or without modification, are permitted under conditions described in \url{https://opensource.org/license/bsd-3-clause/}.

# Examples
*fABBA*  can installed via the Python Package Index or conda forge. Detailed documentation for its installation, usage, API reference, and quick start examples can be found on~\url{https://fabba.readthedocs.io/en/latest/}. Below we provide a brief demonstration. 



## Compress and reconstruct a time series
The following example approximately transforms a time series into a symbolic string representation (`transform()`) and then converts the string back into a numerical format (`inverse_transform()`). fABBA requires two parameters, `tol` and `alpha`. The tolerance `tol` determines how closely the polygonal chain approximation follows the original time series. The parameter `alpha` controls how similar time series pieces need to be in order to be represented by the same symbol. A smaller `tol` means that more polygonal pieces are used and the polygonal chain approximation is more accurate; but on the other hand, it will increase the length of the string representation. Similarly, a smaller `alpha` typically results in more accurate symbolic digitization but a larger number of symbols.

```python
import numpy as np
import matplotlib.pyplot as plt
from fABBA import fABBA

ts = [np.sin(0.05*i) for i in range(1000)]  # original time series
fabba = fABBA(tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=0) 

string = fabba.fit_transform(ts)            # symbolic representation of the time series
print(string)                               # prints aBbCbCbCbCbCbCbCA

inverse_ts = fabba.inverse_transform(string, ts[0]) # reconstruct numerical time series
```


## More ABBA variants}
Other clustering-based ABBA variants are also provided, supported by the clustering methods in the \textit{scikit-learn} library [@scikit-learn]. Below is a basic code example.

```python
import numpy as np
from sklearn.cluster import KMeans
from fABBA import ABBAbase

ts = [np.sin(0.05*i) for i in range(1000)]         # original time series
kmeans = KMeans(n_clusters=5, random_state=0, init='k-means++', verbose=0) # k-means clustering with 5 symbols
abba = ABBAbase(tol=0.1, scl=1, clustering=kmeans)

string = abba.fit_transform(ts)                    # symbolic representation of the time series
print(string)                                      # prints BbAaAaAaAaAaAaAaC

inverse_ts = abba.inverse_transform(string)        # reconstruct numerical time series
```


# Statement of Need
Symbolic representations enhance time series processing by a large number of powerful techniques developed, e.g., by the natural language processing or bioinformatics communities~[@SAX03; @lin2007experiencing]. *fABBA* is a Python module for computing such symbolic time series representations very efficiently, enabling their use for downstream tasks such as time series classification, forecasting, and anomaly detection. 

# Acknowledgement
Stefan Güttel acknowledges a Royal Society Industry Fellowship IF/R1/231032.
