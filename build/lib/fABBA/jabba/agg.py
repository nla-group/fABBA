import numpy as np
from scipy.sparse.linalg import svds


def aggregate(data, sorting="norm", tol=0.5):
    """aggregate the data

    Parameters
    ----------
    data : numpy.ndarray
        the input that is array-like of shape (n_samples,).

    sorting : str
        the sorting method for aggregation, default='norm', alternative option: 'pca'.

    tol : float
        the tolerance to control the aggregation. if the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group.  

    Returns
    -------
    labels (numpy.ndarray) : 
        the group categories of the data after aggregation
    
    splist (list) : 
        the list of the starting points
    
    nr_dist (int) :
        number of pairwise distance calculations
    """

    splist = list() # store the starting points
    len_ind = data.shape[0]

    if sorting == "norm": 
        cdata = data
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
        ind = np.argsort(sort_vals)

    elif sorting == "pca":
        # pca = PCA(n_components=1) 
        # sort_vals = pca.fit_transform(data_memview).reshape(-1)
        # ind = np.argsort(sort_vals)
        
        # change to svd 
        # cdata = data - data.mean(axis=0) -- already done in the clustering.fit_transform
        cdata = data - data.mean(axis=0)
        if data.shape[1]>1:
            U1, s1, _ = svds(cdata, k=1, return_singular_vectors="u")
            sort_vals = U1[:,0]*s1[0]
            # print( U1, s1, _)
        else:
            sort_vals = cdata[:,0]
        sort_vals = sort_vals*np.sign(-sort_vals[0]) # flip to enforce deterministic output
        ind = np.argsort(sort_vals)

    else: # no sorting
        sort_vals = np.zeros(len_ind)
        ind = np.arange(len_ind)
        
    lab = 0
    labels = [-1]*len_ind
    # nr_dist = 0 
    
    for i in range(len_ind): # tqdm（range(len_ind), disable=not verbose)
        sp = ind[i] # starting point
        if labels[sp] >= 0:
            continue
        else:
            clustc = cdata[sp,:] 
            labels[sp] = lab
            num_group = 1

        for j in ind[i:]:
            if labels[j] >= 0:
                continue

            # sort_val_c = sort_vals[sp]
            # sort_val_j = sort_vals[j]
            
            if (sort_vals[j] - sort_vals[sp] > tol):
                break       

            # dist = np.sum((clustc - cdata[j,:])**2)    # slow

            dat = clustc - cdata[j,:]
            dist = np.inner(dat, dat)
            # nr_dist += 1
                
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab

        splist.append([sp, lab] + [num_group] + list(data[sp,:]) ) # respectively store starting point
                                                               # index, label, number of neighbor objects, center (starting point).
        lab += 1

    # if verbose == 1:
    #    print("aggregate {} groups".format(len(np.unique(labels))))

    return np.array(labels), splist




def aggregate_1d(data, tol=0.5):
    """aggregate the data

    Parameters
    ----------
    data : numpy.ndarray
        the input that is array-like of shape (n_samples,).

    sorting : str
        the sorting method for aggregation, default='norm', alternative option: 'pca'.

    tol : float
        the tolerance to control the aggregation. if the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group.  

    Returns
    -------
    labels (numpy.ndarray) : 
        the group categories of the data after aggregation
    
    splist (list) : 
        the list of the starting points
    
    *nr_dist (int) :
    *    number of pairwise distance calculations
    """

    splist = list() # store the starting points
    sort_vals = np.squeeze(data)
    len_ind = len(sort_vals)
    ind = np.argsort(sort_vals) # order by increasing size
    
    lab = 0
    labels = [-1]*len_ind
    nr_dist = 0 
    
    for i in range(len_ind): # tqdm（range(len_ind), disable=not verbose)
        sp = ind[i] # starting point
        if labels[sp] >= 0:
            continue
        else:
            # clustc = data[sp]
            if data[sp] < sort_vals[-1] - tol:
                clustc = data[sp] + tol
            else:
                clustc = data[sp]
            labels[sp] = lab
            num_group = 1

        for j in ind[i:]:
            if labels[j] >= 0:
                continue
            
            if (np.abs(sort_vals[j] - clustc) > tol):
                break       
                
            dat = clustc - data[j]
            dist = np.inner(dat, dat)
            nr_dist += 1
                
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab

        splist.append([sp, lab] + [num_group] + [clustc]) 
        # respectively store starting point
        # index, label, number of neighbor objects, center (starting point).
        lab += 1
    return np.array(labels), splist
