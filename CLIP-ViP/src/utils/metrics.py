import numpy as np

def cal_cossim(feats1, feats2):
    sim_matrix = np.dot(feats1, feats2.T)
    return sim_matrix

def np_softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter, 
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    r1 = float(np.sum(ind == 0))  / len(ind)
    r5 = float(np.sum(ind < 5))  / len(ind)
    r10 = float(np.sum(ind < 10))  / len(ind)
    medr = np.median(ind) + 1
    meanr  = np.mean(ind) + 1
    return r1, r5, r10, medr, meanr

def compute_metrics_multi(x, t2v_labels_list):
    sx = np.sort(-x, axis=1)
    t2v_labels_list = np.array(t2v_labels_list)
    arg = np.arange(x.shape[0])
    d = -x[arg, t2v_labels_list]
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    r1 = float(np.sum(ind == 0))  / len(ind)
    r5 = float(np.sum(ind < 5))  / len(ind)
    r10 = float(np.sum(ind < 10))  / len(ind)
    medr = np.median(ind) + 1
    meanr  = np.mean(ind) + 1
    return r1, r5, r10, medr, meanr


if __name__ == '__main__':

    sim_matrix = np.random.random((5,5))



