import numpy as np
#from scipy import interp
from scipy import interpolate
import scipy.stats as st

def get_confidence_interval(arrays, confidence=0.95, max=None):
    """
    Takes as input a list of arrays with the results for each seed.
    Returns the mean and confidence interval bounds for each array dimension.
    """
    array = np.vstack(arrays)
    mean = np.mean(array, axis=0)
    sem = st.sem(array, axis=0)
    #print(sem, st.t.ppf((1+confidence)/2, array.shape[0]-1))
    n = array.shape[0]
    h = np.array([s * st.t.ppf((1+confidence)/2., n-1) for s in sem])
    lower = mean - h
    upper = mean + h
    if max is not None:
        upper = np.minimum(upper, max)
    return mean, lower, upper


def ci_roc_curve(fprs, tprs, base_fpr):
    new_tprs = []
    for fpr, tpr in zip(fprs, tprs):
        func = interpolate.interp1d(fpr, tpr, kind='previous')
        new_tpr = func(base_fpr)
        new_tpr[0] = 0.0
        new_tprs.append(new_tpr)
    ci_tprs = get_confidence_interval(new_tprs, 0.95, max=1)
    return ci_tprs