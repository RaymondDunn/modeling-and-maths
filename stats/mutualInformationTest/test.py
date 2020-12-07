import random
from sklearn import metrics
import numpy as np
import os
import pdb
from numba import jit

#############################
# bins = 2 ## MI bin number
SEED = 1 ## seed for random

#############################


# calc MI p values
def calc_MI_pvalue(xs, bins):

    # get mutual information of dataset
    mi_mat = calc_MI_matrix(xs, bins)

    # shuffle samples and calculate mutual information
    iterations = 10000
    shuffled_mi_mats = np.zeros((len(xs), len(xs), iterations))
    for i in range(iterations):
        
        # shuffle each vector
        xs_shuffled = []
        for x in xs:

            # shuffle temp var
            y = x.copy()
            random.shuffle(y)
            xs_shuffled.append(y)

        # calculate mi mat of shuffled
        mi_mat_shuffled = calc_MI_matrix(xs_shuffled, bins)
        shuffled_mi_mats[:,:,i] = mi_mat_shuffled

        # update user on progress
        if  i % 1000 == 0:
            print('Iteration {}...'.format(i))

    # calculate two-sided p value as proportion of values big enough away from sample
    p_mat = np.zeros(mi_mat.shape)
    for i in range(iterations):
        larger_arr = np.abs(mi_mat) > np.abs(shuffled_mi_mats[:,:,i])
        p_mat += larger_arr.astype(int)
    
    # return
    return p_mat / iterations

# calculate MI matrix
def calc_MI_matrix(xs, bins):
    
    score_matrix = np.zeros((len(xs), len(xs)))
    for i in range(len(xs)):
        for j in range(len(xs)):
            curr_score = calc_MI(xs[i], xs[j], bins)
            score_matrix[i, j] = curr_score

    return score_matrix

def calc_MI(x, y, bins):
    
    # calcuate the mutual info of a single segment to the 
    # John Webb 2018
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi


def getDataLists(p1_file='p1.txt', p2_file='p2.txt'):

    # local file names
    p_file_list = ["p1.txt", "p2.txt"]
    cwd = os.getcwd()

    # local path
    p_path = []
    for p in p_file_list:
        p_path.append(os.path.join(cwd, p))

    # open files
    filelist = []
    for pp in p_path:
        f = open(pp, 'r')
        filelist.append(f)

    # save in list
    datalist = []

    # iterate files
    for f in filelist:

        # holder for file data
        flist = []
        fcontent = f.readlines()
        
        for line in fcontent:

            # append formatted list
            flist.append(float(line.replace('\n', "")))

        # push file data as array
        datalist.append(np.array(flist))

    # return
    return datalist

# seed rng
random.seed(SEED)

# get data from text files
datalist = getDataLists()

# local vars
xs = []
for x in datalist:
    xs.append(list(x))

# iterate through bin sizes to how p-values depend on bin sizes
bins_sizes = range(2, 20, 2)
p_flat = []
for bins in bins_sizes:

    # get pvalues from random shuffling
    print('Bin size: {}'.format(bins))
    p_mat = calc_MI_pvalue(xs, bins)
    p_flat.append(p_mat.flatten())

# restructure into matrix
mean_p = np.array(p_flat).mean(axis=0).reshape((len(xs),len(xs)))