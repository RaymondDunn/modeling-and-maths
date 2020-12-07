import os
#from mlxtend.evaluate import permutation_test
from permute.core import two_sample, two_sample_conf_int
import numpy as np

#conda enviro
#pip install permute
# see http://statlab.github.io/permute/api/core.html

#correct if the population S.D. is expected to be equal for the two groups.
# calculate cohens d via https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def getDataLists(p1_file='p1.txt', p2_file='p2.txt'):

	# local file names
	p1_file = "p1.txt"
	p2_file = "p2.txt"
	cwd = os.getcwd()

	# local path
	p1_path = os.path.join(cwd, p1_file)
	p2_path = os.path.join(cwd, p2_file)

	# open files
	f1 = open(p1_path, 'r')
	f2 = open(p2_path, 'r')

	# save in list
	filelist = [f1, f2]
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

# get data from text files
datalist = getDataLists()

# local vars
y = datalist[0]
x = datalist[1]

# get test result
reps = 10000
test_result = two_sample(x, y, reps=reps, stat='mean', alternative='two-sided', keep_dist=True, seed=1)

# get confidence interval
conf_int = two_sample_conf_int(x, y, cl=0.95, alternative='two-sided', reps=reps, stat='mean')

# get cohen's d
d = cohen_d(x,y)

# print results
print('p-value: {}\ntest statistic: {}\nconfidence intervals: {}\ncohen\'s d: {}'.format(test_result[0], test_result[1], conf_int, d))















# do stats test
# pvalue = permutation_test(datalist[0], datalist[1], method='exact', num_rounds=100)

# print
#print('P-value is: {}'.format(pvalue))

###############################

#def run_permutation_test(pooled,sizeZ,sizeY,delta):
#	np.random.shuffle(pooled)
#	starZ = pooled[:sizeZ]
#	starY = pooled[-sizeY:]
#	return starZ.mean() - starY.mean()


# run it
#z = (datalist[0] * 1000).astype(np.int)
#y = (datalist[1] * 1000).astype(np.int)
#
#
#pooled = np.hstack([z,y])
#delta = z.mean() - y.mean()
#numSamples = 10000
#estimates = np.array(map(lambda x: run_permutation_test(pooled,z.size,y.size,delta),range(numSamples)))
#diffCount = len(np.where(estimates <= delta)[0])
#hat_asl_perm = 1.0 - (float(diffCount)/float(numSamples))
#hat_asl_perm




###################################

#def exact_mc_perm_test(xs, ys, nmc):
#    n, k = len(xs), 0
#    diff = np.abs(np.mean(xs) - np.mean(ys))
#    zs = np.concatenate([xs, ys])
#    for j in range(nmc):
#        np.random.shuffle(zs)
#        k += diff <= np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
#    return k / nmc
#
#pvalue = exact_mc_perm_test(datalist[0], datalist[1], 10000)