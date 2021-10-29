import numpy
import pandas
def lowVarianceFilter(X_train,X_test,desiredFeature):
	result_X_train=numpy.zeros((X_train.shape[0], 0))
	result_X_test =numpy.zeros((X_test.shape[0], 0))
	# fill index
	vars=numpy.var(X_train,axis=0)
	# append index
	VARS=[]
	for i in range(len(vars)):
		VARS.append((vars[i],i))
	VARS.sort()
	for _, index in VARS[len(VARS):len(VARS)-desiredFeature-1:-1]:
		result_X_train=numpy.append(result_X_train, X_train[:,index].reshape(X_train.shape[0], 1), axis=1)
		result_X_test =numpy.append(result_X_test , X_test [:,index].reshape(X_test.shape[0], 1), axis=1)
	return result_X_train, result_X_test
def anovaFValue(X_train, X_test, desiredFeature, Y_train):
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import f_classif
	fvalue_selector = SelectKBest(f_classif, k=desiredFeature)
	X_train=fvalue_selector.fit_transform(X_train, Y_train)
	X_test=fvalue_selector.transform(X_test)
	return X_train, X_test
def perceptronCheat(X_train,X_test,sqrSize,Y_train, log_file):
	import sklearn
	from sklearn.linear_model import Perceptron
	from sklearn.neural_network import MLPClassifier
	clf = Perceptron(tol=0, max_iter=2000)
	clf.fit (X_train, Y_train)
	coef = clf.coef_[0]
	tmp = []
	i = 0
	for a in coef: 
		tmp.append ((a, i))
		i+=1
	tmp.sort ()
	totalSize = sqrSize
	head = totalSize // 2
	tail = totalSize - head
	selected = [index for (value, index) in tmp[:head]]
	selected.extend ([index for (value, index) in tmp[len(tmp)-1:len(tmp)-tail-1:-1]])

	result_X_train=numpy.zeros((X_train.shape[0], 0))
	result_X_test =numpy.zeros((X_test.shape[0], 0))
	result_X_train=numpy.append(result_X_train, X_train[:,selected].reshape(X_train.shape[0], len(selected)), axis=1)
	result_X_test =numpy.append(result_X_test , X_test [:,selected].reshape(X_test.shape[0],  len(selected)), axis=1)

	print('Writing indexes to file ...',end='')
	f=open(log_file, 'a')
	f.write(str(selected)+"\n")
	f.close()
	print('Done writing')
	
	return result_X_train, result_X_test	
def ridgingLowVarianceFilter(X_train, X_test, desiredFeature, Y_train, Y_test):
	def rigde(X_train, X_test, desiredFeature, Y_train, Y_test):
		from sklearn.feature_selection import SelectFromModel
		from sklearn.linear_model import LogisticRegression
		transformer = SelectFromModel(estimator=LogisticRegression(C=1, penalty='l2'), threshold=-numpy.inf, max_features=desiredFeature).fit(X_train, Y_train)
		X_train = transformer.transform(X_train)
		X_test  = transformer.transform(X_test)
		return X_train, X_test
	# Next stage try Fast Correlation Based Filter
	stage1_X_train, stage1_X_test = rigde(X_train, X_test, 4096, Y_train, Y_test)
	stage2_X_train, stage2_X_test = lowVarianceFilter(stage1_X_train,stage1_X_test, desiredFeature) # desiredFeature is as small as possible
	return stage2_X_train, stage2_X_test
'''
	Following scope is adapted from fcbf implementation of Prashant Shiralkar by Vinh Phuc Ta Dang
'''
"""
fcbf.py
Created by Prashant Shiralkar on 2015-02-06.
Fast Correlation-Based Filter (FCBF) algorithm as described in 
Feature Selection for High-Dimensional Data: A Fast Correlation-Based
Filter Solution. Yu & Liu (ICML 2003)
"""
import sys
import os
import argparse
import numpy as np

def entropy(vec, base=2):
	" Returns the empirical entropy H(X) in the input vector."
	_, vec = np.unique(vec, return_counts=True)
	prob_vec = np.array(vec/float(sum(vec)))
	if base == 2:
		logfn = np.log2
	elif base == 10:
		logfn = np.log10
	else:
		logfn = np.log
	return prob_vec.dot(-logfn(prob_vec))

def conditional_entropy(x, y):
	"Returns H(X|Y)."
	uy, uyc = np.unique(y, return_counts=True)
	prob_uyc = uyc/float(sum(uyc))
	cond_entropy_x = np.array([entropy(x[y == v]) for v in uy])
	return prob_uyc.dot(cond_entropy_x)
	
def mutual_information(x, y):
	" Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y."
	return entropy(x) - conditional_entropy(x, y)

def symmetrical_uncertainty(x, y):
	" Returns 'symmetrical uncertainty' (SU) - a symmetric mutual information measure."
	return 2.0*mutual_information(x, y)/(entropy(x) + entropy(y))

def getFirstElement(d):
	"""
	Returns tuple corresponding to first 'unconsidered' feature
	
	Parameters:
	----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	
	Returns:
	-------
	a, b, c : tuple
		a - SU value, b - original feature index, c - index of next 'unconsidered' feature
	"""
	
	t = np.where(d[:,2]>0)[0]
	if len(t):
		return d[t[0],0], d[t[0],1], t[0]
	return None, None, None

def getNextElement(d, idx):
	"""
	Returns tuple corresponding to the next 'unconsidered' feature.
	
	Parameters:
	-----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	idx : int
		Represents original index of a feature whose next element is required.
		
	Returns:
	--------
	a, b, c : tuple
		a - SU value, b - original feature index, c - index of next 'unconsidered' feature
	"""
	t = np.where(d[:,2]>0)[0]
	t = t[t > idx]
	if len(t):
		return d[t[0],0], d[t[0],1], t[0]
	return None, None, None
	
def removeElement(d, idx):
	"""
	Returns data with requested feature removed.
	
	Parameters:
	-----------
	d : ndarray
		A 2-d array with SU, original feature index and flag as columns.
	idx : int
		Represents original index of a feature which needs to be removed.
		
	Returns:
	--------
	d : ndarray
		Same as input, except with specific feature removed.
	"""
	d[idx,2] = 0
	return d

def c_correlation(X, y):
	"""
	Returns SU values between each feature and class.
	
	Parameters:
	-----------
	X : 2-D ndarray
		Feature matrix.
	y : ndarray
		Class label vector
		
	Returns:
	--------
	su : ndarray
		Symmetric Uncertainty (SU) values for each feature.
	"""
	su = np.zeros(X.shape[1])
	for i in np.arange(X.shape[1]):
		su[i] = symmetrical_uncertainty(X[:,i], y)
	return su

def fcbf(X, y, thresh, X_train, X_test):
	"""
	Perform Fast Correlation-Based Filter solution (FCBF).
	
	Parameters:
	-----------
	X : 2-D ndarray
		Feature matrix
	y : ndarray
		Class label vector
	thresh : float
		A value in [0,1) used as threshold for selecting 'relevant' features. 
		A negative value suggest the use of minimum SU[i,c] value as threshold.
	
	Returns:
	--------
	sbest : 2-D ndarray
		An array containing SU[i,c] values and feature index i.
	"""
	n = X.shape[1]
	slist = np.zeros((n, 3))
	slist[:, -1] = 1

	# identify relevant features
	slist[:,0] = c_correlation(X, y) # compute 'C-correlation'
	idx = slist[:,0].argsort()[::-1]
	slist = slist[idx, ]
	slist[:,1] = idx
	if thresh < 0:
		thresh = np.median(slist[-1,0])
		print ("Using minimum SU value as default threshold: {0}".format(thresh))
	elif thresh >= 1 or thresh > max(slist[:,0]):
		print ("No relevant features selected for given threshold.")
		print ("Please lower the threshold and try again.")
		exit()
		
	slist = slist[slist[:,0]>thresh,:] # desc. ordered per SU[i,c]
	
	# identify redundant features among the relevant ones
	cache = {}
	m = len(slist)
	p_su, p, p_idx = getFirstElement(slist)
	for i in range(m):
		p = int(p)
		q_su, q, q_idx = getNextElement(slist, p_idx)
		if q:
			while q:
				q = int(q)
				if (p, q) in cache:
					pq_su = cache[(p,q)]
				else:
					pq_su = symmetrical_uncertainty(X[:,p], X[:,q])
					cache[(p,q)] = pq_su

				if pq_su >= q_su:
					slist = removeElement(slist, q_idx)
				q_su, q, q_idx = getNextElement(slist, q_idx)
				
		p_su, p, p_idx = getNextElement(slist, p_idx)
		if not p_idx:
			break
	sbest = slist[slist[:,2]>0, :2]
	selected = [int(_[1]) for _ in sbest]
	return X_train[:,selected], X_test[:, selected]
'''
	End of adapting scope
'''
def madFilter(X_train, X_test, firstStageFeature, Y_train, Y_test):
	X_train_stage1, X_test_stage1 = lowVarianceFilter(X_train, X_test, firstStageFeature)
	# binning afterward
	from sklearn.preprocessing import KBinsDiscretizer
	KBin = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform') # Quantile
	KBin.fit(X_train_stage1)
	X_train_stage2 = KBin.transform(X_train_stage1)
	X_test_stage2 = KBin.transform(X_test_stage1)
	# run feature selection
	X_train_stage3, X_test_stage3 = fcbf(X_train_stage2, Y_train, 0.001, X_train, X_test)
	return X_train_stage3, X_test_stage3
if __name__ == "__main__":
	pass