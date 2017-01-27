#!/usr/bin/env/python
"""
preprocess training data for cL
"""
import sys, os
from collections import defaultdict
from numpy.random import choice, shuffle
import numpy as np

def read_file(f_name, f_type):
	try:
		f = open(f_name)
	except:
		print >> sys.stderr, "file %s cannot be opened" % (f_name)
		quit(0)
	annotation = []
	ann = []
	for line in f:
		l = line.strip().split()
		if f_type == 'conll': # add conll
			if len(l) < 2:
				annotation += [ann]
				ann = []
			else:
				ann += ['\t'.join(l)]
		elif f_type == 'oracle': # add oracle
			ann += [" ".join(l)]
			if len(l) >= 1 and l[0] == '[ROOT-ROOT][]':
				annotation += [ann]
				ann = []
		else:
			raise NotImplementedError()
			quit(1)
	if ann != []:
		annotation + [ann]
	return annotation

class Annotation(object):
	def __init__(self, conll , oracle):
		self.conll = conll
		self.oracle= oracle
		self.length= len(conll) # length is for CL criterion others can be added here such as arc complexity

def debug(ann): # print an annotation instalce
	print "\n".join(ann.conll)
	print "_"*50
	print ann.length
	print "_"*50
	print "\n".join(ann.oracle)

def count_buckets(annotation, threshold = 1000, verbose = False):
	stats = defaultdict(int)
	for ann in annotation:           # change field if necessary
		stats[ann.length] += 1

	prev_cnt = 0
	prev_idx = -1
	totl_cnt = 0

	bucket_idx={}
	for l in stats:                  # distribute to 10 similar size buckets, 10 is adhoc
		totl_cnt += stats[l]
		if prev_cnt == 0:
			prev_idx += 1
		prev_cnt += stats[l]
		bucket_idx[l] = prev_idx
		if verbose:
			print l, prev_idx, stats[l], prev_cnt
		if prev_cnt >= threshold:
			prev_cnt = 0
	if verbose:
		print "total", totl_cnt, threshold
		print "_"*50
	return bucket_idx

def print_annotation(annotation, root_folder):
	try:
		f_conll = open(os.path.join(root_folder, 'trn.conll'), 'w')
		f_oracl = open(os.path.join(root_folder, 'trn.oracle'), 'w')
	except:
		print >> sys.stderr, "file %s cannot be opened" % (os.path.join(root_folder, 'trn.conll'))
		quit(0)
	for ann in annotation:
		f_conll.write("\n".join(ann.conll))
		f_conll.write("\n")
		f_conll.write("\n")
		f_oracl.write("\n".join(ann.oracle))
		f_oracl.write("\n")
	f_conll.close()
	f_oracl.close()

if __name__ == '__main__':
	file_conll = sys.argv[1] # raw trn.conll
	file_oracl = sys.argv[2] # raw trn.oracle
	data_root  = sys.argv[3] # new root directory to create

	l_conll = read_file(file_conll, 'conll')
	l_oracl = read_file(file_oracl, 'oracle')

	annotation = []
	for c,o in zip(l_conll, l_oracl):
		annotation += [Annotation(c,o)]

	N = len(annotation)
	if not os.path.exists(data_root):
		os.makedirs(data_root)


	# down sample from training data
	for fraction, threshold in zip(range(0,-6,-1),[3500,1800,900,450,225,110]): 
		n = int((2 ** fraction) * N)
		output_root = os.path.join(data_root, str(n))
		os.makedirs(os.path.join(output_root, 'random'))
		os.makedirs(os.path.join(output_root, 'sorted'))
		os.makedirs(os.path.join(output_root, 'onepass'))
		os.makedirs(os.path.join(output_root, 'babystep'))

		# no need for sampling if full data
		if fraction == 0:
			sampled_annotation = annotation
		else:
			sample_idx = choice(range(N), n , replace = False)
			sampled_annotation = list(np.array(annotation)[sample_idx])

		### 10 shuffled data
		for i in xrange(10):
			root = os.path.join(output_root, 'random', str(i))
			os.makedirs(root)
			shuffled_annotation = shuffle(sampled_annotation)
			print_annotation(sampled_annotation, root)

		### sorted data
		sampled_annotation.sort(key = lambda x: x.length)
		root = os.path.join(output_root, 'sorted')
		print_annotation(sampled_annotation, root)

		### 10 buckets for onepass
		buckets = count_buckets(sampled_annotation, threshold = threshold)
		bucket_annotation = defaultdict(list)
		for ann in sampled_annotation:
			bucket = buckets[ann.length]
			bucket_annotation[bucket] += [ann]
		for i in xrange(10):
			root = os.path.join(output_root, 'onepass', str(i))
			os.makedirs(root)
			print_annotation(bucket_annotation[i], root)

		### 10 concatenated for babystep
		bucket_annotation = defaultdict(list)
		for ii,ann in enumerate(sampled_annotation):
			bucket = buckets[ann.length]
			for j in xrange(bucket, 10):
				bucket_annotation[j] += [ann]

		for i in xrange(10):
			root = os.path.join(output_root, 'babystep', str(i))
			os.makedirs(root)
			print_annotation(bucket_annotation[i], root)
