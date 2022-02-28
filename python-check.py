#!/usr/bin/env python3

logger_name = 'lstnet-check'

import sys
sys.path.insert(0, '/home/ceph/LSTNet/')

import numpy as np
from predict import LSTNetPredictor as LSP

def read_matrix_from_file(path):
	mat = []
	with open(path, 'r') as fin:
		for line in fin.readlines():
			row = [ float(fig) for fig in line.strip().split() ]
			mat.append(row)
	return mat

def main():
	pred = LSP()
	load_matrix = read_matrix_from_file('data/input.txt')
	load_matrix = np.array(load_matrix).transpose()
	print(pred.predict(load_matrix))

if __name__ == '__main__':
	sys.exit(main())
