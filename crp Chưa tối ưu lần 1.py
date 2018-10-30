import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import Counter
import json
import os, sys
import math
import myCrpFunctions


def filter_sort(arr):
		x = []
		if (len(arr) > 0):
			x.append(arr[0])
			for t in arr:
				if x.count(t) == 0:
					x.append(t)
		return x

def state_phase(v, start, dim, tau):
	#v: vecto
	#dim là số chiều (số phần tử)	
	#tau là bước nhảy		
	return [v[start + i*tau] for i in range(0, dim, 1)]

def SyncPredictToTestArr(indexSet, dataSet):
	# Lấy các giá trị của dataSet tại các index liên tiếp trong indexSet
	outputArr = [[0]]
	i = 0
	while (i < len(indexSet)):
		a = [dataSet[indexSet[i]]];
		while (i < len(indexSet) -1 and (indexSet[i+1] - indexSet[i] == 1)):
			i+=1
			a.append(dataSet[indexSet[i]])
		if (len(a) > 1):
			outputArr.append(a)
		i+=1		
	return outputArr


def predict_diagonal(trainSet, testSet, dim=5, tau=2, epsilon=0.7, lambd=3, percent=0.6, titleOfGraph = 'Pretty Girl'):

	# # vectors_train = data['train_data']
	# dim = 5 #data['dim']
	# tau = 2 #data['tau']
	# percent = 0.6 #data['percent']
	# # curve_number = data['curve_number']
	# epsilon = 0.7 #data['epsilon']
	# lambd = 2 #data['lambd']

	vectors_train_1 = []
	for i in range(len(trainSet)-(dim-1)*tau):
		vectors_train_1.append(state_phase(trainSet, i, dim, tau)) 

	#tách statephases
	vectors_test_1 = []
	for i in range(len(testSet)-(dim-1)*tau):
		vectors_test_1.append(state_phase(testSet, i, dim, tau))

		#ép kiểu về array
	vectors_train_1 = np.array(vectors_train_1)
	vectors_test_1 = np.array(vectors_test_1)

	print("train.shape: ", vectors_train_1.shape)

	# r_dist là ma trận khoảng cách
	# cdist là hàm trong numpy
	# minkowski là cách tính
	# p là norm
	# y là train: đánh số từ trên xuống dưới
	# x là test
	r_dist = cdist(vectors_train_1, vectors_test_1, 'minkowski', p=1)
	
	# print("r_dist:",r_dist)

	print('vectors_train.shape: ', vectors_train_1.shape) #in ra shape
	print('vectors_test.shape: ', vectors_test_1.shape)
	print('r_dist.shape: ', r_dist.shape)
	print('r_dist min', r_dist.min())
	print('epsilon: ', epsilon)
	print('r_dist max', r_dist.max())
	print('lambd: ', lambd)

	# predict_label = np.zeros((testing_well_ts[:, -1].shape[0], ), dtype=int)

	# indx_vectors_timeseries_test = indx_vectors_timeseries_test.reshape((vectors_test.shape[0], dim))
	predict_label = np.zeros(len(testSet),dtype=int)
	
	#r1 là ma trận 01
	r1 = np.array(r_dist < epsilon)
	# r1 = np.array(r1 - 2)
	
	print(r1)
	f_crp = myCrpFunctions.crossRecurrencePlots("CRP", r1, dotSize = 0.2)

	len_r1 = int(np.size(r1[0]))
	high_r1 = int(np.size(r1)/len_r1)

	#x is an array whose each element is a diagonal of the crp matrix
	diagonalsMatrix = []

	for offset in range(-high_r1+1, len_r1):
		diagonalsMatrix.append(np.array(np.diagonal(r1, offset), dtype=int))
		print(diagonalsMatrix[len(diagonalsMatrix)-1])
	# print(x)
	
	f_crp = myCrpFunctions.crossRecurrencePlots("CRP2", diagonalsMatrix, dotSize = 0.2)


	index_vecto = []
	#index_vecto là index của các statePhase trong ma trận r_dist
	#đoạn dưới là lấy vị trí statePhase test cho index_vecto từ ma trận đường chéo
	for i_row in range(len(diagonalsMatrix)):
		lenn = np.size(diagonalsMatrix[i_row])
		i = 0
		while i<lenn:
			if (diagonalsMatrix[i_row][i] == 1):
				start = i
				while (i<lenn-1 and diagonalsMatrix[i_row][i+1] == 1 ):
					i+=1
				if (i-start+1 > lambd):
					for temp in range(i-start+1):
						if (i_row < high_r1):
							index_vecto.append(start+temp)
						else:
							index_vecto.append(i_row - high_r1 + start + 1 + temp)			
			i+=1

	print("------------------------------------------\n",
			index_vecto,
			"\n------------------------------------------")
	#lọc y
	index_vecto.sort()
	index_vecto = filter_sort(index_vecto)

	#index của cái feature ban đầu
	index_timeseries_test = []
	#đoạn này chuyển lại statephase thành feature
	for i in index_vecto:
		for j in range(((int(dim*percent))-1)*tau + 1):
			index_timeseries_test.append(i+j)
	#lọc
	index_timeseries_test.sort()
	index_timeseries_test = filter_sort(index_timeseries_test)

	#khởi tạo label
	for i in index_timeseries_test:
		predict_label[i] = 1
	print('predict_label:', predict_label)
	print('num predict: ', np.sum(predict_label))

	testSetPr = SyncPredictToTestArr(index_timeseries_test, testSet)

	f_line = plt.figure("line")
	for i in range(1, np.size(testSetPr)):
		plt.plot(testSetPr[i], ':', label=i)
	plt.plot(trainSet, 'r', label='train')
	plt.legend(loc=0)
	plt.xlabel('index')
	plt.ylabel('feature')
	plt.title(titleOfGraph)
	# plt.show()

	return predict_label.ravel().tolist()

###############################-----________MAIN________-----###############################

if (__name__ == "__main__"):
	dataTrain = myCrpFunctions.readCSVFile("data/15_1-SD-1X_LQC.csv")
	dataTest = myCrpFunctions.readCSVFile("data/15_1-SD-2X-DEV_LQC.csv")

	if (min(dataTrain) > min(dataTest)) :
		minOfNorm = min(dataTest) 
	else: 
		minOfNorm = min(dataTrain) 

	if max(dataTrain) < max(dataTest) : 
		maxOfNorm = max(dataTest) 
	else: 
		maxOfNorm = max(dataTrain) 
	
	trainSet = myCrpFunctions.ConvertSetNumber(dataTrain, minOfSet = minOfNorm, maxOfSet = maxOfNorm)
	testSet = myCrpFunctions.ConvertSetNumber(dataTest, minOfSet = minOfNorm, maxOfSet = maxOfNorm)

	start = 608
	finish = start+16
	s = str(start) + " - " + str(finish)
	print(s)
	print(trainSet[start:finish])
	# predict_diagonal(testSet[800:1100], testSet[800:1100] ,
	# 					dim=5, tau=2, epsilon=0.2, lambd=5, percent=1, titleOfGraph = s)
						
	predict_diagonal(testSet[800:1100], testSet[800:1100] ,
							dim=5, tau=2, epsilon=0.2, lambd=5, percent=1, titleOfGraph = s)

	
	plt.show()

	# for start in range(0, 15237, 8):
		
	# 	finish = start+16
	# 	s = str(start) + " - " + str(finish)
	# 	print(s)
	# 	print(trainSet[start:finish])
	# 	predict_diagonal(trainSet[start:finish], testSet ,
	# 						dim=5, tau=2, epsilon=0.2, lambd=5, percent=1, titleOfGraph = s)

	# 	plt.show()
		

