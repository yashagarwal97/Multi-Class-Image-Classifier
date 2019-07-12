import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import math


# PCA algorithm begins
def pca(files,mean,eigenvectors):
	width=64
	height=64
	data=np.zeros((len(files),width*height))
	i=0
	for f in files:
		# data[i] = np.resize(np.array(Image.open(f).convert('L')),(width,height)).flatten()
		data[i] = np.array(Image.open(f).convert('L').resize((width,height),Image.ANTIALIAS)).flatten()
		i+=1
	# for xx in data:
		# print(xx)
	data=data-mean
	covariance_matrix = np.matmul(data.T,data)
	# print(covariance_matrix)
	# a1,a2,eigenvectors = np.linalg.svd(covariance_matrix)
	# a1,a2,eigenvectors = np.linalg.svd(data)
	# print(eigenvectors.shape)
	k=32
	feature = (np.matmul(data,eigenvectors.T))[:,0:k]
	return feature
	# x_axis = []
	# y_axis = []
	# t = eigenvectors.T


train_file = sys.argv[1]
test_file = sys.argv[2]

training_data=[]
with open(train_file) as f:
    lines = f.read().splitlines()

for i in lines:
	words=list(i.split())
	training_data.append(words)
# print(training_data)



files=[]
files_output=[]
classes=[]

for xx in training_data:
	files.append(xx[0])
# files = os.listdir('./dataset')
for xx in training_data:
	files_output.append(xx[1])
	if xx[1] not in classes:
		classes.append(xx[1])
# print(files)
# print(files_output)

# feature=pca(files);



width=64
height=64
data=np.zeros((len(files),width*height))
i=0
for f in files:
	# data[i] = np.resize(np.array(Image.open(f).convert('L')),(width,height)).flatten()
	data[i] = np.array(Image.open(f).convert('L').resize((width,height),Image.ANTIALIAS)).flatten()
	i+=1
# for xx in data:
	# print(xx)
mean=np.zeros(width*height)
for x in data:	
	mean+=x
mean/=len(files)
# for xx in mean:
	# print(xx)
data=data-mean
covariance_matrix = np.matmul(data.T,data)
# print(covariance_matrix)
# a1,a2,eigenvectors = np.linalg.svd(covariance_matrix)
a1,a2,eigenvectors = np.linalg.svd(data)
# print(eigenvectors.shape)
k=32
feature=np.zeros((len(files),k+1))
feature[:,0:k] = (np.matmul(data,eigenvectors.T))[:,0:k]
for i in range(len(data)):
	feature[i][k] = 1
# print(feature)  

w={}
for every_class in classes:
	w[every_class]=np.zeros(len(feature[0]))
eta =0.002
# w=np.zeros((len(classes),k+1))		
for steps in range(3000):
	for i in range(len(training_data)):
		xx=training_data[i]
		# if xx[1]==every_class:	# alice
		# print(np.dot(w[xx[1]],feature[i].T))
		denom=0
		temp_array=[]
		for every_class in classes:
			temp_array.append(np.dot(w[every_class],feature[i].T))
		max_wx = max(temp_array)
		prob = math.exp(np.dot(w[xx[1]],feature[i].T) - max_wx)

		for every_class in classes:
			# print(np.dot(w[every_class],feature[i].T) - max_wx)
			denom+=math.exp(np.dot(w[every_class],feature[i].T) - max_wx)
		prob/=denom
		w[xx[1]]+=(eta*(1-prob)*feature[i])	



	
# Working with test data
with open(test_file) as f:
    test_data = f.read().splitlines()
# print(test_data)
output_test_data=[]
test_features = np.zeros((len(test_data),33))

test_features[:,0:32]=pca(test_data,mean,eigenvectors)
for i in range(len(test_data)):
	test_features[i][32] = 1

for ind in range(len(test_features)):
	each_feature=test_features[ind]

	prob_max=np.dot(w[classes[0]],each_feature.T)
	output_class=classes[0]

	for every_class in classes:
		prob=np.dot(w[every_class],each_feature.T)
		# print(every_class)
		# print(prob)
		if prob>=prob_max:
			prob_max=prob
			output_class=every_class
	output_test_data.append(output_class)
	# print()
for output in output_test_data:
	print(output)

