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


# Calculating probability(x) from normal distribution(mean,std deviation,x)
def normal_probability(x, mean, var):
    # var = float(sd)*float(sd)
    p=1.0
    k=32
    for i in range(k):
    	denom = math.sqrt((2*3.1415926*var[i]))
    	tp1 = (math.exp(-((float(x[i])-float(mean[i]))**2.0)/(2.0*var[i]))/denom)
    	# print(tp1)
    	p = p* tp1
    return p

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
feature = (np.matmul(data,eigenvectors.T))[:,0:k]



# print(feature)

# Calculating mean,variance for normal probability distribution
dic={}
for every_class in classes:
	temp=[]
	# variance=0
	count=0
	meu = np.zeros(len(feature[0]))
	variance = np.zeros(len(feature[0]))
	# sd = np.asarray(feature[0])
	
	for i in range(len(training_data)):
		xx=training_data[i]
		if xx[1]==every_class:	# alice
			meu+=feature[i]		# feature[i] has dimension 1*32
			variance+=(feature[i]**2) 
			count+=1
			temp.append(feature[i])
	meu/=count
	# meu2=np.mean(temp,axis=0)
	# print("mean")
	# print(meu)
	# var2=np.var(temp,axis=0)
	# print("variance")
	variance=(variance/count)-(meu**2)
	# print(variance)
	temp_list=[]
	temp_list.append(meu)
	temp_list.append(variance)
	dic[every_class]=temp_list
	
# Working with test data
with open(test_file) as f:
    test_data = f.read().splitlines()
# print(test_data)
output_test_data=[]
test_features=pca(test_data,mean,eigenvectors)
for ind in range(len(test_features)):
	each_feature=test_features[ind]
	prob_max=0
	for every_class in classes:
		prob=normal_probability(each_feature,dic[every_class][0],dic[every_class][1])
		# print(every_class)
		# print(prob)
		if prob>=prob_max:
			prob_max=prob
			output_class=every_class
	output_test_data.append(output_class)
	# print()
for output in output_test_data:
	print(output)



# normal_probability


# for k in range(1,(width*height)+1):
#pca_result = np.matmul(feature[:,0:k],(t[:,0:k].T))
# print(feature.shape)
# diff_matrix = pca_result-data


