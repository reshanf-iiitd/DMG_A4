import pandas as pd
import scipy.cluster.hierarchy as sch
from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import A4_PhD19006
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn import preprocessing

def runner(file_path):
		##########################   VISUALIZATION TO find out optimal cluster . . . . . . . ###########################

	data = pd.read_csv(file_path)

	data1 =data
	data = data.drop(labels= 'id',axis=1)
	# print(data.columns)



	data = data.apply(preprocessing.LabelEncoder().fit_transform)

	for (columnName, columnData) in data.iteritems():
	  data[columnName] = data[columnName].astype(int)
	distortions = []
	for i in range(1, 15):
	    km = KMeans(
	        n_clusters=i, init='random',
	        n_init=10, max_iter=250,
	        tol=1e-04, random_state=11
	    )
	    km.fit(data)
	    distortions.append(km.inertia_)


	# plot
	plt.plot(range(1, 15), distortions, marker='o')
	plt.xlabel('Number of clusters')
	plt.ylabel('Distortion')
	plt.show()

	mean = np.mean(data)
	std = np.std(data)
	# print(mean)
	# print(std)
	data = (data-mean)/(0.5*std+1e-7)
	#################################################


	#########################################################   FOR PRODUCING test.csv
	km1 = KMeans(
	    n_clusters=7, init='k-means++',
	    n_init=10, max_iter=250, 
	    tol=1e-05, random_state=11
	)

	# print(len(y_km))

	# pca = PCA(n_components=2)
	# tsne_df = pca.fit_transform(data)

	y_km1 = km1.fit_predict(data)


	# df = pd.DataFrame(
	#                    'Labels':y_km1
	    
	# }) 
	data1['Labels'] = y_km1
	# data1.append(y_km1)
	data1.to_csv('result.csv',index=False)


if __name__ == "__main__":
	print("PhD19006")
	if (len(sys.argv) == 2):
		file_path =""
		file_path = sys.argv[1]
		if os.path.exists(file_path):
			runner(file_path)
		else:
			print("Path of File is wrong")
	else:
		print("Enter Correct Argument")


	print("#########################################################################")
	print("###################### Preprocessing Done ###############################")
	print("\n\n")
	x = int(input("Enter 1 to find different Clustering Result (Any other Number to Quit)"))
	if(x==1):
		A4_PhD19006.kmeans()
		A4_PhD19006.kmeans_plus_plus()
		A4_PhD19006.aggro()
	else:
		print("Thanks")


    
    

