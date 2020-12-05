import pandas as pd

from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

def kmeans():


# colm = ["Review" , "Love"]
	data = pd.read_csv('clustering_data.csv')

	data = data.drop(labels= 'id',axis=1)
	# print(data.columns)



	data = data.apply(preprocessing.LabelEncoder().fit_transform)

	for (columnName, columnData) in data.iteritems():
	  data[columnName] = data[columnName].astype(int)


	############################################################ KMEANS with centoird for 4 clusters    #####################3

	xx = data

	km = KMeans(
	    n_clusters=4, init='random',
	    n_init=10, max_iter=250, 
	    tol=1e-04, random_state=11
	)


	pca = PCA(n_components=2)
	data = pca.fit_transform(data)
	# print(data)
	y_km = km.fit_predict(data)
	print(len(y_km))

	print(km.cluster_centers_)

	plt.figure(figsize=(8,6))
	plt.scatter(
	    data[y_km == 0, 0], data[y_km == 0, 1],
	    s=50, c='lightgreen',
	    marker='s', edgecolor='black',
	    label='cluster 1'
	)


	plt.scatter(
	    data[y_km == 1, 0], data[y_km == 1, 1],
	    s=50, c='magenta',
	    marker='o', edgecolor='black',
	    label='cluster 2'
	)

	plt.scatter(
	    data[y_km == 2, 0], data[y_km == 2, 1],
	    s=50, c='cyan',
	    marker='v', edgecolor='black',
	    label='cluster 3'
	)

	plt.scatter(
	    data[y_km == 3, 0], data[y_km == 3, 1],
	    s=50, c='yellow',
	    marker='p', edgecolor='black',
	    label='cluster 4'
	)


	plt.scatter(
	    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
	    s=300, marker='*',
	    c='red', edgecolor='black',
	    label='centroids'
	)





	plt.legend(loc='best', bbox_to_anchor=(1, 0.5),scatterpoints=1)
	# 
	plt.xticks([x for x in range(-15,21,3)],weight='bold')
	plt.yticks(weight='bold')
	# plt.grid()
	plt.show()
	print(data)


def kmeans_plus_plus():





############################################################ KMEANS +++ with centoird for 7 clusters    #####################3


	# colm = ["Review" , "Love"]
	data = pd.read_csv('clustering_data.csv')

	data = data.drop(labels= 'id',axis=1)
	# print(data.columns)




	data = data.apply(preprocessing.LabelEncoder().fit_transform)

	for (columnName, columnData) in data.iteritems():
	  data[columnName] = data[columnName].astype(int)


	###########  PREPROCESSING ######################
	mean = np.mean(data)
	std = np.std(data)
	# print(mean)
	# print(std)
	data = (data-mean)/(0.5*std+1e-7)
	#################################################

	km = KMeans(
	    n_clusters=7, init='k-means++',
	    n_init=10, max_iter=250, 
	    tol=1e-05, random_state=11
	)

	# print(len(y_km))

	pca = PCA(n_components=2)
	tsne_df = pca.fit_transform(data)

	km = km.fit(tsne_df)
	y_km = km.predict(tsne_df)
	##########################3  TEMP ###########################

	print(km.cluster_centers_)

	plt.figure(figsize=(8,6))
	plt.scatter(
	    tsne_df[y_km == 0, 0], tsne_df[y_km == 0, 1],
	    s=50, c='lightgreen',
	    marker='s', edgecolor='black',
	    label='cluster 1'
	)


	plt.scatter(
	    tsne_df[y_km == 1, 0], tsne_df[y_km == 1, 1],
	    s=50, c='magenta',
	    marker='o', edgecolor='black',
	    label='cluster 2'
	)

	plt.scatter(
	    tsne_df[y_km == 2, 0], tsne_df[y_km == 2, 1],
	    s=50, c='cyan',
	    marker='v', edgecolor='black',
	    label='cluster 3'
	)

	plt.scatter(
	    tsne_df[y_km == 3, 0], tsne_df[y_km == 3, 1],
	    s=50, c='yellow',
	    marker='p', edgecolor='black',
	    label='cluster 4'
	)

	plt.scatter(
	    tsne_df[y_km == 4, 0], tsne_df[y_km == 4, 1],
	    s=50, c='purple',
	    marker='o', edgecolor='black',
	    label='cluster 5'
	)

	plt.scatter(
	    tsne_df[y_km == 5, 0], tsne_df[y_km == 5, 1],
	    s=50, c='indigo',
	    marker='o', edgecolor='black',
	    label='cluster 6'
	)

	plt.scatter(
	    tsne_df[y_km == 6, 0], tsne_df[y_km == 6, 1],
	    s=50, c='orange',
	    marker='o', edgecolor='black',
	    label='cluster 7'
	)

	# plot the centroids
	plt.scatter(
	    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
	    s=300, marker='*',
	    c='red', edgecolor='black',
	    label='centroids'
	)





	plt.legend(loc='best', bbox_to_anchor=(1, 0.5),scatterpoints=1)
	# 
	plt.xticks(weight='bold')
	plt.yticks(weight='bold')
	# plt.grid()
	plt.show()
	print(Counter(km.labels_))



def aggro():
		########################  AgglomerativeClustering
	# colm = ["Review" , "Love"]
	data = pd.read_csv('clustering_data.csv')

	data = data.drop(labels= 'id',axis=1)
	# print(data.columns)



	data = data.apply(preprocessing.LabelEncoder().fit_transform)

	for (columnName, columnData) in data.iteritems():
	  data[columnName] = data[columnName].astype(int)


	# print(data.shape)

	#################  In agglorameartive we form grouping between the clusters there is no concept of centers 
	km = AgglomerativeClustering(n_clusters=4)

	y_km = km.fit_predict(data)


	# create scatter plot for samples from each cluster

	pca = PCA(n_components=2)
	data = pca.fit_transform(data)
	# print(data)
	print(km.children_)
	# print(km.compute_full_tree)

	plt.figure(figsize=(8,6))
	plt.scatter(
	    data[y_km == 0, 0], data[y_km == 0, 1],
	    s=50, c='lightgreen',
	    marker='s', edgecolor='black',
	    label='cluster 1'
	)


	plt.scatter(
	    data[y_km == 1, 0], data[y_km == 1, 1],
	    s=50, c='magenta',
	    marker='o', edgecolor='black',
	    label='cluster 2'
	)

	plt.scatter(
	    data[y_km == 2, 0], data[y_km == 2, 1],
	    s=50, c='cyan',
	    marker='v', edgecolor='black',
	    label='cluster 3'
	)

	plt.scatter(
	    data[y_km == 3, 0], data[y_km == 3, 1],
	    s=50, c='yellow',
	    marker='p', edgecolor='black',
	    label='cluster 4'
	)

	

	plt.legend(loc='best', bbox_to_anchor=(1, 0.5),scatterpoints=1)
	# 
	plt.xticks([x for x in range(-15,21,3)],weight='bold')
	plt.yticks(weight='bold')
	# plt.grid()
	plt.show()


	plt.figure(figsize=(8,6))
	plt.title("Dendrograms")  
	dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))



