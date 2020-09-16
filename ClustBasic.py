import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = pd.read_csv('3.01. Country clusters.csv')


# Creating input for the model
x = pd.DataFrame()
x['Longitude'] = data['Longitude']
x['Latitude'] = data['Latitude']


# Creating the model
kmeans = KMeans(4)  # Input is the number of clusters we want
kmeans.fit(x)

result = kmeans.fit_predict(x)
data['Cluster'] = result
# print(data)

# plt.scatter(data['Longitude'], data['Latitude'], c=data['Cluster'])
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()


###########################################################################


# Optimizing the Number of clusters

# WCSS --> Within Cluster Sum of Squares
# Higher number of clusters = Lower WCSS
# Lower number of clusters = Higher WCSS
# WCSS for cluster is maximum (Not optimal)
# WCSS if every element is a cluster = 0 (Not optimal)
# We have to find the optimal number in between


# inertia_ method gives the WCCC of a Cluster solution
# print(kmeans.inertia_)


# Finding out WCSS for each number of clusters

wcss = []

for i in range(1, 9):
    kmeans = KMeans(i)
    kmeans.fit(x)
    kmeans.fit_predict(x)
    wcss.append(kmeans.inertia_)

print(wcss)

# Plotting WCSS
num_cluster = range(1, 9)
plt.plot(num_cluster, wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# wcss list and the above plot both show that there is no significant improvement after 4 clusters
# Hence, for our problem, 4 clusters are optimal