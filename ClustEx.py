import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing


data = pd.read_csv('3.12. Example.csv')

kmeans = KMeans(2)
kmeans.fit(data)

result = kmeans.fit_predict(data)
data['Cluster'] = result


# plt.scatter(data['Satisfaction'], data['Loyalty'], c=data['Cluster'])
# plt.xlabel('Satisfaction')
# plt.ylabel('Loyalty')
# plt.show()

# Since loyalty is scaled and has much lower value than Satisfaction, which is not scaled, Clustering is
# done on the basis of Satisfaction
# Hence, we need to scale Satisfaction as well

data = data.drop(['Cluster'], axis=1)
data_scaled = preprocessing.scale(data)
print(data_scaled)


wcss = []

for i in range(1, 9):
    kmeans = KMeans(i)
    kmeans.fit(data_scaled)
    kmeans.fit_predict(data_scaled)
    wcss.append(kmeans.inertia_)

print(wcss)

# Plotting WCSS
num_cluster = range(1, 9)
plt.plot(num_cluster, wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Depending on our requirement, we can use either 2 clusters or 4


kmeans = KMeans(2)
kmeans.fit(data_scaled)

result = kmeans.fit_predict(data_scaled)
data['Cluster'] = result

plt.scatter(data['Satisfaction'], data['Loyalty'], c=data['Cluster'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()