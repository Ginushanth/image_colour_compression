from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#-------- Get image and # of clusters to creat --------#
image_file = input("Enter the full path of the image: ")
k = int(input("Enter the number of colours to compress image down to: "))

#-------- Get image RGB data --------#
im = Image.open(r'%s'%image_file)
image_data = np.array(list(im.getdata()))
width, height = im.size

#-------- Setup scatter plot --------#
fig =plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlabel("R")
ax.set_ylabel("G")
ax.set_zlabel("B")
ax.set_title("RGB Clustering")

#-------- Apply KMean clustering --------#
kmeans = KMeans(n_clusters = k).fit(image_data)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

#-------- Apply color compression to image --------#
new_image_data = np.zeros((height,width,3),dtype=np.uint8)
height_counter = 0
label_counter = 0
#Create 2-D pixel array 
for i in range(0,height):
    for j in range(0,width):
        new_image_data[i][j] = [int(centroids[labels[label_counter]][0]),int(centroids[labels[label_counter]][1]),int(centroids[labels[label_counter]][2])]
        label_counter+=1

new_image_data = np.array(new_image_data)
new_image = Image.fromarray(new_image_data, 'RGB')

#-------- Show newly generated image and graph of pixels clustered around K centroids --------#
new_image.show()
ax.scatter3D(image_data[:,0],image_data[:,1],image_data[:,2], c=image_data/255.0, s=10, alpha = 0.01)
ax.scatter3D(centroids[:,0],centroids[:,1],centroids[:,2], color ="black", s=200, marker ="x")
plt.show()

