import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
image = cv2.imread('imagen1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
pixel_values = image_lab.reshape((-5, 6))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
K = 4  
_, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image_lab.shape)
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2RGB) 

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image_rgb)
plt.title('Imagen Segmentada')
plt.axis('off')

plt.show()