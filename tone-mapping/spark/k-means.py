
import cv2
import numpy as np
from numpy import array
from math import sqrt
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel


def gamma_correction(value, gamma, f_stop):
	return pow((value*pow(2,f_stop)),(1.0/gamma));

# Load and parse the data
sc = SparkContext(appName="Kmeans")
#data = sc.textFile("kmeans_data.txt")
data = cv2.imread('../imagesHDR/.exr', -1)
test_data = data.reshape(data.shape[0] * data.shape[1], 3)
#parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))
#result = np.array(ldr, dtype=np.float32).reshape(data.shape[0], data.shape[1], 3)

# Build the model (cluster the data)
clusters = KMeans.train(test_data, 2, maxIterations=10, initializationMode="random")
ldr = hdr.map(lambda pixel: [gamma_correction(val, 1.2, 0.4) for val in pixel]).collect()
# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = ldr.map(lambda point: error(point))
result = np.array(WSSSE, dtype=np.float32).reshape(data.shape[0], data.shape[1], 3)
print("Within Set Sum of Squared Error = " + str(WSSSE))
cv2.imwrite('result_spark.png', result*255)
sc.stop()

# Save and load model
#clusters.save(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
#sameModel = KMeansModel.load(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
