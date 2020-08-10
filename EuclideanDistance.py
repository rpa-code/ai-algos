from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col

dataA = [(0, Vectors.dense([1.0, 1.0]),),
         (1, Vectors.dense([1.0, -1.0]),),
         (2, Vectors.dense([-1.0, -1.0]),),
         (3, Vectors.dense([-1.0, 1.0]),)]
dfA = spark.createDataFrame(dataA, ["id", "features"])

dataB = [(4, Vectors.dense([1.0, 0.0]),),
         (5, Vectors.dense([-1.0, 0.0]),),
         (6, Vectors.dense([0.0, 1.0]),),
         (7, Vectors.dense([0.0, -1.0]),)]
dfB = spark.createDataFrame(dataB, ["id", "features"])

key = Vectors.dense([1.0, 0.0])

brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0,
                                  numHashTables=3)
model = brp.fit(dfA)
model.transform(dfA).show()

model.approxSimilarityJoin(dfA, dfB, 1.5, distCol="EuclideanDistance")\
    .select(col("datasetA.id").alias("idA"),
            col("datasetB.id").alias("idB"),
            col("EuclideanDistance")).show()

model.approxNearestNeighbors(dfA, key, 2).show()
