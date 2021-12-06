import socket
import json
import numpy as np

from pyspark.context import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.sql.functions import udf

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

classfier1 = MultinomialNB()
classifer2 = BernoulliNB()

le = preprocessing.LabelEncoder()
v = HashingVectorizer(alternate_sign=False)


def cleaningDataset(string):
    stopwrds = list(text.ENGLISH_STOP_WORDS)
    removeList = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "!", "(", ")", "-", "[", "]", "{", "}", "@", '#',
              "$", "%", "^", "&", "*", "_", "~", ";", ":", "'", "\"", "\\", "<", ">", ".", ",", "/", "?"]
    removeList = stopwrds + removeList
    keep = [i for i in string.split() if i.lower() not in removeList]
    cleanedDataset = " ".join(keep)

    return cleanedDataset


def main(lines):
    exists = len(lines.collect())
    if exists:
        tempDF = spark.createDataFrame(json.loads(lines.collect()[0]).values(), schema)
        removed0 = udf(cleaningDataset(), StringType())
        df0 = tempDF.withColumn("feature1", removed0(tempDF["feature1"]))
        removed1 = udf(cleaningDataset(), StringType())
        df1 = df0.withColumn("feature0", removed1(df0["feature0"]))

        label_Encoder = le.fit_transform(np.array([row["feature2"] for row in df1.collect()]))
        data = df1.collect()
        vectorizer = v.fit_transform([" ".join([row["feature0"], row["feature1"]]) for row in data])

        X_Train, X_Test, Y_Train, Y_Test = train_test_split(vectorizer, label_Encoder, test_size=0.5)

        model_1 = classfier1.partial_fit(X_Train, X_Train, classes=np.unique(Y_Train))
        pred1 = model_1.predict(X_Test)

        accuScore1 = accuracy_score(pred1, Y_Test)
        precScore1 = precision_score(Y_Test, pred1)
        recScore1 = recall_score(pred1, Y_Test)

        print("-------Model 1--------")
        print("Accuracy Score: ", accuScore1)
        print("Recall Score: ", recScore1)
        print("Precision Score: ", precScore1)



        model_2 = classifer2.partial_fit(X_Train, Y_Train, classes=np.unique(Y_Train))
        prediction2 = model_2.predict(X_Test)

        accuScore2 = accuracy_score(prediction2, Y_Test)
        recScore2 = recall_score(prediction2, Y_Test)
        precScore2 = precision_score(Y_Test, prediction2)


        print(" ------Model 2------")
        print("Accuracy Score: ", accuScore2)
        print("Recall Score: ", recScore2)
        print("Precision Score: ", precScore2)



sparkcont = SparkContext(appName="stream")
StreamCC = StreamingContext(sparkcont, 5)
spark = SparkSession(sparkcont)
schema = StructType([StructField("feature0", StringType(), True), StructField("feature1", StringType(), True),
                     StructField("feature2", StringType(), True)])

lines = StreamCC.socketTextStream("localhost", 6100)
lines.foreachRDD(main)

StreamCC.start()
StreamCC.awaitTermination()
StreamCC.stop()