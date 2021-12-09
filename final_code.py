import json
import numpy as np

from pyspark.context import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import udf
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer, VectorAssembler
from pyspark.sql.functions import length
from pyspark.ml import Pipeline

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


sc = SparkContext(appName="stream")
ssc = StreamingContext(sc, 5)
spark = SparkSession(sc)
schema = StructType([StructField("feature0", StringType(), True), StructField("feature1", StringType(), True), StructField("feature2", StringType(), True)])
		
clf1 = MultinomialNB()
clf2 = Perceptron()
clf3 = BernoulliNB()
le = preprocessing.LabelEncoder()
v = HashingVectorizer(alternate_sign=False)


def stop_word_removal(x):						
	stopwords = list(text.ENGLISH_STOP_WORDS)
	remove = ["!", "(", ")", "-", "[", "]", "{", "}", ";", ":", "'", "\"", "\\", "<", ">", ".", ",", "/", "?", "@", '#', "$", "%", "^", "&", "*", "_", "~", "1", "2", "3", "4", "5", "6" ,"7", "8", "9", "0"]	
	remove = stopwords + remove
	keep = [i for i in x.split() if i.lower() not in remove]
	cleaned = " ".join(keep)
	return cleaned

def dataclean(df):						
	df = df.withColumn('length',length(df['feature1']))
	tokenizer = Tokenizer(inputCol="feature1", outputCol="token_text")
	stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
	count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
	idf = IDF(inputCol="c_vec", outputCol="tf_idf")
	ham_spam_to_num = StringIndexer(inputCol='feature2',outputCol='label')
	clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')
	data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
	cleaner = data_prep_pipe.fit(df)
	cleandata = cleaner.transform(df)
	return cleandata

def main(lines): 						
	exists = len(lines.collect())
	if exists:
		
		temp_df = spark.createDataFrame(json.loads(lines.collect()[0]).values(), schema)
		removed0 = udf(stop_word_removal, StringType())
		df0 = temp_df.withColumn("feature1", removed0(temp_df["feature1"]))
		removed1 = udf(stop_word_removal, StringType())
		df1 = df0.withColumn("feature0", removed1(df0["feature0"]))
		df1 = dataclean(df1)
						
		labelEncoder = le.fit_transform(np.array([i["feature2"] for i in df1.collect()]))				
		data = df1.collect()
		vectorizer = v.fit_transform([" ".join([j["feature0"], j["feature1"]]) for j in data])
	
		x_train, x_test, y_train, y_test = train_test_split(vectorizer, labelEncoder, test_size = 0.5)
		
		model1 = clf1.partial_fit(x_train, y_train, classes = np.unique(y_train))
		pred1 = model1.predict(x_test)
		
		accu_scr1 = accuracy_score(pred1,y_test)
		prec_scr1 = precision_score(y_test,pred1)
		rec_score1 = recall_score(pred1,y_test)

		print("Model 1")
		print("Accuracy Score: ", accu_scr1)
		print("Precision Score: ", prec_scr1)
		print("Recall Score: ", rec_score1)
		
		
		model2 = clf2.partial_fit(x_train, y_train, classes = np.unique(y_train))
		pred2 = model2.predict(x_test)
		
		accu_scr2 = accuracy_score(pred2,y_test)
		prec_scr2 = precision_score(y_test,pred2)
		rec_score2 = recall_score(pred2,y_test)
		
		print("Model 2")
		print("Accuracy Score: ", accu_scr2)
		print("Precision Score: ", prec_scr2)
		print("Recall Score: ", rec_score2)
		
		
		
		model3 = clf3.partial_fit(x_train, y_train, classes = np.unique(y_train))
		pred3 = model3.predict(x_test)
		
		accu_scr3 = accuracy_score(pred3,y_test)
		prec_scr3 = precision_score(y_test,pred3)
		rec_score3 = recall_score(pred3,y_test)
		
		print("Model 3")
		print("Accuracy Score: ", accu_scr3)
		print("Precision Score: ", prec_scr3)
		print("Recall Score: ", rec_score3)


lines = ssc.socketTextStream("localhost", 6100)
lines.foreachRDD(main)
		
ssc.start()
ssc.awaitTermination()
ssc.stop()
