#!/usr/bin/env python3

#Diabetes Prediction Using Support Vector Machine
import pickle
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
spark=SparkSession.builder.appName('log_reg').getOrCreate()

#For training
def train():
    schema = StructType([
    StructField("Pregnancies", DoubleType()),
    StructField("Glucose", DoubleType()),
    StructField("BloodPressure", DoubleType()),
    StructField("SkinThickness", DoubleType()),
    StructField("Insulin", DoubleType()),
    StructField("BMI", DoubleType()),
    StructField("DiabetesPedigreeFunction", DoubleType()),
    StructField("Age", DoubleType()),
    StructField("Outcome", DoubleType())
])
    df = spark.read.schema(schema).csv("/home/admin/Downloads/diabetes.csv",header=True)
    df_assembler = VectorAssembler(inputCols=[
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'], outputCol="features")
    df = df_assembler.transform(df)
    model_df=df.select(['features','Outcome'])
    train_df,test_df=model_df.randomSplit([0.75,0.25])
    rf_classifier=RandomForestClassifier(labelCol='Outcome',numTrees=50).fit(train_df)
    rf_predictions=rf_classifier.transform(test_df)
    rf_accuracy=MulticlassClassificationEvaluator(labelCol='Outcome',metricName='accuracy').evaluate(rf_predictions)
    print(rf_accuracy)
    #Save Model As Pickle File
    rf_classifier.save("/home/admin/Downloads/RF_model")

#Test accuracy of the model
def test(X_test,Y_test):
    with open('svc.pkl','rb') as mod:
        p=pickle.load(mod)
    
    pre=p.predict(X_test)
    print (accuracy_score(Y_test,pre)) #Prints the accuracy of the model


def find_data_file(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen.
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen.
        datadir = os.path.dirname(__file__)

    return os.path.join(datadir, filename)


def check_input(data) ->int :
    spark = SparkSession.builder.appName('abc').enableHiveSupport().getOrCreate()
    sc = spark.sparkContext
    rdd = sc.parallelize([data])
    df = spark.read.json(rdd)
    rdd = sc.parallelize([data])
    df = spark.read.json(rdd)
    df_assembler = VectorAssembler(inputCols=['B', 'C', 'D','E','F','G','H','I'], outputCol="features")
    df = df_assembler.transform(df)
    model_df = df.select('features')
    rf=RandomForestClassificationModel.load("/home/admin/Downloads/RF_model")  
    model_preditions=rf.transform(model_df)
    model_preditions = model_preditions.toPandas()['prediction'].values.tolist()
    return model_preditions[0]
if __name__=='__main__':
    train()    
