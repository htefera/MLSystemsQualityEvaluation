import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler

sc = SparkContext().getOrCreate()

def get_titanic(source="titanic.csv"):
	titanic = pd.read_csv(source)
	titanic.drop(columns="Cabin", inplace=True)
	
	titanic["Age"][titanic["Name"].str.contains("Mr.") & ~titanic["Sex"].str.contains("female") & titanic["Age"].isna()] = \
	round(titanic["Age"][titanic["Name"].str.contains("Mr.") & ~titanic["Sex"].str.contains("female")].mean())
	titanic["Age"][titanic["Name"].str.contains("Master.") & ~titanic["Sex"].str.contains("female") & titanic["Age"].isna()] = \
	round(titanic["Age"][titanic["Name"].str.contains("Master.") & ~titanic["Sex"].str.contains("female")].mean())
	titanic["Age"][titanic["Name"].str.contains("Mrs.") & titanic["Sex"].str.contains("female") & titanic["Age"].isna()] = \
	round(titanic["Age"][titanic["Name"].str.contains("Mrs.") & titanic["Sex"].str.contains("female")].mean())
	titanic["Age"][titanic["Name"].str.contains("Miss.") & titanic["Sex"].str.contains("female") & titanic["Age"].isna()] = \
	round(titanic["Age"][titanic["Name"].str.contains("Miss.") & titanic["Sex"].str.contains("female")].mean())
	titanic["Age"][~titanic["Name"].str.contains("Miss.") & ~titanic["Name"].str.contains("Mr.") & ~titanic["Name"].str.contains("Master.") & titanic["Age"].isna()] = \
	round(titanic["Age"][~titanic["Name"].str.contains("Miss.") & ~titanic["Name"].str.contains("Mr.") & ~titanic["Name"].str.contains("Master.")].mean())
	
	titanic["Embarked"][titanic["Embarked"].isna()] = "S"
	titanic.drop(columns=["PassengerId", "Name"], inplace=True)
	titanic.drop(index=titanic[titanic.drop(columns="Survived").duplicated()].index, inplace=True)
	titanic["Sex"] = np.where(titanic["Sex"] == "male", 1, 0)
	titanic["Ticket"].replace(list(titanic["Ticket"].value_counts().index), range(len(titanic["Ticket"].value_counts())), inplace=True)
	titanic["Embarked"].replace(list(titanic["Embarked"].value_counts().index), range(len(titanic["Embarked"].value_counts())), inplace=True)
	add = titanic.drop(columns="Survived").sum(axis=1)
	titanic = titanic[(add < add.quantile(0.9)) & (add > add.quantile(0.1))]
	titanic = pd.concat([titanic[titanic['Survived'] == 0].sample(len(titanic[titanic['Survived'] == 1])),\
                     titanic[titanic['Survived'] == 1]], axis=0)
	col = list(titanic.columns)
	col.remove("Survived")
	for i in col:
		titanic[i] = (titanic[i] - titanic[i].mean()) / titanic[i].std()
	return titanic
    
def get_diabetes(source="diabetes.csv", seed=0):
	diabetes = pd.read_csv(source)
	col = list(diabetes.columns)
	col.remove("Outcome")
	for i in col:
		diabetes[i] = (diabetes[i] - diabetes[i].mean()) / diabetes[i].std()
	add = diabetes.drop(columns="Outcome").sum(axis=1)
	diabetes = diabetes[(add < add.quantile(0.95)) & (add > add.quantile(0.05))]
	diabetes = pd.concat([diabetes[diabetes['Outcome'] == 0].sample(len(diabetes[diabetes['Outcome'] == 1]), \
                    random_state=seed), diabetes[diabetes['Outcome'] == 1]], axis=0)
	return diabetes

def get_fashion(source="fashion-mnist_train.csv"):
	fashion = pd.read_csv(source)
	#fashion = fashion[:1000]
	col = list(fashion.columns)
	col.remove("label")
	for i in col:
		if ((fashion[i]>0).sum())<25000:
		    fashion.drop(columns=i, inplace=True)
	fashion.drop(index=fashion[fashion.drop(columns="label").duplicated()].index, inplace=True)
	col = list(fashion.columns)
	col.remove("label")
	for i in col:
		fashion[i] = fashion[i] / 255
	new = fashion.iloc[0:2]
	new.drop(index=[0,1], inplace=True)
	for i in range(10):
		add = fashion[fashion["label"]==i].drop(columns="label").sum(axis=1)
		new = pd.concat([new, fashion[fashion["label"]==i][(add < add.quantile(0.99)) & (add > add.quantile(0.01))]], axis=0)
	fashion = new.sort_index()
	return fashion

def get_sklearn_titanic(source="train.csv", test_size=0.2, seed=0):
	titanic = get_titanic(source)
	titanic_label = titanic["Survived"]
	titanic.drop(columns="Survived", inplace=True)
	return train_test_split(titanic, titanic_label, test_size=test_size, random_state=seed)
	
def get_pyspark_titanic(source="train.csv", test_size=0.2, seed=0):
	titanic = get_titanic(source)
	titanic = titanic.rename(columns={'Survived': 'label'})
	sqlContext = SQLContext(sc)
	data = sqlContext.createDataFrame(titanic)
	col = list(titanic.columns)
	col.remove("label")
	va = VectorAssembler(inputCols = col, outputCol='features')
	va_df = va.transform(data)
	va_df = va_df.select(['features', 'label'])
	return va_df.randomSplit([1-test_size, test_size], seed=seed)
	
def get_sklearn_diabetes(source="diabetes.csv", test_size=0.2, seed=0):
	diabetes = get_diabetes(source)
	diabetes_label = diabetes["Outcome"]
	diabetes.drop(columns="Outcome", inplace=True)
	return train_test_split(diabetes, diabetes_label, test_size=test_size, random_state=seed)
	
def get_pyspark_diabetes(source="diabetes.csv", test_size=0.2, seed=0):
	diabetes = get_diabetes(source)
	diabetes = diabetes.rename(columns={'Outcome': 'label'})
	sqlContext = SQLContext(sc)
	data = sqlContext.createDataFrame(diabetes)
	col = list(diabetes.columns)
	col.remove("label")
	va = VectorAssembler(inputCols = col, outputCol='features')
	va_df = va.transform(data)
	va_df = va_df.select(['features', 'label'])
	return va_df.randomSplit([1-test_size, test_size], seed=seed)

def get_sklearn_fashion(source="fashion-mnist_train.csv", test_size=0.2, seed=0):
	fashion = get_fashion(source)
	fashion_label = fashion["label"]
	fashion.drop(columns="label", inplace=True)
	return train_test_split(fashion, fashion_label, test_size=test_size, random_state=seed)
	
def get_pyspark_fashion(source="fashion-mnist_train.csv", test_size=0.2, seed=0):
	fashion = get_fashion(source)
	sqlContext = SQLContext(sc)
	data = sqlContext.createDataFrame(fashion)
	col = list(fashion.columns)
	col.remove("label")
	va = VectorAssembler(inputCols = col, outputCol='features')
	va_df = va.transform(data)
	va_df = va_df.select(['features', 'label'])
	return va_df.randomSplit([1-test_size, test_size], seed=seed)
