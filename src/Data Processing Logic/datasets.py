import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler

def get_titanic(source):
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
	add = titanic.sum(axis=1)
	titanic = titanic[(add < add.quantile(0.9)) & (add > add.quantile(0.1))]
	titanic = pd.concat([titanic[titanic['Survived'] == 0].sample(len(titanic[titanic['Survived'] == 1])),\
                     titanic[titanic['Survived'] == 1]], axis=0)
	col = list(titanic.columns)
	col.remove("Survived")
	for i in col:
		titanic[i] = (titanic[i] - titanic[i].mean()) / titanic[i].std()
	return titanic
    
def get_sklearn_titanic(source="train.csv", test_size=0.2, seed=0):
	titanic = get_titanic(source)
	titanic_label = titanic["Survived"]
	titanic.drop(columns="Survived", inplace=True)
	return train_test_split(titanic, titanic_label, test_size=test_size, random_state=seed)
	
def get_pyspark_titanic(source="train.csv", test_size=0.2, seed=0):
	titanic = get_titanic(source)
	titanic = titanic.rename(columns={'Survived': 'label'})
    del sc
	sc = SparkContext.getOrCreate()
	sqlContext = SQLContext(sc)
	data = sqlContext.createDataFrame(titanic)
	col = list(titanic.columns)
	col.remove("label")
	va = VectorAssembler(inputCols = col, outputCol='features')
	va_df = va.transform(data)
	va_df = va_df.select(['features', 'label'])
	return va_df.randomSplit([1-test_size, test_size], seed=seed)
	
