# [BDMA – Big Data Management and Analytics](https://bdma.ulb.ac.be/)
## [Technische Universität Berlin](https://www.tu.berlin/en/)
## Master Thesis 
## On The Problem of Software Quality In Machine Learning Systems

 1. **Supervisor**: [Prof. Dr. Volker Markl](https://www.bifold.berlin/people/Prof.%20Dr._Volker_Markl.html) <br>
 Volker Markl is a German computer scientist, database systems researcher, and a full professor <br>
 2. **Advisor**: [Juan Soto](https://www.user.tu-berlin.de/juan.soto/) <br>
 Juan Soto is an Academic Director in the Chair of Database Systems and Information Management at the Technische Universität Berlin

## Project Structure


   * **Data**:contain the datasets for experimentation 
   * **Eval**: Implement metrics and visualizations for evaluation
   * **Models**: contains implementation logic of classification models
   * **Preprocess**: stores code artifacts for explanatory data analysis
   * **Tests**: contain model post train test implementation
   * **Utils**: contain data processing logic and some helper functions for the various models and libraries


## Abstract
In an effort to discover previously undisclosed bugs, minimize possible bugs, and find discrepancies, we designed and implemented a machine-learning application to examine and evaluate seventeen (17) classification algorithms from Scikit-learn, PySpark, and Keras Network. We employed systematic and uniform data preprocessing and model pre-train testing to assert the quality of data using Pandas and Spark DataFrames.Moreover, we implemented model post-train test cases using invariant testing and directional expectation tests to examine how the model reacts to changes in relevant and irrelevant features while keeping the other features constant. From the empirical experiments, no observable differences were found to provide evidence of a previously undiscovered bug. The scikit-learn classification models slightly outperform the corresponding pyspark models, however, the variation in performance of the models from varying frameworks is negligible.<br>

## Objective
Evaluating the quality of ML classification algorithms for 17 classifiers from Spark ML, Keras, and Scikit-learn to detect or minimize ML bugs at an early stage before a model is deployed. We use concepts from model behavioral testing for NLP models called [Beyond Accuracy: Behavioral Testing of NLP models with CheckList](https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf), developed to evaluate a sentiment analysis model beyound the performance scores. <br>
 * We created a standardized approach for data preparation, model evaluation, and model robustness testing using Pandas and Spark DataFrames for 17          classifiers  
* We implemented model post-train tests to ensure learned behavior works as expected. We demonstrated this using invariant testing and directional         expectation tests using the scikit-learn saved models for a representative data instance
* We performed trained model and data drift tests using DeepCheecks
* Performing various data quality checks by writing assertions before the data feed to the ML models.
## Tools and Software
We used the following tools to develop the machine-learning prototype 
1. Open ML Systems: Scikit-Learn, Spark ML, and Keras 
2. Deelopment environment and Programming language: Jupyter Notebook and Python 
3. [MLflow: An open-source platform for the machine learning lifecycle](https://mlflow.org/)(on progress)
   *  Track experiment hyperparameters, performance scores, visualizations, processed data, model pickle files, etc   
4. [Deepchecks: Testing Machine Learning Models: ](https://deepchecks.com/)
    * We used deepchecks to evalaute data quality issues and model performance tests

# Datasets Used for the Experimentation
We used three popular classification datasets from Kaggle to evaluate the classification algorithms
1. **Fashion MNIST Dataset** 
  *  The dataset consists of 60,000 training examples and 10,000 test examples of Zalando's article images. 
2. **Titanic Disaster Dataset**
  *  The dataset describes the survival status of individual passengers as a result of the sinking of the Titanic in early 1912.
3. **Pima Indian Diabetes Dataset**
  * The dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It is used to determine whether or not a patient has diabetes, based on certain diagnostic measurements.
## Model Testing 
We took significant inspiration from model behavioral testing for NLP models, **Deepchecks**, a tool for assessing data and model quality, and **Drifter-ML**, a novel framework for performance testing of classification models

1.  **Model Pre-train tests** 
    * To ensure data quality through writing assertion on the different feature of three datasets for example, 
        * For the titanic dataset, the value of the class label must be two **1: Surived** and **0: not surived** 
        * The values of the PClass column must be 1, 2 or 3, otherwise it is unkown value   
    *  We also used **Deepchecks** to detect duplicates, type mismatches, train test distributions, etc 
2.  **Model Post-train tests**
     * We employed post-train tests to evaluate the behavior of the trained models
     * **Invariant tests:** testing to changes in the less relevant features and see how prediction varies  
     * **Directional expectation tests:** testing for changes in relevant features and checking how the model prediction reacts 
3. **Model Performance Evaluation**
   * We used a machine learing model validation tool ***Drifter-ML*** to evaluate performance thresholds of the classification models
5. **Data and Model validation using DeepChecks**
   * We used Deepchecks for testing the train and test data for numerous features and various model capabilities. 
   * For this task we used Deepecheck's full_suite() testing approach   
   

# Classification Algorithms
  * We have chosen 8 classifiers from the Scikit-Learn, 8 from PySpark and 1 from Keras
  * We have choosen classification algorithms that are equally present and have the same mathematical formulation with the same or similar set of             hyperparameters.
1.  LinearSVC
2.  LogisticRegression
3.  DecisionTreeClassifier
4.  RandomForestClassifier
5.  GaussianNB |  NaiveBayes(modeltype='Gussian')
6.  GradientBoostingClassifier |GBTClassifier
7.  OneVsRest | OneVsRestClassifier
8.  MLPClassifier
9.  Keras Network Classifier


## Comparative Analysis
The follosing screenshots are from the experimental analysis of our project
#### Accuracy of Baseline Models
![Accuracy Baseline](Images/accuracybasline.png)
#### Accuracy of Optimized Models
![Accuracy Optimized](Images/accuracyoptimized.png)
#### Recall of Optimized Models
![Recall of Optimized](Images/RecallOpimized.png)
#### Roc Curves of Opimized Models from sklearn

![Sklearn RoC Curve Optimized](Images/RoCsklearn.png)

## Model Post-Train tests using Titanic Datasets
* The following are successful tests for invariant and directional epection tests , and performace degrade tests
![Invariant directional performance](Images/Invariantdirectionalperformance.png)


* The following are some of the failed tests for **GradientBoostingClassifier**, and **LinearSVC** by changing three relevant features 
* PClass, Gender and Fare are the relevant feature of the Titanic dataset



## Future Work

 In the future, I will  implement automatic model bug detection , track model artificats, best hyperparameters, cleaned data, and publish the system online. 
