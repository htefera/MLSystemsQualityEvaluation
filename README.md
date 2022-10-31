# On The Problem of Software Quality In Machine Learning Systems
## Abstract
Machine learning quality evaluation for classification algorithms with model behavioral testing, which is the state of the solution for model testing to improve data and model quality.
## Objective
Evaluating the quality of ML classification algorithms for 17 different classifiers from Spark ML, Keras, and Scikit-learn to detect or minimize ML bugs at an early stage before a model is deployed. This is accomplished through code testing of the data processing, training, and evaluation method. Model testing to check if what the model has learned is correctly applied. Testing of data by writing pre-train tests before the data feed to the ML models. In addition, to improve the quality of the classifier, evaluate them with selected model quality attributes (properties) on three popular classification datasets from [Kaggle](https://www.kaggle.com/) 
## Machine Learning Testing Technique Used
We took significant inspiration from model behavioral testing for NLP models, Deepchecks, a tool for assessing data and model quality, and Drifter-ML, a novel framework for performance testing of classification models

1.  **Model Pre-train tests** 
    * To ensure data quality by writing assertions on various characteristics of the given data
    * Also we used Deepchecks to detect duplicates, type mismatches, train test distributions, etc 
2.  **Model Post-train tests** to evaluate the behavior of the trained models
     * Minimum Functionality test(robustness testing)
     * Invariant tests: testing to changes in the less relevant features and see how prediction varies  
     * Directional expectation tests: testing for changes in relevant features and checking how the model prediction reacts 
3. **Model Performance Evaluation**
   * We used a machine learing model validation tool ***Drifter-ML*** to evaluate performance thresholds
5. **Data and Model drifts using DeepChecks**
   * We used Deepcheecks for testing the train and test data for various features and for testing the behavior of the trained model 
   * We used Deepecheck's full_suite() module to check data and model drifts of the various models and the three datasets chosen for the experimentation    
# Datasets Used for the Experimentation
We used three popular classification datasets from Kaggle to evaluate the classification algorithms
1. **Fashion MNIST Dataset** 
  *  The dataset consists of 60,000 training examples and 10,000 test examples of Zalando's article images. 
2. **Titanic Disaster Dataset**
  *  The dataset describes the survival status of individual passengers as a result of the sinking of the Titanic in early 1912.
3. **Pima Indian Diabetes Dataset**
  * The dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It is used to determine whether or not a patient has diabetes, based on certain diagnostic measurements.

# Machine Learning Algorithms
  * We selected 17 algorithms from three ML libraries: Sxikit-learn, Spark ML, and Keras
  * 8 algorithms from Spark ML and Scikit-learn are chosen for having the same mathematical intuition behind and set of hyperparameters
  * Keras Network Classifier was also considered 

## Scikit-Learn Classifiers
**We have chosen 8 classifiers from the Scikit-Learn Machine Learning Library**
1.  LinearSVC
2.  LogisticRegression
3.  DecisionTreeClassifier
4.  RandomForestClassifier
5.  GaussianNB
6.  GradientBoostingClassifier
7.  OneVsRest
8.  MLPClassifier
## Spark ML classifiers
**We have chosen 8 classifiers from the Spark ML package**
1. LinearSVC
2. LogisticRegression
3. DecisionTreeClassifier
4. RandomForestClassifier
5. NaiveBayes(modeltype='Gussian')
6. GBTClassifier
7. OneVsRestClassifier
8. MLPClassifier

## Keras Classifier 
**We selected the general Keras classifiers from the Keras Deep Learning Library**
* Keras Classifier


## Evaluation Metrics
We have used the following quantitative and qualitative machine learning properties to evaluate and analyze the various classifiers
* **Performance score**
* **Robustness**
    * How the model reacts to slight changes in the Input data. It can be achieved by how the model reacts to relevant and irrelevant features
* **Reproducibility**
    * Model and ML system reproducibility 
    * MLflow to track experiment artifacts  
* **Explainability(Interpretability)** 
     * Visualizing ML Models prediction with LIME
* dd

## Tools and Software
We used the following tools to develop the machine-learning prototype 
1. ML Frameworks: Scikit-Learn, Spark ML, and Keras 
2. IDE: Jupyter Notebook
3. Programming Language: Python
4. [MLflow: An open-source platform for the machine learning lifecycle](https://mlflow.org/)
   * To track experiment hyperparameters, performance scores, visualizations, cleaned and processed data, model pickle files, etc   
5. [Deepchecks: Testing Machine Learning Models: ](https://deepchecks.com/)
    * We used deepchecks to detect data and model drifts such as data leakage between train and test data, train test size ratio, etc
7. 
