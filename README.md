# On The Problem of Software Quality In Machine Learning Systems
## Abstract
Machine learning quality evaluation for classification models using model behavioral testing to improve data and model quality
## Thesis Objective
Evaluating the quality of ML classification algorithms for 17 different classifiers from Spark ML, Keras, and Scikit-learn to detect or minimize ML bugs at an early stage before a model is deployed. This can be achieved by testing the Code, the Model, and the Data and evaluating the individual classifiers using ML quality attributes using three popular classification datasets 
## Machine Learning Testing Technique Used
We took an inspiration behavioral model testing from NLP models using

1.  **Model Pre-train tests** 
    * To ensure data quality by writing assertions on various characterstics of the given data
    * Also we used Deepchecks to detect duplicates, type mismatches, train test distributions,etc 
2.  **Model Post-train tests** to evaluate the behavior of the trained models
     * Minimum Functionality test(robustness testing)
     * Invaraint tests: testing to changes in the less relevant features and see how prediction varies  
     * Directional expectation tests: testing to changes in relevant features and check on the model prediction reacts 
3. **Model Perfomance Evaluation**
   * We used a machine learing model valdiation tool ***Drifter-ML*** to evalaute performance threshoulds
5. **Data and Model drifts using DeepChecks**
   * We used Deepcheecks for testing the train and test data for various features and for testing the behavior of the trained model 
   * We used Deepecheck's full_suite() module to check data and model drifts of the various models and the three datasets choosen for the experimentation    
# Datasets Used for the Experimentation
We used three popular classification datasets from Kaggle to evaluate the classification algorithms
1. **Fashion MNIST Dataset** 
  *  The dataset consists of 60,000 training examples and 10,000 test examples of Zalando's article images. 
2. **Titanic Disaster Dataset**
  *  The dataset describes the survival status of individual passengers as a result of the sinking of the Titanic in early 1912.
3. **Pima Indian Diabetes Dataset**
  * The dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It is used to determine whether or not a patient has diabetes, based on certain diagnostic measurements.

# Machine Learning Libraries Considered

## Scikit-Learn Classifiers
**We have choosen 8 classifiers from the Scikit-Learn Machine Learning Library**
1.  LinearSVC
2.  LogisticRegression
3.  DecisionTreeClassifier
4.  RandomForestClassifier
5.  GaussianNB
6.  GradientBoostingClassifier
7.  OneVsRest
8.  MLPClassifier
## Spark ML classifiers
**We have choosen 8 classifiers from Spark ML package**
1. LinearSVC
2. LogisticRegression
3. DecisionTreeClassifier
4. RandomForestClassifier
5. NaiveBayes(modeltype='Gussian')
6. GBTClassifier
7. OneVsRestClassifier
8. MLPClassifier

## Keras Classifier 
**We selected the general keras classifiers from the Keras Deep Learning Library**
* Keras Classifier


## Evaluation Metrics
We have used the following quantitative and qualitative machine learning properties to evaluate and analyze the various classifiers
* **Performance score**
* **Robustness**
    * How the model reacts to slight changes in the Input data. It can be achieved how the model reacts to relevant and irrelevant features
* **Reproducibility**
    * Model and ML system reproducibility 
    * MLflow to track experts artifacts  
* **Explainability(Interpretability)** 
     * Visualizing ML Models prediction with LIME
* dd

## Tools and Softwares
We used the following tools to develop the machine learning prototype 
1. ML Frameworks: Scikit-Learn, Spark ML and Keras 
2. IDE: Jupyter Notebook
3. Programming Language : Python
4. [MLflow: An open source platform for the machine learning lifecycle](https://mlflow.org/)
   * To track experiment hyperparameters, performacne scores, visualizations, cleaned and processed data,model pickle files,etc   
5. [Deepchecks:Testing Machine Learning Models: ](https://deepchecks.com/)
    * We used deepchecks to detect data and model drifts such as data leakage between train and test data, train test size ratio, etc
7. 
