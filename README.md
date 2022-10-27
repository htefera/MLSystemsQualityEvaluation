# Machine Learning Quality Evaluation for Machine Learning Systems 

## Thesis Objective

Evaluating the quality of ML classification algorithms for 17 different classifiers from Spark ML, Keras, and Scikit-learn to detect or minimize ML bugs at an early stage before a model is deployed. This can be achieved by testing the Code, the Model, and the Data and evaluating the individual classifiers using ML quality attributes using three popular classification datasets 
## Testing Approaches
We took an inspiration behavioral model testing from NLP models using

1.  Model Pre-train tests to ensure data quality
2.  Model Post-train tests to evaluate the behavior of the trained models
  * Unit testing(Minimum Functionality test)
  * Invaraint tests  
  * Directional expectation tests
3. Model Perfomance Evaluation
  * We use a tool called Drifter-ML to evalaute performance threshoulds
4.    
# Datasets Used for the Experimentation
1. **Fashion MNIST Dataset** The dataset consists of 60,000 training examples and 10,000 test examples of Zalando's article images. 
2. **Titanic Disaster Dataset** The dataset describes the survival status of individual passengers as a result of the sinking of the Titanic in early 1912.
3. **Pima Indian Diabetes Dataset** The dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It is used to determine whether or not a patient has diabetes, based on certain diagnostic measurements.

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
* Performance scores
* Robustness
* Reproducibility
* Explainability(Interpretability) 
* 

## Tools and Softwares
We used the following tools to develop the machine learning prototype 
1. ML Frameworks: Scikit-Learn, Spark ML and Keras 
2. IDE: Jupyter Notebook
3. Programming Language : Python
4. [MLflow: An open source platform for the machine learning lifecycle](https://mlflow.org/)
   * To track experiment Hyperparameters, performacne scores, visualizations, latest data used,etc   
5. [Deepchecks:Testing Machine Learning Models: ](https://deepchecks.com/)
    * To detect data and model drifts
7. 
