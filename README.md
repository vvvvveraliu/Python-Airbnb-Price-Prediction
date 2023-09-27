# Airbnb Price Prediction & Superhost Classification 

## Introduction

Airbnb is a popular online platform focusing on short-term accommodation and homestay. Airbnb provides a marketplace for hosts (i.e., property owners) to publish their listings (spare houses, apartments, or rooms) for guests to view and lease. In New York City, there are approximately 40,000 listings available for the last quarter. This project is primarily motivated to predict the price per day of Airbnb listings based on other features and attributes, (e.g., neighborhood, number of rooms, amenities, etc.) With specific regression models chosen, we are further curious about which features contribute the most to the increase/decrease of the daily price of a listing. It is also worth mentioning that Airbnb endows a “Superhost” title to the hosts who provide the most hospitable service and receive the highest ratings. Historical data shows that superhosts account for approximately 20% of all hosts. In consideration, in this project, we also want to predict whether a host is a superhost, given the conditions of his/her listings.
  
## Source Code 
There are four ipython notebooks: 
1. Price prediction
2. Superhost classification
3. Superhost classification neural network
4. Visualizations

## Dataset
The dataset contains 39,881 Airbnb listings in New York for the last quarter, with 75 attributes such as the type of room,  the host’s response rate, etc. These data were scraped in September 2022. All raw data are in string format with units and other punctuation.

## Price Prediction 
Three models are chosen: 
- Linear Regression (Ridge)
- Regression Tree
- Neural Networks

All 49 variables are selected to conduct prediction
- For Linear regression, cross-validation and grid search is used to optimize the hyperparameter alpha.
- For the Regression Tree, grid search is used to find optimal parameters, and pre-pruning is used to prevent overfitting by setting the max depth of the decision tree model.
- The Neural Network model has 6 dense layers with ‘relu’ as activation and 3 dropout layers with a rate of 20%. To avoid weight explosion due to highly variable price data, the parameters in 3 major hidden layers are penalized by Elastic Net Regularization.

Mean squared error (MSE) is used to compare the results of the different models: 
- Regression tree performs the best with an MSE of 0.18 on test data
- The top four features: 'room_type0', 'neighbourhood_cleansed', 'bath_num', and 'accommodates'. 

## Superhost Classification 
Three types of models are chosen:
- Logistic Regression
- Trees: Decision Trees, Random Forest, Gradient Boosting Trees, and XGBoost Trees 
- Neural Networks

16 variables are used for classification. Cross-validation is performed for every model and regularization is tuned for logistic regression. Since the distribution of labels is highly imbalanced (20% superhost and 80%, not superhost), we tried different sampling methods, including stratified sampling, synthetic minority oversampling technique (SMOTE), and balanced class weights to process the data before passing them to the models. Among the three sampling methods, we found that SMOTE gives the highest AUC. Therefore, we focused on the SMOTE models and compared the results of their corresponding AUC scores for the test dataset. 

AUC scores to evaluate performance: 
- Neural Network and Random Forest models give relatively good performance among the six models.
- The top 3 features are: “host_response_rate”, “host_acceptance_rate”, and “host_has_profile_pic”.
