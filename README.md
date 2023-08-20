# Python-Price-Prediction

## Introduction
- Airbnb is a popular online platform focusing on short-term accommodation and homestay. Airbnb provides a marketplace for hosts (i.e., property owners) to publish their listings (spare houses, apartments, or rooms) for guests to view and lease. In New York City, there are approximately 40,000 listings available for the last quarter. This project is primarily motivated to predict the price per day of Airbnb listings based on other features and attributes, (e.g., neighborhood, number of rooms, amenities, etc.) With specific regression models chosen, we are further curious about which features contribute the most to the increase/decrease of the daily price of a listing. It is also worth mentioning that Airbnb endows a “Superhost” title to the hosts who provide the most hospitable service and receive the highest ratings. Historical data shows that superhosts account for approximately 20% of all hosts. In consideration, in this project, we also want to predict whether a host is a superhost, given the conditions of his/her listings.
  
## Source Code 
There are four ipython notebooks: 
1. Price prediction
2. Superhost classification
3. Superhost classification neural network
4. Visualizations

## Dataset
The dataset contains 39,881 Airbnb listings in New York for the last quarter, with 75 attributes such as the type of room,  the host’s response rate, etc. These data were scraped in September 2022. All raw data are in string format with units and other punctuation.

## Preprocessing
Identifiers and irrelevant attributes, such as URL links and host names, are first eliminated. Each column is then converted to its proper type using regular expression. Approximately 30 columns have missing values. Depending on whether the column is categorical or numerical, these values are replaced with false values or column averages. Listing prices range from 0 to 16,500, so listings with prices above 1,500 or below 10 are removed, leaving a proposed dataset of 39,560 rows. Data related to minimum and maximum length of stay are transformed into categorical variables: short-term and long-term, indicating whether the listing could be rented within seven days or for more than three months. The host's registration date is also converted from a date-time format to the number of months since he/she became a host. Moreover, after visualization of the correlation matrix, highly correlated variables are removed, such as the total number of listings by host and the number of listings by the host, leaving 49 variables. Numerical features are scaled using a standard scaler while categorical features are further encoded according to their characteristics. In detail, the target encoder is used for geographical data with over 200 categories, and the ordinal encoder is used for sequential data. Other categorical variables are one hot encoded. The prices are logarithmically transformed due to their large range and high skewness. Sampling is separately performed in these two tasks. Features are selected according to their characteristics and correlation to the target variables.

## Price Prediction 
Three models are chosen: 
- Linear Regression (Ridge)
- Regression Tree
- Neural Networks

All 49 variables are selected to conduct prediction
- For Linear regression, cross-validation and grid search is used to optimize the hyperparameter alpha.
- For the Regression Tree, grid search is used to find optimal parameters, and pre-pruning is used to prevent overfitting by setting the max depth of the decision tree model.
- The Neural Network model has 6 dense layers with ‘relu’ as activation and 3 dropout layers with a rate of 20%. To avoid weight explosion due to highly variable price data, the parameters in 3 major hidden layers are penalized by Elastic Net Regularization.
