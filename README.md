# House-price-prediction
<h5>House Price Analysis & Prediction</h5>

This project explores a residential property dataset to identify factors associated with house prices and develops machine-learning regression models for price prediction. The workflow includes exploratory data analysis, missing-value handling, categorical encoding, model training, performance evaluation, feature-importance analysis, and actual-vs-predicted visualization.

Three regression algorithms—Linear Regression, Random Forest Regression, and Gradient Boosting Regression—are trained and evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

## Overview

This project analyses a house-price dataset to identify factors
associated with property prices and develop machine-learning
models capable of predicting house prices.

The project covers the complete machine-learning workflow,
including data exploration, missing-value handling, feature
preprocessing, model training, evaluation, and prediction.

## Problem Statement

House prices are influenced by several factors, including
property size, number of bedrooms and bathrooms, location,
property type, year built, and distance from the city centre.

The objective of this project is to analyse these factors and
develop machine-learning models that can predict house prices
based on available property characteristics.

The project also compares different regression algorithms to
determine which approach provides the most accurate predictions.

Three regression models are compared:

- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression

The dataset is stored in:<a href= "data/house_prices_large_dataset.csv"> Dataset </a>


## Features

| Feature | Description | Type |
|---|---|---|
| Area_sqft | Property area in square feet | Numerical |
| Bedrooms | Number of bedrooms | Numerical |
| Bathrooms | Number of bathrooms | Numerical |
| Garage | Garage information | Numerical |
| Year_Built | Year the property was built | Numerical |
| Floor | Property floor | Numerical |
| Distance_to_CityCenter_km | Distance to city centre | Numerical |
| Location | Property location | Categorical |
| Property_Type | Type of property | Categorical |
| Price_USD | Property price | Target |

## Methodology

The project follows the following machine-learning workflow:

1. Load the dataset
2. Inspect the dataset
3. Check for missing values
4. Perform exploratory data analysis
5. Analyse correlations
6. Separate features and target
7. Split the dataset into training and testing sets
8. Preprocess numerical and categorical features
9. Train regression models
10. Evaluate model performance
11. Compare models
12. Analyse feature importance
13. Compare actual and predicted prices


## Exploratory Data Analysis

### Price Distribution

The distribution of house prices was examined using a histogram
to understand the spread and distribution of property prices.

<img src="image/Counts by Price.png" alt="Price Distribution" width="600" height= "400" >

### Area vs Price

A scatter plot was used to examine the relationship between
property area and house price.

<img src="image/house_area vs price.png" alt="Area vs Price" width="600" height= "400" >

### Price by Property Type 
 
<img src="image/Property_Type vs Price.png" alt="Price by Property Type" width="600" height= "400" >


### Price by Location

<img src="image/Location vs Price.png" alt="Price by Location" width="600" height= "400">

## Correlation Analysis

A correlation matrix was used to examine relationships between
the numerical variables and house price.

<img src="image/correlation.png" alt="Price by Location" width="750" height= "500">

Area_sqft has a strong positive correlation with Price_USD (0.87), indicating that larger properties generally have higher prices.

## Models

Three regression algorithms were trained and compared.

### 1. Linear Regression

Linear Regression was used as a baseline model for predicting
house prices.

### 2. Random Forest Regression

Random Forest Regression was used to capture nonlinear
relationships between property characteristics and price.

The model was configured with 200 trees.

### 3. Gradient Boosting Regression

Gradient Boosting Regression was used as another tree-based
ensemble approach.

The model was configured with 200 estimators, a learning rate
of 0.05, and a maximum tree depth of 3.

## Evaluation Metrics

The models were evaluated using three metrics:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and
predicted house prices.

Lower values indicate better performance.

### Root Mean Squared Error (RMSE)

Measures the square root of the average squared prediction
error. RMSE gives greater weight to larger errors.

Lower values indicate better performance.

### R² Score

Measures the proportion of variation in the target variable
accounted for by the model.

Higher values generally indicate better performance.
