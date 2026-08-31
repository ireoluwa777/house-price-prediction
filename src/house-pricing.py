# ==========================================
# HOUSE PRICE ANALYSIS & PREDICTION
# ==========================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv("house_prices.csv")

print(df.head())
print(df.info())
print(df.describe())


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 4. CHECK FLOOR MISSINGNESS
# ==========================================

print("\nFloor vs Property Type:")
print(pd.crosstab(
    df["Property_Type"],
    df["Floor"].isna()
))


# ==========================================
# 5. EXPLORATORY DATA ANALYSIS
# ==========================================

# Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Price_USD"], kde=True)
plt.title("Distribution of House Prices")
plt.show()


# Area vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Area_sqft",
    y="Price_USD",
    alpha=0.3
)
plt.title("Area vs House Price")
plt.show()


# Distance vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Distance_to_CityCenter_km",
    y="Price_USD",
    alpha=0.3
)
plt.title("Distance to City Center vs Price")
plt.show()


# Price by Property Type
plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df,
    x="Property_Type",
    y="Price_USD"
)
plt.xticks(rotation=45)
plt.title("House Price by Property Type")
plt.show()


# Price by Location
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df,
    x="Location",
    y="Price_USD"
)
plt.xticks(rotation=90)
plt.title("House Price by Location")
plt.show()


# ==========================================
# 6. CORRELATION
# ==========================================

numeric_cols = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Garage",
    "Year_Built",
    "Floor",
    "Distance_to_CityCenter_km",
    "Price_USD"
]

plt.figure(figsize=(10, 7))

sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()


# ==========================================
# 7. DEFINE FEATURES AND TARGET
# ==========================================

# House_ID is removed because it is just an identifier
X = df.drop(columns=["House_ID", "Price_USD"])

y = df["Price_USD"]


# ==========================================
# 8. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 9. DEFINE COLUMNS
# ==========================================

numeric_features = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Garage",
    "Year_Built",
    "Floor",
    "Distance_to_CityCenter_km"
]

categorical_features = [
    "Location",
    "Property_Type"
]


# ==========================================
# 10. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ==========================================
# 11. LINEAR REGRESSION
# ==========================================

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)


# Evaluate Linear Regression
linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\n===== LINEAR REGRESSION =====")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R²:", linear_r2)


# ==========================================
# 12. RANDOM FOREST
# ==========================================

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)


# Evaluate Random Forest
rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\n===== RANDOM FOREST =====")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R²:", rf_r2)


# ==========================================
# 13. GRADIENT BOOSTING
# ==========================================

gb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)


# Evaluate Gradient Boosting
gb_mae = mean_absolute_error(
    y_test,
    gb_predictions
)

gb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gb_predictions
    )
)

gb_r2 = r2_score(
    y_test,
    gb_predictions
)

print("\n===== GRADIENT BOOSTING =====")
print("MAE:", gb_mae)
print("RMSE:", gb_rmse)
print("R²:", gb_r2)


# ==========================================
# 14. COMPARE MODELS
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],

    "MAE": [
        linear_mae,
        rf_mae,
        gb_mae
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse,
        gb_rmse
    ],

    "R2": [
        linear_r2,
        rf_r2,
        gb_r2
    ]
})

print("\n===== MODEL COMPARISON =====")
print(results.sort_values("RMSE"))


# ==========================================
# 15. FEATURE IMPORTANCE
# ==========================================

# Get trained preprocessing and model
rf_preprocessor = rf_model.named_steps["preprocessor"]
rf_estimator = rf_model.named_steps["model"]

# Get feature names after one-hot encoding
feature_names = rf_preprocessor.get_feature_names_out()

# Get Random Forest importance
importances = rf_estimator.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\n===== TOP FEATURES =====")
print(importance_df.head(20))


# Plot feature importance
plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top Features Driving House Prices")
plt.show()


# ==========================================
# 16. ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=y_test,
    y=gb_predictions,
    alpha=0.4
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.show()


# ==========================================
# 17. SAMPLE PREDICTIONS
# ==========================================

prediction_df = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": gb_predictions
})

print("\n===== SAMPLE PREDICTIONS =====")
print(prediction_df.head(10))