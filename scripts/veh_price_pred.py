import pandas as pd
import numpy as np
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import plotly.express as px
import shap
shap.initjs()

import warnings
warnings.filterwarnings("ignore")

def read_csv(url):
    
    """
    Takes in the url of the dataset
    Reads it into a Pandas DataFrame
    
    Returns DataFrame 
    """
    df = pd.read_csv(url)
    return df

def preprocessing(df):
    
    """
    This function takes in the DataFrame
    Cleans, transforms, and prepares it
    For machine learning

    Steps taken:
    1. Replace invalid values in year column (0, 2557, 2561, 2550) with NaN
    2. Replace recondition with reconditioned and e with NaN
    3. Create list of valid transmission types and use it to filter out invalid transmission type
    4. Group or add SUV / 4X4 to SUV values, unregistered to other values in the body column
    5. Convert values in capacity column to int
    6. Convert values in mileage column to int
    7. Convert values in price column to int
    8. Drop redundant columns
    9. Handle missing values (drop column with missing % > 51)
    10. Impute missing values in price column via regression
    11. Drop rows containing remaining missing values < 5%
    12. Rename price, mileage, and capacity columns to reflect their units
    13. Create new price column in pound sterling
    14. Split DataFrame into features and target variables
    15. Split data into train, validation and test sets
    16. Feature scaling

    Returns X, y, X_train, X_test, X_val, y_train, y_test, y_val
    """
    
    df["Year"] = df["Year"].replace([0, 2557, 2561, 2550], np.nan)
    df["Condition"] = df["Condition"].replace({"Recondition": "Reconditioned", "e": np.nan})
    valid_transmissions = ["Automatic", "Manual", "Tiptronic", "Other transmission"]
    df = df[df.Transmission.isin(valid_transmissions)]
    df["Body"] = df["Body"].replace("SUV / 4x4", "SUV")
    df.loc[:, "Body"] = df["Body"].replace("Unregistered", "Other")
    
    df["Capacity"] = df["Capacity"].str.replace(",", "")
    df["Capacity"] = df["Capacity"].str.replace("cc", "")
    df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
    df["Mileage"] = df["Mileage"].str.replace(",", "")
    df["Mileage"] = df["Mileage"].str.replace("km", "")
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
    df["Price"] = df["Price"].str.replace(",", "")
    df["Price"] = df["Price"].str.replace("Rs", "")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    
    df = df.drop(["Title", "Sub_title", "Seller_name", "published_date", "Description", "Post_URL"], axis=1)
    
    missing_percentage = df.isna().sum() / len(df) * 100
    for col in df.columns:
        if missing_percentage[col] > 51:
            df = df.drop(col, axis=1)      
    
    price_missing = df[df["Price"].isna()]
    price_not_missing = df[~df["Price"].isna()]
    X = price_not_missing.drop("Price", axis=1)
    y = price_not_missing["Price"]
    X_missing = price_missing.drop("Price", axis=1)
    X = pd.get_dummies(X, drop_first=True)
    X_missing = pd.get_dummies(X_missing, drop_first=True)
    X_missing = X_missing.reindex(columns=X.columns, fill_value=0)
    
    model = RandomForestRegressor(random_state=36)
    model.fit(X, y)
    predicted_prices = model.predict(X_missing)
    predicted_prices
    df.loc[df["Price"].isna(), "Price"] = predicted_prices
    df = df.dropna()

    if df.isna().sum().sum() == 0:
        print("Hurray!!! There are no more missing values in the dataset.")
    else:
        print("Keep cleaning!")

    df.rename(columns={"Price": "Price (Rs)", "Mileage": "Mileage (km)", \
                            "Capacity": "Capacity (cc)"}, inplace=True)
    
    df["Price (GBP)"] = df["Price (Rs)"] * 0.0027
    df["Price (GBP)"] = df["Price (GBP)"].round(2)
    df = df.drop("Price (Rs)", axis=1)

    cat_columns = df.select_dtypes(include="object").columns
    for col in cat_columns:
        freq_encoding = df[col].value_counts()
        df[col] = df[col].map(freq_encoding)

    X = df.drop("Price (GBP)", axis=1)
    y = df["Price (GBP)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=36)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X, y, X_train, X_test, X_val, y_train, y_test, y_val

#load and preprocess data
url = "https://raw.githubusercontent.com/toobask/introtoAI1/main/dataset/vehicle_data.csv"
df = read_csv(url)
X, y, X_train, X_test, X_val, y_train, y_test, y_val = preprocessing(df)


def baseline_model(X_train, y_train):
    
    """
    This function instantiates a Linear Regressor
    Fits it to the train data

    Returns the fitted model
    """
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    return lr_model

def model_one(X_train, y_train):
    
    """
    This function instantiates a Decision Tree Regressor
    With some parameters like, criterion, maximum depth, & minimuum samples split
    Fits it to the train data

    Returns the fitted model
    """
    dt_model = DecisionTreeRegressor(
        criterion="squared_error", 
        max_depth=30, 
        min_samples_split=10, 
        random_state=36
        )
    dt_model.fit(X_train, y_train)

    return dt_model

def model_two(X_train, y_train):
    
    """
    This function instantiates a Kneighbors Regressor
    With some parameters like, number of neighbors, weights, & metrics
    Fits it to the train data

    Returns the fitted model
    """
    kn_model = KNeighborsRegressor(
        n_neighbors=10, 
        weights="distance", 
        metric="manhattan"
        )
    kn_model.fit(X_train, y_train)

    return kn_model

def model_three(X_train, y_train):
    
    """
    This function instantiates a Random Forest Regressor
    With some parameters like, number of estimators, maximum depth, & maximum features
    Fits it to the train data

    Returns the fitted model
    """
    rf_model = RandomForestRegressor(
        n_estimators=300, 
        max_depth=30, 
        max_features="sqrt", 
        random_state=36)
    rf_model.fit(X_train, y_train)

    return rf_model

def model_four(X_train, y_train, X_val, y_val):
    
    """
    This function instantiates a Sequential Model from tensor flow
    With some parameters like, hidden layers, neurons, regularizers, & activation functions
    Compiles the model using mean_squared_error and Adam as the optimizer
    
    Fits it to the train data with some validation data passed to check progress

    Returns the fitted model
    """
    #set random seed for reproducibility
    seed = 36
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #build model
    nn_model = Sequential()
    nn_model.add(Dense(1024,
                    input_dim = X_train.shape[1],
                    activation="relu",
                    kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)
                    )
    )
    nn_model.add(Dense(512,
                    activation="relu",
                    kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)
                    )
    )
    nn_model.add(Dense(256,
                    activation="relu",
                    kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)
                    )
    )
    nn_model.add(Dense(1))

    #set learning rate
    learning_rate = 0.01

    #create optimizer
    optimizer = Adam(learning_rate=learning_rate)

    #compile model
    nn_model.compile(loss="mean_squared_error", optimizer=optimizer)

    #set epochs and batch size
    epochs = 50
    batch_size = 32

    #train model
    nn_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        validation_data=(X_val, y_val))
    
    #retrain model for 100 epochs
    epochs = 100
    
    nn_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        validation_data=(X_val, y_val))
    
    #retrain model for 50 epochs
    epochs = 50
    
    nn_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        validation_data=(X_val, y_val))
    
    return nn_model

def feature_importances(model, X, X_val=None, y_val=None):

    """
    This function calculates and plots feature importances for a given model

    Args:
        model: The trained machine learning model.
        X: The feature matrix.
        X_val: The validation feature matrix (optional, used for permutation importance)
        y_val: The validation target variable (optional, used for permutation importance)

    Returns figure
    """

    if isinstance(model, LinearRegression):
        feature_importance = pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": model.coef_
            }
        )

    elif isinstance(model, (DecisionTreeRegressor, RandomForestRegressor)):
        feature_importance = pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }
        )

    else:
        #check if X_val and y_val are provided
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val are required for permutation importance.")

        perm_importance = permutation_importance(model,
                                                 X_val,
                                                 y_val,
                                                 n_repeats=10,
                                                 scoring="neg_mean_squared_error",
                                                 random_state=36)
        feature_importance = pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": perm_importance.importances_mean
            }
        )

    #sort features in order of importance
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    #plot features importance
    figure = plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    return figure

#plot features importance for model three
model3 = model_three(X_train, y_train)
feature_importances(model3, X)

def evaluate_model(model, X_test, y_test):
    
    """
    This function takes in the model, gets predictions 
    Evaluates test sets

    Parameters
    - model: trained model
    - X_test: test set features
    - y_test: test set labels
    
    Returns predictions, root mean squared error and r2 score
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return y_pred, rmse, r2

#load baseline model, get predictions, and evaluate
baseline_model = baseline_model(X_train, y_train)
y_pred_baseline, rmse_baseline, r2_baseline = evaluate_model(baseline_model, X_test, y_test)

#load model one, get predictions, and evaluate
model1 = model_one(X_train, y_train)
y_pred1, rmse1, r2_1 = evaluate_model(model1, X_test, y_test)

#load model two, get predictions, and evaluate
model2 = model_two(X_train, y_train)
y_pred2, rmse2, r2_2 = evaluate_model(model2, X_test, y_test)

#load three, get predictions, and evaluate
model3 = model_three(X_train, y_train)
y_pred3, rmse3, r2_3 = evaluate_model(model3, X_test, y_test)

#load model four, get predictions, and evaluate
model4 = model_one(X_train, y_train, X_val, y_val)
y_pred4, rmse4, r2_4 = evaluate_model(model4, X_test, y_test)

def compare_model(
        y_test, 
        y_pred_baseline, 
        y_pred1, 
        y_pred2, 
        y_pred3, 
        y_pred4,
):
    
    """
    This function takes in the actual price and price predictions of the models
    Creates a dataframe to store actual price and price predictions
    Creates a dataframe to store metrics

    Returns price predictions and metrics dataframe
    """
    #create a df to compare prices
    price_comparison = pd.DataFrame({
        "Actual Price" : y_test,
        "Baseline Price": y_pred_baseline,
        "Model1 Price" : y_pred1,
        "Model2 Price" : y_pred2,
        "Model3 Price" : y_pred3,
        "Model4 Price" : y_pred4
    })

    #create a data frame to compare metrics
    metrics = [
        {
            "Model" : "Baseline (LR)",
            "RMSE" : rmse_baseline,
            "R2" : r2_baseline
        },
        {
            "Model" : "Model One (DT)",
            "RMSE" : rmse1,
            "R2" : r2_1
        },
        {
            "Model" : "Model Two (KN)",
            "RMSE" : rmse2,
            "R2" : r2_2
        },
        {
            "Model" : "Model Three (RF)",
            "RMSE" : rmse3,
            "R2" : r2_3
        },
        {
            "Model" : "Model Four (NN)",
            "RMSE" : rmse4,
            "R2" : r2_4
        }
    ]

    metrics_data = pd.DataFrame(metrics)

    return price_comparison, metrics_data

#retrieve price comparison and metrics dataframes
price_comparison, metrics_df = compare_model(
    y_test, 
    y_pred_baseline, 
    y_pred1, 
    y_pred2, 
    y_pred3, 
    y_pred4)

def plot_predictions(y_test, predictions, y_label="", title=""):
    
    """
    This function takes in the actual prices and predicted prices
    Makes a plot of actual vs predictions

    Returns plot
    """
    figure = plt.figure(figsize=(10, 6))

    plt.scatter(y_test, y_test, color="blue", label="Test Data")
    plt.scatter(y_test, predictions, color="red", label="Predictions")
    plt.xlabel("Actual Price")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    return figure

#get plot predictions for baseline
plot_predictions(y_test, y_pred_baseline)

def plot_comparison(metrics_df):

    """
    This function takes in a dataframe of metrics
    Retrieves metrics RMSE and R2
    Makes two plots to compare RMSE and R2 across models

    Returns plots
    """
    #plot rmse
    fig1 = px.bar(
        metrics_df,
        x="Model",
        y="RMSE",
        title="RMSE Comparison Across Models",
        text="RMSE",
        color="RMSE",
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig1.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig1.update_layout(
        xaxis_title="Models",
        yaxis_title="Root Mean Square Error (RMSE)",
        template="plotly_white",
        title_font=dict(size=20),
        showlegend=False
    )

    fig1.show()

    #plot r score
    fig2 = px.bar(
        metrics_df,
        x="Model",
        y="R2",
        title="R² Comparison Across Models",
        text="R2",
        color="R2",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig2.update_layout(
        xaxis_title="Models",
        yaxis_title="R² Score",
        template="plotly_white",
        title_font=dict(size=20),
        showlegend=False
    )

    fig2.show()

    return fig1, fig2

fig1, fig2 = plot_comparison(metrics_df)