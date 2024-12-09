import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler

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
    4. Group or add unregistered to other values in the body column
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
url = "."
df = read_csv(url)
X, y, X_train, X_test, X_val, y_train, y_test, y_val = preprocessing(df)