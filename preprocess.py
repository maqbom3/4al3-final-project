import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy import sparse
import os

def preprocess_data(path, outdir= "preprocessed", test_size=0.2, random_state=42, save_transformers=True):
    
    #defeining our data schema

    BINARY_FEATURES = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC", "FAF" ] # these features are binary in response. yes/nos or two categories
    ORDINAL_FEATURES = ["FCVC", "NCP", "CH2O", "CAEC", "CALC", "TUE"] #these features have an order but are categorical. "how many times do you x"- questions
    NUMERICAL_FEATURES = ["Age"] 
    NOMINAL_FEATURES =[ "MTRANS"] #transportation method- no order, multiple categories

    TARGET = "Obesity_Level"  #TODO- RENAME THE FEATURE COL IN ACCORDANCE TO PREPROCESSING IN DOC

    DROP_FEATURES = ["Height", "Weight"] # we drop these features as they are used to calculate BMI which is already included in the dataset
    



    df = pd.read_csv(path)
    df = df.rename(columns={"NObeyesdad": TARGET})
    df = df.drop(columns=[c for c in DROP_FEATURES if c in df.columns], errors="ignore")
    
    y_raw = df[TARGET] 
    X_raw = df.drop(columns=[TARGET]) 

    for c in set(BINARY_FEATURES + NOMINAL_FEATURES) & set(X_raw.columns):
        X_raw[c] = X_raw[c].astype(str).str.strip() #get rid of whitespice in categorical features

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=random_state, stratify = y_raw) #split before we fit any transformers to avoid data leakage
    
    bin_cols = [col for col in BINARY_FEATURES if col in X_train.columns]
    ord_cols = [col for col in ORDINAL_FEATURES if col in X_train.columns]
    num_cols = [col for col in NUMERICAL_FEATURES if col in X_train.columns]
    nom_cols = [col for col in NOMINAL_FEATURES if col in X_train.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            # fill missing values with most frequent instead of dropping,  one hot encode a binary, 
            ("bin", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(sparse_output=True, handle_unknown="ignore", drop="if_binary")),
            ]), bin_cols),

            # imput median if missing valuue, scale (no centering for sparse safety)
            ("ordnum", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False)),  # with_mean=False for sparse safety
            ]), ord_cols + num_cols),

            # impute most frequent, one hot encode
            ("nom", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(sparse_output=True, handle_unknown="ignore")),
            ]), nom_cols)
        ],
        remainder="drop",
        sparse_threshold=1.0
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    y_encoder = LabelEncoder()
    y_train = y_encoder.fit_transform(y_train_raw)
    y_test = y_encoder.transform(y_test_raw)

    if save_transformers:
        os.makedirs(outdir, exist_ok=True)
        sparse.save_npz(os.path.join(outdir, "X_train.npz"), X_train_processed)
        sparse.save_npz(os.path.join(outdir, "X_test.npz"),  X_test_processed)
        np.save(os.path.join(outdir, "y_train.npy"), y_train)
        np.save(os.path.join(outdir, "y_test.npy"),  y_test)
        joblib.dump(preprocessor, os.path.join(outdir, "preprocessor.pkl"))
        joblib.dump(y_encoder,   os.path.join(outdir, "label_encoder.pkl"))

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, y_encoder


    



