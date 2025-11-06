import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(path, outdir= "preprocessed", test_size=0.2, random_state=42, outlierz = 3):
    
    #defeining our data schema

    BINARY_FEATURES = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"] # these features are binary in response. yes/nos or two categories
    ORDINAL_NUM_FEATURES = ["FCVC", "NCP", "CH2O", "FAF", "TUE"] #these features have a numeric order but are categorical. "how many times do you x"- questions
    ORDINAL_STRING = ["CAEC", "CALC"] #string atergorical orders, like alcohol consumption
    NUMERICAL_FEATURES = ["Age"] 
    NOMINAL_FEATURES =[ "MTRANS"] #transportation method- no order, multiple categories

    TARGET = "Obesity_Level"  

    DROP_FEATURES = ["Height", "Weight"] # we drop these features as they are used to calculate BMI which is already included in the dataset
    


    #cleanigg, renaming column and drop features in drop features
    df = pd.read_csv(path)
    df = df.rename(columns={"NObeyesdad": TARGET})
    df = df.drop(columns=[c for c in DROP_FEATURES if c in df.columns], errors="ignore")
    
    #strip whitespace from string columns
    for c in set(BINARY_FEATURES + NOMINAL_FEATURES + ORDINAL_STRING) & set(df.columns):
        df[c] = df[c].astype(str).str.strip()

    # enforce numeric types , convert to NANs if anything bad
    for c in ORDINAL_NUM_FEATURES + NUMERICAL_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(axis=0) #drop nan rows
    
    #split data
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    #OUTLIERS REMOVED ONLY ON TRAINING DATA. not test data to ensure accurate to reality
    outlier_cols = [c for c in (ORDINAL_NUM_FEATURES + NUMERICAL_FEATURES) if c in X_train.columns]
    if outlier_cols and (outlierz is not None):
        z_scaler = StandardScaler() #init scaler w std and mean of training
        z_train = z_scaler.fit_transform(X_train[outlier_cols]) #cimpute z score 
        keep = (np.abs(z_train) <= outlierz).all(axis=1)
        X_train = X_train.loc[keep].copy() #filter
        y_train = y_train.loc[keep].copy() #filter

    #dynamically check again if cols exist in dataset
    bin_cols = [c for c in BINARY_FEATURES if c in X_train.columns]
    ord_num_cols = [c for c in ORDINAL_NUM_FEATURES if c in X_train.columns]
    ord_str_cols = [c for c in ORDINAL_STRING if c in X_train.columns]
    nom_cols = [c for c in NOMINAL_FEATURES if c in X_train.columns]
    num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]

    # explicit order for CAEC/CALC according to csv
    ord_categories = [["no", "Sometimes", "Frequently", "Always"]] * len(ord_str_cols) if ord_str_cols else None

    preprocessor = ColumnTransformer(
        transformers=[
            # binarys one hot encoded
            ("bin", OneHotEncoder(sparse_output=True, handle_unknown="ignore", drop="if_binary"), bin_cols),

            # numerics scaled 
            ("ordnum", Pipeline([
                ("sc", StandardScaler(with_mean=False)),
            ]), ord_num_cols + num_cols),

            # Ordinal strings ordinal encoded + scaled
            ("ordstr", Pipeline([
                ("oe", OrdinalEncoder(categories=ord_categories)),
                ("sc", StandardScaler(with_mean=False)),
            ]), ord_str_cols),

            # nominals one hot
            ("nom", OneHotEncoder(sparse_output=True, handle_unknown="ignore"), nom_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    y_enc = LabelEncoder()
    y_train_enc = y_enc.fit_transform(y_train)
    y_test_enc  = y_enc.transform(y_test)


    #save preprocessed data
    os.makedirs(outdir, exist_ok=True)
    sparse.save_npz(os.path.join(outdir, "X_train.npz"), X_train_proc)
    sparse.save_npz(os.path.join(outdir, "X_test.npz"),  X_test_proc)
    np.save(os.path.join(outdir, "y_train.npy"), y_train_enc)
    np.save(os.path.join(outdir, "y_test.npy"),  y_test_enc)
    joblib.dump(preprocessor, os.path.join(outdir, "preprocessor.pkl"))
    joblib.dump(y_enc,       os.path.join(outdir, "label_encoder.pkl"))

    return X_train_proc, X_test_proc, y_train_enc, y_test_enc, preprocessor, y_enc


    



