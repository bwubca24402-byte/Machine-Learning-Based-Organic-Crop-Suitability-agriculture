"""Train and predict using a Random Forest classifier."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import load_dataset, split_data


def train_model(dataset_path="organic_farming_suitability_1000_source_calibrated.csv"):
    df = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = split_data(df)
    clf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    metrics = {"accuracy": accuracy_score(y_test, predictions),
               "classification_report": classification_report(y_test, predictions, zero_division=0)}
    return clf, metrics


# App can import this function and train the model on startup.

def predict_suitability(model, values):
    """values: dictionary with keys matching FEATURES."""
    import pandas as pd
    row = pd.DataFrame([values])
    prediction = model.predict(row)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(max(model.predict_proba(row)[0]))
    return prediction, probability
