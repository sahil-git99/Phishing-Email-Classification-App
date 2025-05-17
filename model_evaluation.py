import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib



def get_model(model_path):
    model = joblib.load(model_path)
    return model 


def data_preparation(data_path):
    try:
        df = pd.read_csv(data_path)

        X = df["Email Text"].fillna("")
        y = df["Email Type"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,
            random_state=42
            )
        
        return X_test, y_test
    except Exception as e:
        raise



def get_model_performance_metrics(test_data, predicted_data):

    accuracy_result = accuracy_score(test_data, predicted_data)
    classification_report_result = classification_report(test_data, predicted_data)

    performace_results = {
        "accuracy" : accuracy_result,
        "classification_report" : classification_report_result
    }

    print(f"Performance Results :\n", performace_results)
    return performace_results



def main_model_evaluator(data_path, model_path):
    try:
        X_test, y_test = data_preparation(data_path)

        model = get_model(model_path)

        y_pred = model.predict(X_test)

        performace_results = get_model_performance_metrics(y_test, y_pred)
        return performace_results
    except Exception as e:
        print(f"Error in calculating performance : {e}")
        return None

if __name__=="__main__":
    data_path = "data/Phishing_Email.csv"
    model_path = "phishing_model.pkl"
    main_model_evaluator(data_path, model_path)