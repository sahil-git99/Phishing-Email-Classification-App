import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline



def get_trained_model():
    try:
        # Load dataset
        df = pd.read_csv("data/Phishing_Email.csv")
        X = df["Email Text"].fillna("")
        y = df["Email Type"]

        # data split
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Pipeline - feature er & Model
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english")),
                ("clf", LogisticRegression(solver="liblinear")),
            ]
        )


        # model training 
        pipeline.fit(X_train, y_train)

        # save the model
        model_save_location = "phishing_model.pkl"
        joblib.dump(pipeline, model_save_location)
        print("Model trained and saved as phishing_model.pkl")

        return {
            "status" : "success",
            "message" : "Model Trained Successfully",
            "output" : model_save_location
        }
    
    except Exception as e:
        print(f"Error in Model training : {e}")
        return {
            "status" : "fail",
            "message" : f"Error in Model Training : {e}",
            "output" : None
        } 
    
if __name__=="__main__":
    get_trained_model()