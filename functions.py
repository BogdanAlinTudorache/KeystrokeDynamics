import joblib
import os

# Get the absolute path of the directory where this file is located
basedir = os.path.abspath(os.path.dirname(__file__))


def predict_user(event):
    """ Gets the input details from the body of the POST request and returns the predicted user
    """
    # Print the event for debugging purposes
    print(event)

    # Check if the message has the correct body structure
    if ['Model', 'HT', 'PPT', 'RRT', 'RPT'] == list(event.keys()):
        print(f"Model is:", event['Model'])

        if event["Model"] == "SVM":
            # Load the trained SVM model from the joblib file
            model_path = os.path.join(basedir, 'models', 'svm_model.joblib')
            model = joblib.load(model_path)

        elif event["Model"] == "RF":
            # Load the trained Random Forest model from the joblib file
            model_path = os.path.join(basedir, 'models', 'rf_model.joblib')
            model = joblib.load(model_path)

        elif event["Model"] == "XGBoost":
            # Load the trained XGBoost model from the joblib file
            model_path = os.path.join(basedir, 'models', 'xgb_model.joblib')
            model = joblib.load('model_path')

        # Extract the features from the event dictionary
        features = [
            event['HT']['Mean'],
            event['HT']['STD'],
            event['PPT']['Mean'],
            event['PPT']['STD'],
            event['RRT']['Mean'],
            event['RRT']['STD'],
            event['RPT']['Mean'],
            event['RPT']['STD']
        ]

        # Make a prediction using the loaded model and the extracted features
        prediction = model.predict([features])

        # Return the predicted user
        return prediction[0]
