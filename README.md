# KeystrokeDynamics
This repository focuses on the use of keystroke dynamics as a behavioral biometric to build machine learning models for user recognition. 


Hi!

This directory is used to host the source code that builds 3 ML models in order to predict an user based on keystrokes.

Please note that all necessary libraries are in the [requirements.txt](https://github.com/BogdanAlinTudorache/KeystrokeDynamics/blob/main/requirements.txt) file.

You can install them using pip/pip3 command:
`pip3 install -r requirements.txt`

# Training the models
As original **input** we use the: [keystroke.csv](https://github.com/BogdanAlinTudorache/KeystrokeDynamics/blob/main/keystroke.csv)

The .csv holds the data required by the [keystrokes_build_ml.py](https://github.com/BogdanAlinTudorache/KeystrokeDynamics/blob/main/keystrokes_build_ml.py)/[.ipynb](https://github.com/BogdanAlinTudorache/KeystrokeDynamics/blob/main/keystrokes_build_ml.ipynb) to train the 3 MLs(SVM, FG, XGBoost)

The **output** of the python file is 3 models (.joblib), they can be found in the /models directory.

# Predicting the user (_to be made public soon_)
* To simulate the AWS Lambda, I've used flask.

* I've used IntelliJ for this, but you can use any other IDE.

* We now run the flask_lambda_function.py file in the IDE.

* This starts a flask app, locally on : http://127.0.0.1:5000

In order to predict the user we must do a POST in Postman to that URL, having a body with this format.
```
{
    "Model": "RF",
   "HT": {
        "Mean": 48.43,
        "STD": 23.34
    },
    "PPT": {
        "Mean": 120.43,
        "STD": 37.41
    },
    "RRT": {
        "Mean": 124.43,
        "STD": 45.34
    },
    "RPT": {
        "Mean": 132.56,
        "STD": 47.12
    }
}
```
