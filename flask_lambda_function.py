import functions

from flask import Flask, request, jsonify
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    # Bellow code should be uncommented when running in AWS:Lambda
    # above should be commented as well as flask import + app definition
# def lambda_handler(event, context):
    """
     Lambda handler: When a request hits the API gateway linked to this lambda_function this is the function that gets
     called.
     The request data is passed as the event variable which is a dictionary object, in this case it the json of
     the POST request from which we extract the body details
    """

    # Parses the details from the POST request: extracts model and input data
    # Based on model it imports the trained model from local
    # Outputs the predicted user based on input data
    try:
        prediction = functions.predict_user(request.get_json())
        # Below code should be uncommented when running from AWS, above should be commented.
        # prediction = functions.predict_user(event)
        return jsonify({'statuscode': 200,
                        'status': 'success',
                        'predicted user': str(prediction)
                        })
    except Exception as e:
        return jsonify({'statuscode': 400,
                        'status': 'error',
                        'message': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
