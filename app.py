from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from pathlib import Path
from src.cancer_detection.utils.common import decodeImage
from src.cancer_detection.pipeline.stage_03_prediction import PredictionPipeline
from src.cancer_detection.config.configuration import ConfigurationManager
from src.cancer_detection.components.data_module import ImageTransform

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        """
        Initializes the ClientApp.
        """
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """
    Home route to render the index.html page.
    """
    return render_template('index.html')

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    """
    Training route to initiate model training using DVC.
    """
    os.system("python main.py")
    os.system("dvc repro")
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    """
    Prediction route to receive an image, make predictions, and return the result.
    """
    image = request.json['image']
    decodeImage(image, clApp.filename)
    path_to_image = os.path.join(os.getcwd(), clApp.filename)

    try:
        training_config = ConfigurationManager().get_training_config()
        inference_config = ConfigurationManager().get_inference_config()
        image_transformation_pipeline = ImageTransform(training_config.params_image_size[0])
        result = clApp.classifier._predict(training_config, inference_config, image_transformation_pipeline, path_to_image)
        if result['image'] == "Normal":
            return jsonify(["Normal"])
        else:
            return jsonify(["Cancerous"])
        # return jsonify(result)
    except:
        return jsonify(["Something went wrong!!"])

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)  # For AWS server
