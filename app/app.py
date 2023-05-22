from flask import Flask, request
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from app.utils import get_first_file_path
import numpy as np
import os

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "GET":
        return "API is ready to recieve POST requests."

    classification = ""

    image_file = request.files['image']
    os.makedirs("classify", exist_ok=True)
    image_file.save("classify/classify.jpg")

    # get the most recent model
    root_path = os.path.join("models")
    path = get_first_file_path(root_path)[0]
    model = keras.models.load_model(path)

    classify_dataset = image_dataset_from_directory(
        "classify",
        batch_size=1,
        image_size=(180, 320),
        labels=None
    )

    prediction = model.predict(classify_dataset)
    print(prediction[0].tolist())

    classification = convert_label(prediction[0].tolist())

    print(classification)
    return classification


def convert_label(prediction_list: list):
    class_strings = {
        0: "Bed",
        1: "Missing",
        2: "Rug",
        3: "Somewhere in the room",
    }
    prediction = np.argmax(prediction_list)
    return class_strings[prediction]
