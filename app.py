# pip install flask tensorflow numpy pillow

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = tf.keras.models.load_model("best_model.h5")
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            preds = model.predict(img_array)
            class_index = np.argmax(preds, axis=1)[0]
            prediction = class_names[class_index]
            image_path = filepath

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
