from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
from keras_preprocessing.image import img_to_array


app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("AIGeneratedModel.h5")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        img = request.files['image']
        if img:
            image = Image.open(img)
            image = ImageOps.fit(image, (48, 48), Image.Resampling.LANCZOS)
            img_array = img_to_array(image)
            new_arr = img_array / 255
            test = [new_arr]
            test = np.array(test)
            y = model.predict(test)
            percentage = 0
            if y[0] <= 0.5:
                percentage = 99
            elif y[0] <= 0.6:
                percentage = 83
            elif y[0] <= 0.7:
                percentage = 67
            elif y[0] <= 0.8:
                percentage = 55
            elif y[0] <= 0.9:
                percentage = 49
            elif y[0] <= 1:
                percentage = 19
            else:
                percentage = 3
            return render_template('result.html', percentage=percentage)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)