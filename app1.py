from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your semantic segmentation model here
model_path = 'path/to/your/landcover_25_epochs_RESNET_backbone_batch16.hdf5'
model = load_model(model_path)
model._make_predict_function()  # Necessary for using the model in a Flask app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # Assuming you have a function to perform semantic segmentation
    result = perform_semantic_segmentation(file)

    return 'File uploaded successfully and processed by the model'

def perform_semantic_segmentation(image_file):
    # Resize the image to match your model's expected sizing (height, width)
    img = image.load_img(image_file, target_size=(224, 224))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand the dimensions to match the model's expected input dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image (normalize values)
    img_array = img_array / 255.0

    # Make predictions
    predictions = model.predict(img_array)

    # Assuming your model outputs segmentation masks
    # You may need to post-process the predictions based on your model's output
    # For simplicity, let's assume predictions is the segmentation mask
    result = predictions[0]  # Replace with your actual post-processing logic

    return result

if __name__ == '__main__':
    app.run(debug=True)
