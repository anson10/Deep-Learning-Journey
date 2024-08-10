import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Classes of traffic signs
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Vehicle > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing vehicle > 3.5 tons' }

def image_preprocessing(img_path):
    model = load_model('./GTSRB.h5')
    image = Image.open(img_path)
    
    # Resize the image to (32, 32)
    image = image.resize((32, 32))
    
    # Convert image to an array
    image_array = np.array(image)
    
    # Check if the image is grayscale or RGB and convert to 3 channels if needed
    if image_array.ndim == 2:
        # If the image is grayscale (2D), convert it to a 3D array by adding a channel dimension
        image_array = np.stack((image_array,)*3, axis=-1)
    
    # Expand dimensions to match model input shape (1, 32, 32, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Normalize the pixel values to the range [0, 1]
    image_array = image_array / 255.0
    
    # Predict the class
    y_pred = np.argmax(model.predict(image_array), axis=-1)
    
    return y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            f = request.files['file']
            filename = secure_filename(f.filename)
            
            # Save the file in the static/uploads directory
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path)
            
            # Perform image processing and prediction
            result = image_preprocessing(file_path)
            s = [str(i) for i in result]
            a = int("".join(s))
            prediction = "🚦 System classified the uploaded sign as: " + classes[a] + " 🚦"
            
            # Pass only the filename to the template
            return render_template('result.html', image_url='uploads/' + filename, prediction=prediction)
        except Exception as e:
            print(f"Error: {e}")
            return "An error occurred while processing the image. Please try again."
    
    return None

if __name__ == "__main__":
    app.run(debug=True)
