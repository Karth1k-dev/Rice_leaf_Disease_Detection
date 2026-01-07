from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, 'model.keras')

# IMPORTANT FIX
model = load_model(MODEL_PATH, compile=False)

CLASS_NAMES = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

def predict(request):
    prediction = None

    if request.method == 'POST' and request.FILES.get('leaf'):
        img_file = request.FILES['leaf']

        img_path = os.path.join(PROJECT_DIR, 'temp.jpg')
        with open(img_path, 'wb+') as f:
            for chunk in img_file.chunks():
                f.write(chunk)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)
        prediction = CLASS_NAMES[np.argmax(result)]

    return render(request, 'plantapp/upload.html', {'prediction': prediction})
