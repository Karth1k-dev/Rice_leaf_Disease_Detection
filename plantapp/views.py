'''from django.shortcuts import render
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

    return render(request, 'plantapp/upload.html', {'prediction': prediction})'''

from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import os

# ===============================
# MODEL LOADING (UNCHANGED PATH)
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_model", "model.keras")

model = load_model(MODEL_PATH)

CLASS_NAMES = [
    "Leaf Smut",
    "Bacterial Leaf Blight",
    "Brown Spot"
]

DISEASE_INFO = {
    "Leaf Smut": {
        "description": "Leaf smut is a fungal disease that causes black spots on rice leaves and reduces yield.",
        "pesticide": "Use Carbendazim or Mancozeb. Spray at early stage."
    },
    "Bacterial Leaf Blight": {
        "description": "A bacterial disease causing yellowing and drying of leaves from the tips.",
        "pesticide": "Spray Streptocycline + Copper Oxychloride."
    },
    "Brown Spot": {
        "description": "A fungal disease causing brown circular spots on leaves.",
        "pesticide": "Apply Propiconazole or Hexaconazole."
    }
}


def home(request):
    return render(request, "plantapp/upload.html")


def predict(request):
    if request.method == "POST" and request.FILES.get("image"):
        uploaded_file = request.FILES["image"]

        # ✅ Convert Django UploadedFile → BytesIO
        image_bytes = BytesIO(uploaded_file.read())

        # ✅ Load & preprocess image
        img = image.load_img(image_bytes, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ Prediction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions)) * 100

        result = CLASS_NAMES[class_index]
        description = DISEASE_INFO[result]["description"]
        pesticide = DISEASE_INFO[result]["pesticide"]


        return render(
            request,
            "plantapp/upload.html",
            {
                "result": result,
                "confidence": round(confidence, 2),
                "description": description,
                "pesticide": pesticide
            }
        )
        

    return render(request, "plantapp/upload.html")

