import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the trained model
        model_path = os.path.join("model", "model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        model = load_model(model_path)

        # Load and preprocess the image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
        test_image = test_image / 255.0  # Normalize as done during training

        # Predict
        probabilities = model.predict(test_image)[0]
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])

        # Print for debugging
        print(f"Probabilities: {probabilities}")
        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

        prediction_label = 'Tumor' if predicted_class == 1 else 'Normal'

        return [{
            "image": prediction_label,
            "confidence": confidence
        }]
