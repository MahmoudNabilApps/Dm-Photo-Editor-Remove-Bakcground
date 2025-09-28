import os
import numpy as np
import cv2
import requests
import zipfile
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import onnxruntime
from PIL import Image
import io
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

MODEL_URL = "https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.onnx"
MODEL_PATH = "isnetis.onnx"
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


class BackgroundRemover:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None

    def download_model(self):
        if os.path.exists(MODEL_PATH):
            logger.info("Model already exists, skipping download")
            return True

        try:
            logger.info("Downloading model...")
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()

            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info("Model downloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False

    def load_model(self):
        try:
            providers = ['CPUExecutionProvider']
            if onnxruntime.get_device() == 'GPU':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

            self.session = onnxruntime.InferenceSession(MODEL_PATH, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def preprocess_image(self, image):
        original_size = image.size
        image = image.convert('RGB')
        image = image.resize((1024, 1024))
        image_array = np.array(image).astype(np.float32)
        image_array = image_array / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        return image_array, original_size

    def postprocess_mask(self, mask, original_size):
        mask = mask.squeeze()
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask, mode='L')
        mask = mask.resize(original_size, Image.LANCZOS)
        return mask

    def remove_background(self, image_path):
        try:
            image = Image.open(image_path)
            preprocessed, original_size = self.preprocess_image(image)

            mask = self.session.run([self.output_name], {self.input_name: preprocessed})[0]
            mask = self.postprocess_mask(mask, original_size)

            image = image.convert('RGBA')
            image.putalpha(mask)

            return image
        except Exception as e:
            logger.error(f"Failed to remove background: {e}")
            return None


background_remover = BackgroundRemover()

@app.route('/remove-background', methods=['POST'])
def remove_background_endpoint():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.content_type.startswith('image/'):
            return jsonify({'error': 'File must be an image'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result = background_remover.remove_background(filepath)

        if result is None:
            return jsonify({'error': 'Failed to process image'}), 500

        result_path = os.path.join(RESULT_FOLDER, f"result_{filename.split('.')[0]}.png")
        result.save(result_path, 'PNG')

        os.remove(filepath)

        return send_file(result_path, mimetype='image/png', as_attachment=False)

    except Exception as e:
        logger.error(f"Error in remove_background_endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if not background_remover.download_model():
        exit(1)

    if not background_remover.load_model():
        exit(1)

    app.run(host='0.0.0.0', port=5000)
