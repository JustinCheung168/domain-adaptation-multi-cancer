from flask import Flask, jsonify, request, render_template, send_file
from PIL import Image
import os
from io import BytesIO

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith('.png'):
        image = Image.open(file).convert('L')  # Convert to grayscale
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    return jsonify({"error": "Invalid file type. Only PNG files are allowed."}), 400


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
