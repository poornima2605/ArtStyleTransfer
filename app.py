from flask import Flask, render_template, request, send_from_directory
import os
from style_transfer import perform_style_transfer  # Import your function

app = Flask(__name__)

# Set static folder for storing processed images
UPLOAD_FOLDER = 'static/ProcessedImage'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # Get content and style images from the form
    content_image = request.files['content']
    style_image = request.files['style']

    # Save uploaded files
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg')
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style.jpg')
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed.jpg')

    content_image.save(content_path)
    style_image.save(style_path)

    # Perform style transfer
    perform_style_transfer(content_path, style_path, processed_path)

    # Return the processed image to the user
    return render_template('result.html',
                           content_image='ProcessedImage/content.jpg',
                           style_image='ProcessedImage/style.jpg',
                           processed_image='ProcessedImage/processed.jpg')

@app.route('/static/ProcessedImage/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
