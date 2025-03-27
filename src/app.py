from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

import cv2
from utils import get_all_object, get_bounding_box_from_point
from function import multi_scale_template_matching

app = Flask(__name__) 

upload_folder = os.path.join('static', 'images')

app.config['UPLOAD_FOLDER'] = upload_folder
app.config['ALL_OBJECT'] = None
app.config['IMAGE_PATH'] = None


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def render_upload():
    return render_template('upload_image.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Create uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        # Save the file
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)

        app.config['IMAGE_PATH'] = img_path
        image = cv2.imread(img_path)
        app.config['ALL_OBJECT'] = get_all_object(image)
        
        return render_template('render_image.html', img=img_path)
    
    return 'File type not allowed', 400

@app.route('/coordinates', methods=['POST'])
def handle_coordinates():
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')
    # Do something with the coordinates
    all_objects = app.config['ALL_OBJECT']
    image = cv2.imread(app.config['IMAGE_PATH'])
    matches = multi_scale_template_matching(
        image,
        (x, y),
        all_objects[-1], #The target
        all_objects,
        scale_range=(0.5, 1),  # Try scales from 0.2x to 1.5x
        scale_steps=20,          # Number of different scales to try
        threshold=0.3         # Confidence threshold
    )
    source_bbox = get_bounding_box_from_point(image, (x, y), app.config['ALL_OBJECT'])
    x,y,w,h = source_bbox
    source_bbox = {
        'x': x,
        'y': y,
        'width': w,
        'height': h
    }
    target_bbox = matches[0]
    target_bbox.pop('scale')
    target_bbox.pop('confidence')

    return jsonify([source_bbox, target_bbox])