import os, sys

here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, "..", "pymodules"))

from flask import Flask, render_template, request, jsonify
from PIL import Image
from BrainProcessor import BrainProcessor # change how this is imported
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__, static_url_path='/server/static')

UPLOAD_FOLDER = os.path.join('server','static','uploads')
IMG_FOLDER = os.path.join('server','static','img')
ALLOWED_EXTENSIONS = set(['mha', 'nii', 'png', 'jpg', 'jpeg'])

brain_processor = BrainProcessor()

@app.route('/')
def landing():
    return render_template("landing.html")


@app.route('/upload_file', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        file = request.files['chooseFile']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            mha_loc = os.path.join(UPLOAD_FOLDER, filename)
            file.save(mha_loc)
            if brain_processor.load_mri(mha_loc):
                brain_data = brain_processor.get_data()
                data_as_list = brain_data['data'].tolist()
                brain = {
                    'data': data_as_list,
                    'dim': list(brain_data['dim'])
                }
                return render_template('index.html', brain=brain)
            else:
                return render_template('landing.html')
            

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/landing')
def index():
    return render_template('landing.html')


if __name__ == "__main__":
    app.run(debug=True)
