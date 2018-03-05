from flask import Flask, render_template, request
from PIL import Image
from brain_img_processor import BrainData
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, static_url_path='/Flask/static')
UPLOAD_FOLDER = '..\\Flask\\static\\uploads'
IMG_FOLDER = '..\\Flask\\static\\img'
ALLOWED_EXTENSIONS = set(['mha', 'nii', 'png', 'jpg', 'jpeg'])


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
            # run(mha_loc)
            return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/index')
def index():
    return render_template('index.html')


# TO DO figure out how to save slice as jpg or png

# def run(p):
#     brain = BrainData(p)
#     s = brain.get_slice(BrainData.TOP_PROF, 120)
#     img = Image.fromarray(s)
#     img.save(IMG_FOLDER + 'brain_img.jpg', 'RGB')


if __name__ == "__main__":
    app.run(debug=True)
