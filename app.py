from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
import os
import shutil

import model
# from flask_wtf import FlaskForm
# from wtforms import FileField
app=Flask(__name__)

@app.route("/")
@app.route("/index.html")
@app.route("/index")
def func():
	return render_template("index.html")


ALLOWED_EXTENSIONS = ['mp3','mp4','MP3']
 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handleUploadFile(f):  
    with open("audio_data/" + f.name, "wb+") as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)

@app.route('/upload-audio', methods=['POST'])
def upload():
    curr_dir = os.getcwd()
    if request.method == 'POST':
        audio_file = request.files["audio"]
        audio_file_path = os.path.join("audio_data/", audio_file.filename.rsplit('\\')[-1])
        audio_file.save(audio_file_path)
        os.makedirs("./mel_data", exist_ok=True)
        mel_data_file_path_list = model.preprocess_data(audio_file_path)
        predict = model.model_predict(mel_data_file_path_list)
        try:
            shutil.rmtree(os.path.join(curr_dir, "mel_data"), ignore_errors=True)
            print("Deleted mel_data.")
        except:
            print("Could not delete mel_data.")
            return {"status": False}
        # handleUploadFile(request.files["audio"])
        if predict:
            print("Distress Detected.")
            return {"status": True, "result": True}
        else:
            print("Distress not detected.")
            return {"status": True, "result": False}
        
    return {"status": False}

@app.route('/upload-audio.html', methods=['GET', 'POST'])
def upload_file():
    return render_template("upload-audio.html")
                


if __name__ == '__main__':
    app.run(debug=True)
