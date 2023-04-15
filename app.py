from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
import os

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
    print("aaaaaaaaaa")
    if request.method == 'POST':
        print(request.files.keys())
        print(request.files["audio"])
        audio_file = request.files["audio"]
        audio_file_path = os.path.join("audio_data/", audio_file.filename.rsplit('\\')[-1])
        print(audio_file.content_length)
        print(audio_file)
        audio_file.save(audio_file_path)
        mel_data_file_path = model.preprocess_data(audio_file_path)
        model.model_predict(mel_data_file_path)
        
        print("aagayaaaaaaa")
        # handleUploadFile(request.files["audio"])
        return {"status": True, "result": True}
    print("dhoooooooom")
    return {"status": True}

@app.route('/upload-audio.html', methods=['GET', 'POST'])
def upload_file():
    return render_template("upload-audio.html")
                


if __name__ == '__main__':
    app.run(debug=True)
