# from flask import Flask,render_template,request
# # from flask_wtf import FlaskForm
# # from wtforms import FileField
# app=Flask(__name__)

# @app.route("/")
# @app.route("/index.html")
# @app.route("/index")
# def func():
# 	return render_template("index.html")


# ALLOWED_EXTENSIONS = ['mp3','mp4','MP3']
 
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route('/upload-audio', methods=['GET', 'POST'])
# def upload():
#     print("aaaaaaaaaa")
#     if request.method == 'POST':
#             print(request.files.keys())
#             # print(request['data'])
#             print("aagayaaaaaaa")
#             return render_template("success.html")
#     print("dhoooooooom")
#     return {"status": True}

# @app.route('/upload-audio.html', methods=['GET', 'POST'])
# def upload_file():
#     return render_template("upload-audio.html")
                


# if __name__ == '__main__':
#     app.run(debug=True)
