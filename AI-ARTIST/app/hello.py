# -*- coding:utf-8 -*-
import os # We'll render HTML templates and access data sent by POST

from flask import Flask,send_file, render_template, request, redirect, url_for, send_from_directory,make_response
from werkzeug import secure_filename
import sys
from datetime import timedelta
import new_style

defaultencoding = 'utf-8'

file_list=["no",'result']

if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__,static_url_path='')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    
    uploaded_files = request.files.getlist("file[]")
    filenames = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            suffix=file.filename.rsplit('.', 1)[1]
            filename = secure_filename(file.filename)
            file_list[0] = "raw."+suffix
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file_list[0]))
            filenames.append(filename)
    return render_template('upload.html', filenames=filenames)

@app.route('/uploads/<filename>',methods=['GET'])
def uploaded_file(filename):
    fname = filename.encode('cp936')
    return send_from_directory(app.config['UPLOAD_FOLDER'], fname, mimetype='application/octet-stream')

@app.route('/art',methods=['POST'])
def art():
    if file_list[0]=="no":
        return render_template('static/choose_style.html',origin = "../uploads/"+"temp.jpg")
    else:
        origin="../uploads/"+file_list[0]
        return render_template('static/choose_style.html',origin = origin)

@app.route('/art',methods=['GET'])
def art_get():
    if file_list[0]=="no":
        return render_template('static/choose_style.html',origin = "../uploads/"+"temp.jpg")
    else:
        origin="../uploads/"+file_list[0]
        return render_template('static/choose_style.html',origin = origin) 


@app.route('/style',methods=['POST'])
def style():
    if file_list[0]=="no":
        return render_template('static/choose_style.html',origin = "../uploads/"+"temp.jpg")
    else:
        file_dir=file_list[0]
        origin="../uploads/"+file_list[0]
        new_style.style_interface(file_dir=file_dir)
        return render_template('static/show_style.html',origin = origin)

@app.route('/style',methods=['GET'])
def style_get():
    if file_list[0]=="no":
        return render_template('static/choose_style.html',origin = "../uploads/"+"temp.jpg")
    else:
        origin="../uploads/"+file_list[0]
        return render_template('static/show_style.html',origin = origin)    


if __name__ == '__main__':
    app.run()
