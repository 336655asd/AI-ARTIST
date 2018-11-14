# -*- coding:utf-8 -*-
from __future__ import absolute_import
from app import app
import flask
from flask import render_template

from .forms import Iterms
import os
import sys
import codecs
from flask import request, send_from_directory
from werkzeug import secure_filename

from datetime import timedelta
from app import new_style
from app import new_poem
from app.color import plot
#from app import new_music

defaultencoding = 'utf-8'

#相关参数
file_list=["no",'result']
style_iterm=[0]
style_dir = "static/image/"
style_list = ["starry.jpg","cubist.jpg","feathers.jpg","mosaic.jpg","udnie.jpg","wave.jpg"]
word1=["我要开始作诗了!"]
word2=["~~~"]
word3=["~~~"]
sth=[]
poem_start=[]
poem_word=[]
tojs=[0]
#os.system('./music.sh')

if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = 'app/static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    #new_poem.poem_interface()
    #new_music.music_interface()
    return render_template('mix1/home.html')

@app.route('/uppic')
def uppic():
    print style_iterm[0]
    return render_template('uppic.html')
    
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
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
            filenames.append(filename)
    return render_template('choose_style.html',origin = "static/uploads/"+"raw.jpg", style = style_dir+style_list[style_iterm[0]])

@app.route('/uploads/<filename>',methods=['GET'])
def uploaded_file(filename):
    fname = filename.encode('cp936')
    return send_from_directory(app.config['UPLOAD_FOLDER'], fname, mimetype='application/octet-stream')

@app.route('/art',methods=['POST'])
def art():
    if file_list[0]=="no":
        return render_template('choose_style.html',origin = "static/uploads/"+"raw.jpg", style = style_dir+style_list[style_iterm[0]])
    else:
        origin="static/uploads/"+file_list[0]
        return render_template('choose_style.html',origin = origin,style = style_dir+style_list[style_iterm[0]])

@app.route('/art',methods=['GET'])
def art_get():
    if file_list[0]=="no":
        return render_template('choose_style.html',origin = "static/uploads/"+"raw.jpg", style = style_dir+style_list[style_iterm[0]])
    else:
        origin="static/uploads/"+file_list[0]
        return render_template('choose_style.html',origin = origin,style = style_dir+style_list[style_iterm[0]]) 


@app.route('/style',methods=['POST'])
def style():
    if file_list[0]=="no":
        return render_template('choose_style.html',origin = "static/uploads/"+"raw.jpg", style = style_dir+style_list[style_iterm[0]])
    else:
        file_dir=file_list[0]
        origin="static/uploads/"+file_list[0]
        new_style.style_interface(file_dir=file_dir,style_iterm=style_iterm[0])
        return render_template('show_style.html',origin = origin,style = style_dir+style_list[style_iterm[0]])

@app.route('/style',methods=['GET'])
def style_get():
    if file_list[0]=="no":
        return render_template('choose_style.html',origin = "static/uploads/"+"raw.jpg", style = style_dir+style_list[style_iterm[0]])
    else:
        origin="static/uploads/"+file_list[0]
        return render_template('show_style.html',origin = origin,style = style_dir+style_list[style_iterm[0]])

# index view function suppressed for brevity

@app.route('/select', methods = ['GET', 'POST'])
def select():
    form = Iterms()
    print(form.iterm.data)
    if form.validate_on_submit():
        print('select iterm =' + form.iterm.data)
        style_iterm[0]=int(form.iterm.data)
        return flask.redirect('/art')
    return render_template('choose.html',
        form = form,
        providers = app.config['OPENID_PROVIDERS'])
        
@app.route('/poem')
def poem():
    #new_poem.poem_interface()
    #new_music.music_interface()
    word1[0]="我要开始作诗了!"
    word2[0]="~~~"
    return render_template('poem.html',word1=word1[0],word2=word2[0])

@app.route('/detect',methods=['POST'])
def detect():
    try:
        new_poem.darknet_interface()
    except:
        word1[0]='我好像出了些问题'
    word1[0]="瞧瞧我发现了什么!"
    with codecs.open("app/sth.txt",'r','utf-8') as f:
        lines=f.read()
        sth=lines.split(',')
    word2[0]='~~'.join(sth)
    ner=[]
    print("**************")
    print (len(sth),sth)
    if len(sth)==1 and sth[0]==u'':
        word1[0]="很遗憾,我还没能力识别这幅图"
        word2[0]="不如我以'君'为题即兴做一首可否呢"
        poem_start.append(u'君')
        return render_template('poem.html',word1=word1[0],word2=word2[0],rect = "static/uploads/"+"rect.jpg")

    for iterm in sth:
        ner.append(iterm[-1])
    try:
        poem_start.append(ner[0])
        poem_start.append(ner[1])
    except:
        poem_start.append(u'君')
    if u'人' in poem_start:
        poem_start[0]=u'君'
    return render_template('poem.html',word1=word1[0],word2=word2[0],rect = "static/uploads/"+"rect.jpg")

@app.route('/poetry',methods=['POST'])
def poetry():
    try:
        new_poem.poem_interface(poem_start[0])
    except:
        return render_template('poem.html',word1="您还没有检测物体哦",word2="点下detect键开始检测吧!")
    with codecs.open('app/poem.txt','r','utf-8') as f:
        words=f.read()
        for i in range(4):
            poem_word.append(words[6*i:6*i+6])
    return render_template('poem.html',word1=poem_word[0],word2=poem_word[1],word3=poem_word[2],word4=poem_word[3],rect = "static/uploads/"+"rect.jpg")

@app.route('/mix',methods=['POST'])
def mix():
    new_poem.mix_interface()
    if len(poem_word)<4:
        return render_template('poem.html',word1="您还没有作诗哦",word2="点下poetry键开始作诗吧!")
    else:
        return render_template('poem.html',word1=poem_word[0],word2=poem_word[1],word3=poem_word[2],word4=poem_word[3],rect = "static/art/res_poem.jpg")


@app.route('/music-ana')
def analyse():
    try:
        judge=plot.plot("app/static/uploads/raw.jpg")
    except:
        return render_template('music/analyse.html',word1="请重新上传照片哦")
    word1[0]="我猜这是"
    
    if judge==True:
        word2[0]="暖色调~"
        tojs[0]=1
    else:
        word2[0]="冷色调!"
        tojs[0]=0
    return render_template('music/analyse.html',word1=word1[0],word2=word2[0])

@app.route('/music')
def music():
    #plot.plot("app/static/uploads/raw.jpg")
    return render_template('music/index.html',coh=tojs[0])




