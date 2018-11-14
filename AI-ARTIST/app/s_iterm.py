# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:35:11 2018

@author: cc
"""

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('choose.html')

if __name__ == '__main__':
    app.debug=True
    app.run()
    
