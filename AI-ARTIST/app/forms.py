# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:36:01 2018

@author: cc
"""
from __future__ import absolute_import
from flask_wtf import FlaskForm  as Form
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired
#验证
class Iterms(Form):
    iterm = StringField('iterm',validators=[DataRequired()])
