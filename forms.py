from flask import Flask, render_template, redirect, url_for

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import DataRequired



class someform(FlaskForm):
    novax=IntegerField('Enter the number of vaccinations', validators=[DataRequired()])
    pop=IntegerField('Enter the population', validators=[DataRequired()])
    country=StringField('Enter the country', validators=[DataRequired()])
    submit = SubmitField('Submit')
