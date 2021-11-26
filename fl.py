from flask import Flask, render_template,redirect,url_for
from flask.globals import request
from forms import someform
from task2_code import predict_num_new_cases as p
from flask_bootstrap import Bootstrap
from app.middleware import PrefixMiddleware
from config import Config

app=Flask(__name__)

# Flask-WTF requires an encryption key - the string can be anything
app.config['SECRET_KEY'] = '4TcUAnNR6523gReAuPVMLJ0nBJ8d9cDY'
app.config.from_object(Config)
app.wsgi_app = PrefixMiddleware(app.wsgi_app, voc=True)

# Flask-Bootstrap requires this line
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Task1')
def Task1():
    
    return render_template('Task 1.html')

@app.route('/Task2', methods=['GET','POST'])
def Task2():
    return render_template('Task 2.html')

@app.route('/pdfview1')
def pdffor1():
    return render_template

@app.route('/Task 2/display')
def display(novax,pop,country):
    predicted=p(novax,pop,country)
    return render_template('display.html',predicted)


@app.route('/bibliography')
def bibliography():
    return render_template('bibliography.html')


'''def functionultimate(no_cases,population):
    #calculate stuff
    return #a string value that can be concatenated into task 2'''

if __name__=="__main__":
    app.run(debug=True)

