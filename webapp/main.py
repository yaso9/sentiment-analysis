from flask import Flask
from flask import render_template
from flask import url_for
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', data='hello, world')

with app.test_request_context():
    url_for('static', filename='kube.min.css')
    url_for('static', filename='style.css')
