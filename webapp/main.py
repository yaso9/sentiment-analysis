import tensorflow as tf
from useNetwork import use_neural_network
from flask import Flask
from flask import render_template
from flask import url_for
from flask import request
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    with tf.Graph().as_default():
        return render_template('index.html', happy=(True if use_neural_network(request.form['text']) == 2 else False))


with app.test_request_context():
    url_for('static', filename='kube.min.css')
    url_for('static', filename='style.css')
