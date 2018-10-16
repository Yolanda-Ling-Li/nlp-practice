from flask import Flask
import json
from flask_environments import Environments
from keras.models import load_model
import tensorflow as tf
from api.similar_score import show_similar_score

app = Flask(__name__, static_folder='static')


env = Environments(app)
env.from_object('config')
graph = None
nlp_model = None

def load_nlp_model():
    global graph
    graph = tf.get_default_graph()
    global nlp_model
    nlp_model = load_model(app.config['NLP_MODEL_FILE'])

@app.route("/eval", methods=['POST'])
def predict_result():
    score = show_similar_score(graph,nlp_model)
    return json.dumps(score)


if __name__ == "__main__":
    load_nlp_model()
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])