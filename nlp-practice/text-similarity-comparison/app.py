from keras.models import load_model
from flask_environments import Environments
from flask import Flask
import tensorflow as tf
import json
import sys

sys.path.append("/api")
from api.similar_score import show_similar_score

from ImageNet_model import create_siamese
input_shape = (224, 224, 3)


app = Flask(__name__, static_folder='static')  #Create a flask object
env = Environments(app)  #config
env.from_object('config')
graph = None
siamese_model = None


def load_siamese_model():
    """Load the trained model."""
    global graph
    graph = tf.get_default_graph()  #Get the default map
    global siamese_model
    siamese_model = load_model(app.config['SIAMESE_MODEL_FILE'])
    global siamese_model_pic
    siamese_model_pic = create_siamese(input_shape)
    siamese_model_pic.load_weights(app.config['SIAMESE_MODEL_PIC_FILE'])


@app.route("/eval", methods=['POST'])
def predict_result():
    """Get the articles and its similar degree."""
    score, the_log = show_similar_score(graph, siamese_model, siamese_model_pic, app.config['REQUIRE_URL'])
    print(json.dumps(the_log))
    return json.dumps(score)  #Convert dict type data to str


if __name__ == "__main__":
    load_siamese_model()
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])
