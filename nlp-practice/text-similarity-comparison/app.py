from api.similar_score import show_similar_score
from keras.models import load_model
from flask_environments import Environments
from flask import Flask
import tensorflow as tf
import json


app = Flask(__name__, static_folder='static')  #创建flask对象


env = Environments(app)  #配置
env.from_object('config')
graph = None
nlp_model = None


def load_siamese_model():
    """Load the trained model."""
    global graph
    graph = tf.get_default_graph()  #获得默认图
    global nlp_model
    nlp_model = load_model(app.config['NLP_MODEL_FILE'])


@app.route("/eval", methods=['POST'])  #限制url请求方式，post参数获取是通过request.form['传进来的参数']取到
def predict_result():
    """Get the articles and its similar degree."""
    score = show_similar_score(graph, nlp_model)
    return json.dumps(score)  #将dict类型的数据转成str


if __name__ == "__main__":
    load_siamese_model()
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])
