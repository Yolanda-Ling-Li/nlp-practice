from input_handler import create_test_data, data_predict
from config import siamese_config
from flask import request
import uuid
import json
import os
import base64
import requests


class ConfiGuration(object):
    """Dump stuff here"""


config = ConfiGuration()
config.max_len = siamese_config['MAX_DOCUMENT_LENGTH']
config.data_dir = siamese_config['DATA_DIR']


now_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
upload_path = os.path.join(now_path, "data//temp")
if os.path.exists(upload_path) is False:
    os.makedirs(upload_path)



def show_similar_score(graph, nlp_model):
    """
    Predict the similar degree between two articles.
    :param graph:
    :param nlp_model:
    :return:
        {'similar_score': preds_score}(dict): preds_score(str) is the he similar degree between two articles
    """
    w2index_file = os.path.join(config.data_dir, "model//w2index.txt")
    file_1 = request.form['article_content']
    file_2 = request.form['record_content']
    content_type = request.form['content_type']
    record_id = request.form['record_id']

    if content_type != 'txt':
        return 'Only support txt text input NOW!'

    decode_file_1 = base64.b64decode(file_1).decode('utf-8')
    decode_file_2 = base64.b64decode(file_2).decode('utf-8')
    file_1_name = 'predict1.txt'
    file_2_name = 'predict2.txt'
    save_data_dir = os.path.join(upload_path, str(uuid.uuid1()))
    os.makedirs(save_data_dir)
    savefile_1 = os.path.join(save_data_dir, file_1_name)
    with open(savefile_1,'w+',encoding='utf-8') as f:
        f.write(decode_file_1)
    savefile_2 = os.path.join(save_data_dir, file_2_name)
    with open(savefile_2,'w+',encoding='utf-8') as f:
        f.write(decode_file_2)
    with open(w2index_file, 'r+', encoding='utf-8') as f:
        w2index = f.read()
    index_dict = json.loads(w2index)

    predict_documents1, predict_documents2 = data_predict(save_data_dir)
    predict_document_pairs = [(x1, x2) for x1, x2 in zip(predict_documents1, predict_documents2)]
    predict_data_x1, predict_data_x2, leaks_predict = create_test_data(index_dict, predict_document_pairs,
                                                                       config.max_len)
    with graph.as_default():
        preds = nlp_model.predict([predict_data_x1, predict_data_x2, leaks_predict], verbose=1).ravel()
    preds_score = str(round(preds[0] * 100, 2))
    preds_result = {'record_id': record_id, 'score': preds_score}
    send_status_code = send_score(preds_result)
    if send_status_code != 200:
        preds_result['send_res']='Fail'
    return preds_result

def send_score(preds_result):
    send_url = 'http://172.29.226.64:8080/api/score/receive'
    req = requests.post(url=send_url,data=preds_result)
    return req.status_code


if __name__ == "__main__":
    res = send_score({'id':1,'score':20})

