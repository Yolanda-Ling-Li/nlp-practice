from input_handler import create_test_data, data_predict
from ImageNet_predict import create_predict_data
from config import siamese_config
from flask import request
from PIL import Image
import requests
import base64
import json
import os
import io




class ConfiGuration(object):
    """Dump stuff here"""


config = ConfiGuration()
config.max_len = siamese_config['MAX_DOCUMENT_LENGTH']
config.data_dir = siamese_config['DATA_DIR']


now_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
upload_path = os.path.join(now_path, "data//temp")
if os.path.exists(upload_path) is False:
    os.makedirs(upload_path)



def show_similar_score(graph, siamese_model, siamese_model_pic, send_url):
    """
    Predict the similar degree between two articles.
    Args:
        graph(graph): fundamental graph
        siamese_model: the trained siamese model
        siamese_model_pic: the trained picture model
        send_url(str): url of request address
    Return:
        preds_result(dict): preds_score(str) is the he similar degree between two articles or pictures
        the_log(dict): log of this predict
    """
    w2index_file = os.path.join(config.data_dir, "model//txt//w2index.txt")
    data = json.loads(request.get_data())
    file_1 = data['article_content']
    file_2 = data['record_content']
    content_type = data['content_type']
    record_id = data['record_id']
    article_id = data['article_id']

    if content_type == 'pic':
        # return 'Only support txt text input NOW!'
        decode_image_1 = base64.b64decode(file_1)
        decode_image_2 = base64.b64decode(file_2)

        image_bytes_1 = io.BytesIO(decode_image_1)
        image_bytes_2 = io.BytesIO(decode_image_2)

        image_1 = Image.open(image_bytes_1)
        image_2 = Image.open(image_bytes_2)

        file_1_name = 'article_' + article_id + '.jpg'
        file_2_name = 'record_' + record_id + '.jpg'
        save_data_dir = upload_path + '/pic/' + article_id

        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)

        image_1.save(save_data_dir + '/' + file_1_name)
        image_2.save(save_data_dir + '/' + file_2_name)

        predict_image_pair = create_predict_data(save_data_dir)
        with graph.as_default():
            preds = siamese_model_pic.predict([predict_image_pair[:, 0], predict_image_pair[:, 1]])
        preds = 100 - (preds[0][0] * 10000 + 0.5) // 1 / 100
        preds_score = str(preds)
    elif content_type == 'txt':
        decode_file_1 = base64.b64decode(file_1).decode('utf-8')
        decode_file_2 = base64.b64decode(file_2).decode('utf-8')
        print("file1:" + decode_file_1[0:50])
        print("file2:" + decode_file_2[0:50])
        file_1_name = 'article_' + article_id + '.txt'
        file_2_name = 'record_' + record_id + '.txt'
        save_data_dir = os.path.join(upload_path, 'txt//' + article_id + '//' + record_id)
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        savefile_1 = os.path.join(save_data_dir, file_1_name)
        with open(savefile_1, 'w+', encoding='utf-8') as f:
            f.write(decode_file_1)
        savefile_2 = os.path.join(save_data_dir, file_2_name)
        with open(savefile_2, 'w+', encoding='utf-8') as f:
            f.write(decode_file_2)
        with open(w2index_file, 'r+', encoding='utf-8') as f:
            w2index = f.read()
        index_dict = json.loads(w2index)

        predict_documents1, predict_documents2 = data_predict(save_data_dir)
        predict_document_pairs = [(x1, x2) for x1, x2 in zip(predict_documents1, predict_documents2)]
        predict_data_x1, predict_data_x2, leaks_predict = create_test_data(index_dict, predict_document_pairs,
                                                                           config.max_len)
        with graph.as_default():
            preds = siamese_model.predict([predict_data_x1, predict_data_x2, leaks_predict], verbose=1).ravel()
        preds_score = str(round(preds[0] * 100, 2))
    else:
        warning_str = '###Wrong Input: Only support txt or pic input!'
        return warning_str, warning_str

    the_log = {'article_id': article_id, 'record_id': record_id, 'score': preds_score}
    preds_result = {'record_id': record_id, 'score': preds_score}
    send_status_code = send_score(preds_result, send_url)
    if send_status_code != 200:
        preds_result['send_res'] = 'Fail'
    return preds_result, the_log


def send_score(preds_result, send_url):
    """
    Send post request.
    Args:
         preds_result(dict): cotent to be sent
         send_url(str): url of request address
    Return:
        req.status_code(int): status code of requestion
    """
    req = requests.post(url=send_url, data=str(preds_result))
    return req.status_code


if __name__ == "__main__":
    res = send_score({'id': 1, 'score': 20})
