## Text Similarity Using Siamese Deep Neural Network


Siamese neural network is a class of neural network architectures that contain two or more identical subnetworks. identical here means they have the same configuration with the same parameters and weights. Parameter updating is mirrored across both subnetworks.

It is a keras based implementation of deep siamese Bidirectional LSTM network to capture phrase/sentence similarity using word embeddings.

Below is the architecture description for the same.

![rch_imag](images/arch_image.png)

## Usage

###Training Data Preparation

1. Download the required data from the website

      [Written Composition Corpus of Japanese Learners](http://sakubun.jpn.org/)
 
2. Unzip the data and name the folder "thedata"


###Installation of Mecab Word Segmentation System


1. When you cannot install Mecab through pip, download the installation package from the website
 
    [Mecab Japanese word segmentation system](http://taku910.github.io/mecab/)

2. Install Mecab according to the installation tutorial

    [windows10+py36+MeCab installation](https://blog.csdn.net/ZYXpaidaxing/article/details/81913708)


### Configuration Environment

1. Install python3.6.6
2. Install pycharm
3. Install virtualenv10.0.1

       pip install virtualenv

4. Construct the project directory to install the virtual environment for the project

       virtualenv venv --no-site-packages

5. Start the virtual environment and install the required libraries
    * Start the virtual environment in Linux or Mac
    
          source venv/bin/activate
    
    * Start the virtual environment in Windows
    
          venv\Scripts\activate
    
6. Install the required libraries
    
       pip install -r requirements.txt

7. Operations such as running programs can be performed in a virtual environment 

       python controller.py 
       python ImageNet_utils.py
       python ImageNet_train.py
  
8. Close the virtual environment

       deactivate


###Predict Text/Photo Similarity

1. Run app.py

       python app.py

2. Use postman software to open http://172.29.226.176:5500/eval

3. Set the request to Post, KEY to Content-Type and VALUE to application/form-data in Headers.

4. Set the KEY for Body to article_content, record_content, content_type, record_id and article_id.

5. Enter the base64 encoding of the text/photo in article_content and record_content, enter 'txt' or 'photo' in content_type, set record_id and article_id, and then click Send to send the data.

6. Results will be returned and sent to  http://172.29.226.64:8080/api/score/receive


### Reference resources

1. [Siamese Recurrent Architectures for Learning Sentence Similarity (2016)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195)
2. Inspired from Tensorflow Implementation of  https://github.com/dhwajraj/deep-siamese-text-similarity
3. [Lstm Siamese Text Similarity from Github](https://github.com/amansrivastava17/lstm-siamese-text-similarity)
