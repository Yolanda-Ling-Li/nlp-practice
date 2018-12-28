Neural Network for Emotional Classification of IMDB Data Sets
---
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

        python train.py -m 1  
        #1--embedding+cnn+lstm  2--word2vec+cnn+lstm 
        #3--predict through model1  4--predict through model2
  
8. Close the virtual environment

        deactivate

PS. The prediction module has not been debugged yet and couldn't be executed


### Data Sources
[Download aclImdb_v1.tar](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)



### Modules and Functions
1. Data processing module：dataprocessing.py

        def rm_tags(text) #remove tags
        def read_files(filetype,data_dir) #read files, generate tag arrays and article lists
        def embeding_load_data(max_len,vocabulary_size=2000) #generate training set and verification set of model1
        def w2v_load_data(data_dir) #generate training set and verification set of model2
        
2. Training model module：train.py

        class Config() #parameter configuration
        def train_embedding(config) #train model1 and save the best model
        def train_w2v(config) #train model2 and save the best model
        
3. Generating model module：model.py
   
        def Embedding_CNN_LSTM_Model(config) #model1
        def create_dictionaries(vocab_dim,max_len,model=None, combine=None):  #Create a dictionary of words
        Word2Vec_Model(config,combine)  #Create a dictionary of words through Word2Vec


4. Predict module：predict.py （not debug yet）

        def predict_save(data_dir, x_predict, y_predict)  #save model
        def predict_load(data_dir) #load model
        def predict(config) #predict
   
        

###Input and Output Documents
1. The system will automatically download data from the website. After the download and decompression, the following documents will appear.

    * aclImdb_v1.tar
    * aclImdb
 
2. Log files are generated during system operation
 
     * logs
     * events.out.tfevents.1538191499.ai-1070

3. The model and parameters are stored after the system runs.
 
     * Model1.hdf5
     * Model2.hdf5
     * x.npy
     * y.npy
     * Word2vec_model.pkl
     
     
