imdb 数据集的情感分类的神经网络
---
### 配置环境

1. 安装python3.6.6
2. 安装pycharm
3. 安装virtualenv10.0.1

        pip install virtualenv

4. 构造项目目录，为项目安装虚拟环境

        virtualenv venv --no-site-packages

5. 启动虚拟环境，安装所需类库
    * Linux或Mac启用虚拟环境
    
            source venv/bin/activate
    
    * Windows启用虚拟环境
    
            venv\Scripts\activate
    
    * 安装所需类库
    
            pip install TensorFlow
            pip install Keras
            pip install numpy
            pip install gensim
            pip install sklearn

6. 在虚拟环境中可以进行运行脚本等操作  

        python train.py -m 1  
        #1--embedding+cnn+lstm  2--word2vec+cnn+lstm 
        #3--predict through model1  4--predict through model2
  
7. 离开虚拟环境

        deactivate

P.S. 代码中涉及从外网下载数据，需要开翻墙VPN

PPS. 预测模块还没有调试完毕，应该无法执行


### 各模块及函数
1. 数据处理模块：dataprocessing.py

        def rm_tags(text) #去标签
        def read_files(filetype,data_dir) #读文件，生成标签array和文章list
        def w2v_load_data(data_dir) #生成训练集验证集并返回
        def embeding_load_data(max_len,vocabulary_size=2000) #筛去过多词语，截断过长句子，生成训练集验证集并返回
        
2. 训练模型模块：train.py

        class Config() #所有参数
        def train_embedding(config) #model1训练并生成对应h5py
        def train_w2v(config) #model2训练并生成对应h5py
        
3. 生成模型模块：model.py
   
        def Embedding_CNN_LSTM_Model(config) #生成模型1
        def create_dictionaries(vocab_dim,max_len,model=None, combine=None):
        #创建词语字典，并返回每个词语的索引，词向量，以及每个文本所对应的词语索引矩阵
        Word2Vec_Model(config,combine)
        #创建并训练Word2Vec，#创建词语字典，并返回词典大小，权重矩阵以及每个词语索引矩阵
        def W2v_CNN_LSTM_Model(config) #生成模型2

4. 预测模块：predict.py （未调试完毕）

        def predict_save(data_dir, x_predict, y_predict)  #存储模型
        def predict_load(data_dir) #加载模型
        def predict(config) #预测
   
        

### 输入输出文档
1. 系统会自动从网站上下载数据，下载解压完成后会有以下文档

    * aclImdb_v1.tar
    * aclImdb
 
2. 系统运行过程中会生成日志文件
 
     * logs
     * events.out.tfevents.1538191499.ai-1070

3. 系统运行完后会存储模型和参数
 
     * Model1.hdf5
     * Model1.hdf5
     * Word2vec_model.pkl
