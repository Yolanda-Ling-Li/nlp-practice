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
    
            venv/Scripts/activate
    
    * 安装所需类库
    
            pip install TensorFlow
            pip install Keras

6. 在虚拟环境中可以进行运行脚本等操作    
7. 离开虚拟环境

        deactivate

P.S. windows用户可以使用`anaconda`的环境来取代`virtualenv`

