## 项目一：问答摘要与推理

###   1.项目简介

 项目源于  **百度PaddlePaddle AI 产业应用赛——汽车大师问答摘要与推理**

###   2.数据集简介

​数据产生的场景是这样的，一些车主在自己的爱车遇到问题时，在汽车大师App上发起提问，专业技师会根据问题（Problem）和用户进行一段对话（Conversation），从而帮助用户解决问题，最后技师根据问题和对话生成一个报告(Report)
  ![image](https://github.com/mingdaoyang/workshop/blob/master/imges/20200430153435.jpg)

​        **整个数据集共有82000多行数据。**

​        **数据预处理过程中只用到了“Probem+Conversation+Report”**

### 3.数据预处理

- 主要使用了开源的**jieba**分词和**pandas**包（后面词向量训练用到了gensim）；
- 简单的清洗，主要去除一些无效字符，如[图片]等，stopwords只包含了英文停用词；
- 将数据集划分为四个部分，即train_x,train_y,test_x,test_y(需要预测生成，初始为空)；
- 第一周作业要求合并前三个部分后，生成一个word+index的vocab。
