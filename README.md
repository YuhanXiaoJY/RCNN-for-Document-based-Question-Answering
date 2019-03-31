### Recurrent Convolutional Neural Network  for Document-based Question Answering

**Yuhan Xiao**

------



#### 1. 编译/运行环境  

- 开发及测试均在windows下进行  
- tensorflow-gpu 1.12.0  
- python 3.6.6  
- 用到的库：jieba, tqdm, gensium, numpy



#### 2. 数据说明  

- rawdata中的train-set.data和validation-set.data    

- 其它数据：  

  - 来源：科学空间（http://spaces.ac.cn）

    词向量说明：

    - 由800万篇来自各个微信公众号平台的文章所训练出来
    - 词向量使用Gensim训练，模型为Skip-Gram + Huffman Softmax，维度为256，窗口大小是5，迭代次数为10次

    大小：约1G



#### 3. 代码文件说明  

共6个python文件：

- preprocess.py：对项目给的数据集（train-set.data, validation-set.data）进行数据预处理，将每一行都分好词，且记录相应的问题编号和答案编号。结果写入data/preprocess_data文件夹  
- word2vec.py：将train-set.data和validation-set.data作为语料训练词向量，使用gensium模型，词向量维数为100，结果存入data/embedding文件夹  
- data_util.py：集成了与数据处理相关的函数（这些函数要基于预处理好的数据实现），包括载入词向量，建立单词与词向量矩阵的索引、给每个query配对一个（true answer, false answer）对等等  
- BiLSTM.py：BiLSTM的结构与loss计算  
- RCNN_train.py：进行训练和效果评估，评估结果写入代码同文件夹中的res文件夹中  
- module_test.py：用于程序编写过程中的debug    

运行说明：

- 首先运行preprocess.py进行数据预处理（需要建立文件夹data/preprocess_data，输入的原始数据应放在rawdata文件夹中，停用词放到data/stopwords文件夹中）
- 词向量应事先放入文件夹data/embedding/word2vec中，采用上面提到的微信文章所训出的词向量  
- 运行RCNN_train.py的main函数即可  



#### 4. 系统架构/关键技术  

- 以双向LSTM为主，但在LSTM的基础上做了一点改进。这一改进是从论文```Recurrent convolutional neural networks for text classification```中得到启发的，如下图：

![](http://ww1.sinaimg.cn/mw690/0071tMo1ly1fyfgvryk3ej30t10ca0v2.jpg)

- 简略地说，就是在双向RNN（在项目中，将原始RNN替换为双向LSTM）的输出结果上再套一层max-pooling layer。从CNN的角度来看，即卷积层为BiLSTM，然后经过max-pooling layer，最后达到输出层，得出对(query, answer) pair的score  
- 对于BiLSTM而言，训练时的输入是三元组（query, true answer, false answer）*batch size。由于true answer普遍要少一点，因此，首先要总结每个query所对应的true answer集合，然后对每个（query, false answer）二元组，在true answer集合中随机取一个来配对形成（query, true answer, false answer）三元组。max-pooling layer的输入则是BiLSTM的输出，激活函数为tanh  
- 采用SGD进行训练  



#### 5. 使用的方法  

在以上网络架构的基础上，使用max-margin loss来计算损失：

- 经过max-pooling layer之后，得到了query, true answer, false answer分别对应的结果（记为q, t, f），这三个结果都是向量形式   

- 计算q和t的cosine similarity（记为$$cos_t$$）以及q和f的cosine similarity（记为$$cos_f$$）：  
  $$
  cos(q,t) = \frac {\vec{q} \cdot \vec{t}} {|q|\cdot|t|}
  $$

  $$
  cos(q,f) = \frac {\vec{q} \cdot \vec{f}} {|q|\cdot|f|}
  $$





- 使用max-margin计算loss：
  $$
  max-margin_{loss} = max(0, margin - cos_t + cos_f)
  $$
  margin设为0.1  

- 测试时的输入为（query, answer）二元组，模型为该二元组打分，打分公式为：
  $$
  score = cos(query, answer)
  $$
  也即query和answer的余弦相似度



进行训练：  

- 目标：最小化max-margin loss  
- 采用SGD进行训练，16个epoch，训完前8个epoch后，学习率衰减40%，batch size为64    



#### 6. 在验证集上的MAP, MRR结果  

- BiLSTM 的hidden size:100,  dropout: 0.5,  forget biase: 1。词向量：基于项目给的数据集所训出的词向量，维数100（这个参数下只训了10个epoch）    

  | epoch | MAP                   | MRR                   |
  | ----- | --------------------- | --------------------- |
  | 0     | 0.638343766152204     | 0.640327537530481     |
  | 3     | **0.714369153971196** | **0.715946720569638** |
  | 6     | 0.711034363292755     | 0.712552274812597     |
  | 9     | 0.709558775817833     | 0.710548387703324     |

- hidden size:100,  dropout: 0.5, forget biase: 1。词向量：基于微信文章所得到的词向量，维数256  

  | epoch | MAP                   | MRR                   |
  | ----- | --------------------- | --------------------- |
  | 0     | 0.689849251243349     | 0.691169851983792     |
  | 3     | 0.750057080319555     | 0.751860574102684     |
  | 6     | 0.748193524065388     | 0.749732674448657     |
  | 9     | **0.750350416824997** | **0.752339680850078** |
  | 12    | 0.74946854613519      | 0.751495861927022     |
  | 15    | 0.748345972494942     | 0.748013944602643     |

  换了词向量后，效果涨了5个点，可见合适的词向量还是很重要的   

- hidden size:100,  dropout: 0.5, forget biase: 0.8。词向量：基于微信文章所得到的词向量，维数256  

  | epoch | MAP                   | MRR                   |
  | ----- | --------------------- | --------------------- |
  | 0     | 0.727224287434013     | 0.728385903035214     |
  | 3     | 0.751939631315762     | 0.753260148619263     |
  | 6     | 0.755331323271746     | 0.756429246356953     |
  | 9     | 0.758423312478014     | 0.759572096027829     |
  | 12    | **0.760786710460919** | **0.762129054757838** |
  | 15    | 0.755978170224526     | 0.757160548021247     |



综上：  

- 效果最高：MAP达0.760，MRR达0.762  
- 经过6轮训练之后，MAP和MRR稳定在0.755以上  
- 训练轮数变多后，效果会小幅度下降，但没有明显过拟合现象，最后一轮的效果与最好的效果相差不到0.005  

