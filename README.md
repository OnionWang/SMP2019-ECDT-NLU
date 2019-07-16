SMP2019 ECDT 中文人机对话技术测评 任务一 自然语言理解
===================================

任务说明: http://conference.cipsc.org.cn/smp2019/evaluation.html

LeaderBoard: https://adamszq.github.io/smp2019ecdt_task1/


Process
---------
本次比赛的主要工作在于预训练模型的选择 + 分析数据并编写规则。
目前中文已有的开源预训练模型有3种，google的bert-chinese[1]，百度的ernie[2]以及哈工大的中文bert[3]。关于三者的比较可以参见[3]。

主要基于百度开源的ernie进行微调，ernie和bert是一样的架构，只是训练方式不同，主要采用whole word masking模型，加入了百度贴吧等非正式场合的语料，对于日常对话这种非正式语料有更好的效果，不过需要采用更大的初始学习率，在槽位识别上会有一个较大的提升。模型架构不算复杂，感兴趣可以看看modeling.py里的BertForTaskNLU，loss为三者loss相加，。

对这次比赛没什么太多要说的，我之所以没有提及比赛的名次，是因为这个比赛刚结束的第二天早上，我一时兴起在github上搜索了一下SMP-ECDT，结果绝望的发现该比赛17、18年都有举办，最关键的是，官方开源的数据竟然来自同一批数据集，区别在于前两年只有意图和领域识别，没有槽位标注，但是数据量非常可观，而且绝对是和本次比赛数据来自同一批数据当中的随机采样。所以这比赛要想拿第一，大家可以把这些数据利用起来。。。话就说到这里（我也很绝望啊。。。发现的晚了一天）


Code Framework
---------
* baidu_ernie/: 百度开源的ernie
* dataSet/:
	* train.json: 官方给定的训练数据
	* dic/: 自己搜集制作的部分槽位字典（用于规则）
	* process.py: 当时做数据分析时随手敲的代码，想分析数据的可以看一下（有点乱）
	* 往年数据.zip: 17、18年SMP-ECDT官方数据
* docs/:
	* rule.xlsx: 个人对数据做出的一些分析，用excel做成表格形式
* sample/: 提交和评估参考代码
* convert_tf_checkpoint_to_pytorch.py: 将tensorflow模型转成pytorch可读参数
* modeling.py: 模型架构（重点关注BertForTaskNLU），这是本项目用到的模型
* optimization.py: 优化器（周期学习率 + warmup_step）
* rule.py: 针对此数据写的一些规则（乱是乱了点...）
* run_classifier.py: 可以理解为main函数
* run_classifier_dataset_utils.py: 数据处理部分
* tokenization.py: 分词和建词表


Enviroment
---------
Python3.5

pytorch1.0.0

GPU(模型3G左右显存就够了)


Usage
---------
先下载预训练好的ernie模型（这里直接用了	[4]已经转好的模型）: https://pan.baidu.com/s/1I7kKVlZN6hl-sUbnvttJzA 

bash run.sh 0 (0是你想使用的GPU编号)


References:
---------
[1] https://github.com/google-research/bert

[2] https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE

[3] https://github.com/ymcui/Chinese-BERT-wwm

[4] https://github.com/ArthurRizar/tensorflow_ernie
