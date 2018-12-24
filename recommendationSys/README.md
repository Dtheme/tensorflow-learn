## 简单的推荐系统实现
基于项亮的《推荐系统实践》所写的练习代码

* 基于协同过滤(UserCF)的模型
* 基于隐语义(LFM)的模型


### 运行

下载数据(http://grouplens.org/datasets/movielens/1m)，解压到data/目录中

* 数据预处理

    python3 manage.py preprocess

* 模型运行

    python3 manage.py [cf/lfm]

