
# 从词到向量：

其实embedding models不仅仅是指word-embedding模型，Embedding models（嵌入模型）是一类用来将数据，例如文本、图片或其他类型的离散信息，映射到一个连续、低维度的向量空间的机器学习模型。在这个向量空间中，相似的数据距离更近，不相似的数据距离更远。我们这里主要是说word-embedding模型

## 我自己试过的项目

我之前在博士论文研究老年歧视的时候原本想使用一些文本分析的方法，包括但不限于主题分析、词嵌入模型。从中文社交媒体上爬取了大量的文本数据，进行了一些简单尝试但是最终没有在博士论文中做这个分析。这里就简单的介绍一下[notebook](./word2vec-weibodata-test.ipynb)的内容:

使用基于之前在微博爬取的社交文本数据集，首先对文本内容进行了基本清洗与分词处理，使用 Gensim 训练 300 维 Word2Vec 词向量模型（相当于基于Word2Vec模型使用我自己的数据fine-tune），并通过 t-SNE / PCA 将词向量降维后进行可视化展示，分析词语之间的语义距离、主题聚类结构和相似词关系。

随后围绕“年轻人”与“老年人”构建语义对比，使用词向量的相似度函数 most_similar() 展示了不同群体的关联词汇。在可视化部分，Notebook 首先通过构造“年轻人-老年人”“贫穷-富裕”“男性-女性”等语义轴，将关键词投影到二维坐标系中，探索这些词在特定社会语义维度下的分布特征；之后，进一步引入三维可视化，将年龄、性别、贫富等维度融合，用 3D 散点图（Matplotlib 与 Plotly 实现）呈现复杂的语义空间结构；最后，利用 t-SNE 技术对一组关键词进行降维，绘制其整体语义关系图。

 > 这个大概是我一年前2024年做的一个小测试，然后一年的时间嵌入模型的发展已经不可同日而语，已经发展出来了基于更新的神经网络架构、以及多语言、多模态等嵌入模型……


## 我学习过的一些在线课程：

1.UCL的课程[《Representing Text as Data (II): Word Embeddings》](https://uclspp.github.io/PUBL0099/seminars/seminar7.html)

2.NanJing University[《计算传播学》](https://chengjun.github.io/mybook/10-word2vec.html)

3.Word Embeddings in Python的一篇科普文章[Word Embeddings in Python](https://medium.com/biased-algorithms/word-embeddings-in-python-a8085488d244)

4.UBC Vancouver School of Economics[<4.4 - Advanced - Word Embeddings>](https://comet.arts.ubc.ca/docs/4_Advanced/advanced_word_embeddings/advanced_word_embeddings_python_version.html)

5.[Natural Language Processing with Python](https://www.nltk.org/book/)

6.一个好玩的东西[embedding-explorer](https://centre-for-humanities-computing.github.io/embedding-explorer/)


## 为什么要用词向量：把语言嵌入量化坐标系

词向量（word embeddings）把“词”映射到实数向量，使“语义相似→几何相近、关系→向量运算”成为可能。它让文本材料进入回归、因果、网络、时序等量化分析，是将语言变为“可测量社会事实”的核心步骤。
早期静态嵌入（每个词一个向量）由 **word2vec** 与 **GloVe** 奠基；随后 ELMo/BERT 等**上下文嵌入**让同一词在不同语境中拥有不同表示；Transformer 体系（“Attention Is All You Need”）成为当代嵌入与大模型的主干。[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

---

## 模型谱系与要点

### 一、word embedding models的发展与谱系 ###

 1.早期方法（基于统计和矩阵分解）

  - LSA（Latent Semantic Analysis）：通过对词-文档共现矩阵做奇异值分解（SVD）提取低维稠密表达。

  - HAL、COALS、PPMI等：依赖各类词共现统计的方法。

 2.神经网络word embedding的黄金时代

  - Word2Vec（CBOW/Skip-gram，2013）：采用浅层神经网络，极大提升了embedding表达的语义能力与效率，成为分布式语义的核心工具。

  - GloVe（2014）：结合全局词频统计与局部上下文窗口，让向量既有统计优势也有神经表示能力。

 - FastText（2016）：在Word2Vec基础上引入子词信息，不仅适配无OOV词，还能更好地捕捉词内结构。

3.Contextual Word Embedding（上下文相关词向量）

 - ELMo（2018）：用双向LSTM处理文本获得动态词表示。

 - BERT（2018）：基于Transformer结构，不同输入上下文可生成不同的词向量，极大提升了文本理解能力。

 - GPT、XLNet、RoBERTa等：Transformer家族的其他优秀变体。

### 二、通用Embedding/语句Embedding模型的发展（2018至今）###

 1.Sentence Embedding阶段

 - Universal Sentence Encoder、InferSent：尝试获得句子的固定向量表示。

 - SBERT/SentenceTransformer：引入Siamese结构优化BERT、RoBERTa用于语句/段落嵌入。

 - SimCSE：利用对比学习进一步提升embedding质量。

 2.专用Embedding（跨模态、领域）

 - 针对代码（CodeBERT）、法律、医疗等专业领域做定制化训练。

 - 跨模态领域如CLIP（文本-图像）、MMTEB等。

 - 多语言/大规模Embedding模型
 > 这个领域是我目前在开展研究的一个部分，中国有很多不错的多语言嵌入模型比如Qwen3-4B、Qwen3-8B占据huggingface的benchmark前几名

 - LaBSE、XLM-R等支持多语言语句嵌入。


---

## 如何评估一个嵌入模型的质量

衡量 embedding models 质量的指标主要包括以下几种：

 1.语义相似性:检查相似概念或词语在嵌入空间中距离是否较近。用 cosine similarity、欧氏距离等进行定量评估。

 2.下游任务表现:将嵌入作为分类、聚类、推荐等任务的特征输入，看任务的准确率、查准率、查全率等是否提升。

 3.邻近检索/最近邻查询（Nearest Neighbor Search）: 测试 embedding 对检索任务有多大帮助，例如相似文档、相似图片的召回率。

 4.可视化分析: 利用 PCA 或 t-SNE 等方法降维，可视化聚类效果和分布，观测语义结构是否符合预期。

 5.外部基准数据集评测: 用公开评测集（如Textual Similarity，Image Retrieval等）对模型结果评分，如 Spearman/Pearson correlation、MRR、NDCG 等指标。  
 > 比如hugging face上常用的“[Massive Text Embedding Benchmark (MTEB)](https://huggingface.co/mteb)"，还有排行版[Embedding Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
 > 参考资料：“[What are benchmark datasets in machine learning, and where can I find them?](https://milvus.io/ai-quick-reference/what-are-benchmark-datasets-in-machine-learning-and-where-can-i-find-them)”

 6.偏差与公平性测试:检查 embedding 是否有不公平的偏见牵连，如性别、种族偏见。

 > ⚠️但是注意有时候这个就是我们社会科学研究想要研究和分析的部分


---

## 词向量的维度以及其降维方法与可视化

### 一、什么是词向量维度：
在embedding任务中，“维度”指的是每个向量的分量数量。常见的理解和解答如下：

**1. 为什么embedding会是多维度？**

- 维度（dimension）是数学和编程中的术语。例如：
  - 100维embedding向量：每个词或句子被表示为具有100个实数分量的向量 $$(x_1, x_2, ..., x_{100})$$。
  - 也可以理解为有100个特征描述一个对象。

- 高维embedding能捕捉更丰富、复杂的语义信息。例如一个单词的特征——词性、含义、情感色彩、上下文用法等，都可以由不同维度协同编码。

- 神经网络（如Word2Vec、BERT等）训练embedding时，通常会将每个词/句子/文档映射成一个**高维向量**，以便模型更好地区分不同语义和结构。

**2. embedding常见的“多少维”？**
- 词向量（word embedding）：常见为50、100、200、300（如GloVe/Word2Vec），有的自定义为512或更高。
- 句子向量（sentence embedding）：通常为512、768、1024（如BERT-base为768维），大模型甚至2048维、4096维等。
- 维度选择权衡：维数高表达力强但训练、存储/计算资源消耗大，低维简单但信息表达能力有限。

> embedding的多维度本质上是为了表达“多种潜在特征”。
> 维度数量取决于模型设计、任务需求及实际场景，几百到几千维较常见。
> 维度参数的介绍可以见embedding model的说明文档！！！！

### 二、降维与可视化：

**为什么降维**：

 - 降维（如t-SNE、PCA、UMAP）就是为了将这高维空间（如768维的BERT向量）“压缩”到2维或3维，便于可视化（2D/3D）与后续统计（如聚类/回归避免共线）。

**机器学习中的降维技术**

 - 其实机器学习中有很多降维技术，而且也不只是在词嵌入分析中需要降维技术，这个部分主要介绍几种词嵌入分析中常用的降维技术，其他的降维技术及应用可以参考以下链接：

 > 1.[Introduction to Dimensionality Reduction](https://www.geeksforgeeks.org/machine-learning/dimensionality-reduction/)
 > 2.[WiKi-Dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)
 > 3.essay [IBM: What is dimensionality reduction?](https://www.ibm.com/think/topics/dimensionality-reduction)
 > 4.academic paper literiture review <[Review of Dimension Reduction Methods](https://www.scirp.org/journal/paperinformation?paperid=111638)>
 > 5.essay with code <[Dimensionality Reduction for Machine Learning](https://www.scirp.org/journal/paperinformation?paperid=111638)>
 > 6.[scikit-learn官方文档：降维模块](https://scikit-learn.org/stable/modules/unsupervised_reduction.html)

**本文详细介绍的降维技术：PCA、t-SNE、UMAP**


---
## 社会科学领域研究中的词向量分析

### 1.常见的研究设计

1. **态度/隐性偏见测量**：WEAT/向量距离差；与人口学/调查变量联动回归/分层模型。 
2. **价值观/意识形态演化**：时间切片+正交对齐+变化指标；与历史事件/政策变迁做断点/面板。 
3. **语义网络构建**：以相似度为边权，做群落发现与中心性比较。
4. **跨文化比较**：跨语对齐后比较同义场（如“family/权利/环境”话语）。 

---

### 2.基本的研究工作流

1. **数据准备**：定界语料（时间/地域/平台），清洗（去噪、标准化、去机器人/转推等）。
2. **训练**：fastText（静态）或 BERT/roberta（上下文）获取词/句向量；记录完整超参。 ([aclanthology.org][3])
3. **后处理**：ABTT 或 SIF；L2 规范化。 ([arxiv.org][8])
4. **评估**：WordSim/SimLex/类比 + 你的下游任务（如回归/分类/网络/因果前处理）。 ([aclanthology.org][21])
5. **降维与可视化**：PCA→t-SNE/UMAP；写清超参与局限解释。 ([jmlr.org][10])
6. **对齐分析**（如时间/跨语）：正交 Procrustes 或自学习跨语映射。 ([arxiv.org][24])
7. **偏见审计与治理**：WEAT/子空间检测 + 硬去偏（若用于应用）；论文中把偏见当作社会事实报告。 ([science.org][17])
8. **复现包**：发布词表、向量、训练脚本、评估脚本与可视化 notebook，确保可复跑。

---
## 参考文献（精选，按主题）

**基础与模型**：

* Mikolov, T. et al. *Efficient Estimation of Word Representations in Vector Space*（word2vec）. ([arxiv.org][1])
* Pennington, J. et al. *GloVe: Global Vectors for Word Representation*. ([aclanthology.org][2])
* Levy, O. & Goldberg, Y. *Neural Word Embedding as Implicit Matrix Factorization*. （word2vec≈SPPMI 分解） ([proceedings.neurips.cc][19])
* Bojanowski, P. et al. *Enriching Word Vectors with Subword Information*（fastText）。 ([aclanthology.org][3])
* Peters, M. et al. *Deep Contextualized Word Representations*（ELMo）。 ([aclanthology.org][4])
* Devlin, J. et al. *BERT*. ([arxiv.org][5])
* Vaswani, A. et al. *Attention Is All You Need*（Transformer）。 ([papers.nips.cc][20])

**评估基准**：

* Finkelstein, L. et al. *WordSim-353*. ([gabrilovich.com][9])
* Hill, F. et al. *SimLex-999*. ([aclanthology.org][21])
* Mikolov, T. et al. *Linguistic Regularities…*（类比测试）。 ([aclanthology.org][22])

**降维与可视化**：

* van der Maaten, L. & Hinton, G. *t-SNE*. ([jmlr.org][10])
* McInnes, L. et al. *UMAP*. ([arxiv.org][23])
* Wattenberg, M. et al. *How to Use t-SNE Effectively*（实践注意）。 ([Distill][12])

**对齐/时间/跨语**：

* Hamilton, W. et al. *Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change*. ([arxiv.org][13])
* Smith, S. et al. *Offline Bilingual Word Vectors, Orthogonal Transformations…*（正交 Procrustes）。 ([arxiv.org][24])
* Artetxe, M. et al. *A Robust Self-Learning Method for Fully Unsupervised Cross-Lingual Mappings*. ([aclanthology.org][14])
* Grave, E. et al. *Unsupervised Alignment with Wasserstein Procrustes*. ([arxiv.org][15])

**句子表示**：

* Arora, S. et al. *A Simple but Tough-to-Beat Baseline for Sentence Embeddings*（SIF）。 ([openreview.net][25])
* Reimers, N. & Gurevych, I. *Sentence-BERT*. ([arxiv.org][7])

**偏见与治理**：

* Caliskan, A. et al. *Semantics derived automatically… contain human-like biases*（WEAT，Science）。 ([science.org][17])
* Bolukbasi, T. et al. *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings*. ([papers.neurips.cc][18])

**后处理**：

* Mu, J. et al. *All-but-the-Top: Simple and Effective Postprocessing for Word Representations*. ([arxiv.org][8])