
# 从词到向量：


其实embedding models不仅仅是指word-embedding模型，Embedding models（嵌入模型）是一类用来将数据，例如文本、图片或其他类型的离散信息，映射到一个连续、低维度的向量空间的机器学习模型。在这个向量空间中，相似的数据距离更近，不相似的数据距离更远。我们这里主要是说word-embedding模型

## 我学习过的一些在线课程：
1.UCL的课程[《Representing Text as Data (II): Word Embeddings》](https://uclspp.github.io/PUBL0099/seminars/seminar7.html)
2.NanJing University[《计算传播学》](https://chengjun.github.io/mybook/10-word2vec.html)
3.[<Word Embeddings in Python>](https://medium.com/biased-algorithms/word-embeddings-in-python-a8085488d244)
4.UBC Vancouver School of Economics[<4.4 - Advanced - Word Embeddings>](https://comet.arts.ubc.ca/docs/4_Advanced/advanced_word_embeddings/advanced_word_embeddings_python_version.html)
5.[<Natural Language Processing with Python>](https://www.nltk.org/book/)
6.一个好玩的东西[embedding-explorer](https://centre-for-humanities-computing.github.io/embedding-explorer/)


## 1）为什么要用词向量：把语言嵌入量化坐标系


词向量（word embeddings）把“词”映射到实数向量，使“语义相似→几何相近、关系→向量运算”成为可能。它让文本材料进入回归、因果、网络、时序等量化分析，是将语言变为“可测量社会事实”的核心步骤。
早期静态嵌入（每个词一个向量）由 **word2vec** 与 **GloVe** 奠基；随后 ELMo/BERT 等**上下文嵌入**让同一词在不同语境中拥有不同表示；Transformer 体系（“Attention Is All You Need”）成为当代嵌入与大模型的主干。[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

---

## 2）模型谱系与要点

### 2.1 静态嵌入（每个词一个向量）

* **word2vec（CBOW/Skip-gram + 负采样）**：通过上下文预测目标词或反向，学习向量；经典的“king − man + woman ≈ queen”来自该系。理论上它等价于对 **Shifted-PMI** 共现矩阵的隐式加权分解（理解其何以有效的关键）。 
* **GloVe**：显式用全局共现统计构造损失，本质是矩阵分解视角。 ([aclanthology.org][2])
* **fastText（子词/字符 n-gram）**：解决 OOV 与形态丰富语言的小样本表示。 ([aclanthology.org][3])

### 2.2 上下文嵌入与大模型

* **ELMo**：双向语言模型内部态的函数，显著提升多项 NLP 任务。 ([aclanthology.org][4])
* **BERT/Transformer**：双向自注意力预训练，成为通用表示主流。 ([arxiv.org][5])

### 2.3 句子/段落嵌入

* **平均/加权平均**：简单有效，但忽视词序；可用 **SIF**（去均值+频率加权）改善。 ([openreview.net][6])
* **Sentence-BERT（SBERT）**：Siamese/Triplet 结构，将句子编码为可直接余弦检索的向量。 ([arxiv.org][7])

---

## 3）如何训练：语料、超参与稳定性

**语料**：规模越大、领域越贴近，词向量越可信。
**窗口大小**：小窗口偏句法，大窗口偏语义。
**负采样 k**、**子采样阈值**（丢弃高频词）、**维度 d**、**迭代数**：共同影响邻近性结构与收敛。
word2vec 原论文与“隐式矩阵分解”工作是理解这些超参如何改变几何结构的最佳入口。 ([arxiv.org][1])

**后处理**：

* **All-but-the-top**（去均值+去前几主导方向）可稳定提升下游效果；对上下文嵌入也有研究延伸。 ([arxiv.org][8])

---

## 4）如何评估：内在 vs 外在

衡量 embedding models 质量的指标主要包括以下几种：

 1.语义相似性:检查相似概念或词语在嵌入空间中距离是否较近。用 cosine similarity、欧氏距离等进行定量评估。
 2.下游任务表现:将嵌入作为分类、聚类、推荐等任务的特征输入，看任务的准确率、查准率、查全率等是否提升。
 3.邻近检索/最近邻查询（Nearest Neighbor Search）: 测试 embedding 对检索任务有多大帮助，例如相似文档、相似图片的召回率。
 4.可视化分析: 利用 PCA 或 t-SNE 等方法降维，可视化聚类效果和分布，观测语义结构是否符合预期。
 5.外部基准数据集评测: 用公开评测集（如Textual Similarity，Image Retrieval等）对模型结果评分，如 Spearman/Pearson correlation、MRR、NDCG 等指标。
 > 比如hugging face上常用的“[Massive Text Embedding Benchmark (MTEB)]()"
 > 参考资料：“[What are benchmark datasets in machine learning, and where can I find them?](https://milvus.io/ai-quick-reference/what-are-benchmark-datasets-in-machine-learning-and-where-can-i-find-them)”
 6.偏差与公平性测试:检查 embedding 是否有不公平的偏见牵连，如性别、种族偏见。
 > ⚠️但是注意有时候这个就是我们社会科学研究想要研究和分析的部分


---

## 5）如何降维与可视化：PCA / t-SNE / UMAP（你问到的重点）

**为什么降维**：便于可视化（2D/3D）与后续统计（如聚类/回归避免共线）。

* **PCA**：线性、全局结构好，速度快；适合先做“预降维到 50–100”，再接非线性方法。
* **t-SNE**：保局部邻域的可视化神器；**不要**解读簇大小与簇间距离的全局意义；**调参很关键**（perplexity、学习率、迭代数），且不提供“对新样本的显式映射”。 ([jmlr.org][10])
* **UMAP**：在保留全局/局部结构与速度上常优于 t-SNE，可用于可视化与通用非线性降维（支持任意目标维度，能学显式映射）。 ([arxiv.org][11])

**实操建议**：
1）先 **PCA→50**，再 **t-SNE/UMAP→2**；
2）t-SNE 的 perplexity 与样本量/密度相关，需网格搜索；
3）可视化只作“探索性”与“交流性”证据，统计推断用原空间或经验证的低维表征。 ([Distill][12])

---

## 6）如何对齐空间：跨时间（语义漂移）与跨语言

**对齐动机**：不同时间/语种各自训练的空间坐标系不一致，直接比较无意义。
**方法**：

* **正交 Procrustes** 对齐（SVD 闭式解）——主流稳健基线；时间序列嵌入分析与跨语词典诱导常用。 ([arxiv.org][13])
* **无监督/自学习** 跨语映射（Artetxe 等）：弱/零词典下的自举式对齐。 ([aclanthology.org][14])
* **Wasserstein-Procrustes** 及噪声显式建模等增强版本。 ([arxiv.org][15])

**时间语义变化（diachronic embeddings）**：
构建年代切片语料→分别训练→正交对齐→比较词向量漂移（如“gay”“family”“climate”等）；提出“频率/多义性与语义变化速率”的统计律。 ([Computer Science][16])

---

## 7）偏见测量与去偏（社会科学高频应用）

**WEAT**：用词向量复现 IAT 类型的联结差异，度量性别/族群等隐性偏见（Science）。 ([science.org][17])
**去偏**：

* **Hard debiasing**（性别子空间投影/中和/均衡）；但后续研究提示“表面去偏”未必消除所有间接偏见，需结合任务场景与数据治理。 ([papers.neurips.cc][18])

> 重要提示：嵌入反映的是“语言使用的统计规律”，不是“客观世界真理”；研究中应把偏见当作 **社会事实** 来测量，同时在应用中考虑 **伦理治理** 与 **风险隔离**。 ([science.org][17])

---

## 8）从词到句：如何得到句子/文档表征

* **加权平均 + SIF 去均值**：强力无监督基线，快且稳定。 ([openreview.net][6])
* **SBERT**：对语义匹配/聚类/检索极其高效（余弦即可），适合社会媒体帖子、新闻议题聚合、舆论簇分析。 ([arxiv.org][7])

---

## 9）把嵌入接入定量社会科学管线：常见设计

1. **态度/隐性偏见测量**：WEAT/向量距离差；与人口学/调查变量联动回归/分层模型。 ([science.org][17])
2. **价值观/意识形态演化**：时间切片+正交对齐+变化指标；与历史事件/政策变迁做断点/面板。 ([Computer Science][16])
3. **语义网络构建**：以相似度为边权，做群落发现与中心性比较。
4. **跨文化比较**：跨语对齐后比较同义场（如“family/权利/环境”话语）。 ([aclanthology.org][14])

---

## 10）可复现性与稳健性清单（实践强烈建议）

* **固定随机种子与版本**，记录：语料来源/清洗规则、分词/子词、窗口、维度、负采样、轮次、子采样阈值等；保存向量与词表映射。
* **报告内在+外在评估**，并给出可视化仅作探索说明。
* **后处理**（ABTT）与 **标准化**（L2）一并记录。 ([arxiv.org][8])
* **降维参数**（t-SNE perplexity/学习率、UMAP n_neighbors/min_dist）随附敏感性分析。 ([Distill][12])

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

---

## 附：一个“开箱即用”的研究工作流（你可以直接照做）

1. **数据准备**：定界语料（时间/地域/平台），清洗（去噪、标准化、去机器人/转推等）。
2. **训练**：fastText（静态）或 BERT/roberta（上下文）获取词/句向量；记录完整超参。 ([aclanthology.org][3])
3. **后处理**：ABTT 或 SIF；L2 规范化。 ([arxiv.org][8])
4. **评估**：WordSim/SimLex/类比 + 你的下游任务（如回归/分类/网络/因果前处理）。 ([aclanthology.org][21])
5. **降维与可视化**：PCA→t-SNE/UMAP；写清超参与局限解释。 ([jmlr.org][10])
6. **对齐分析**（如时间/跨语）：正交 Procrustes 或自学习跨语映射。 ([arxiv.org][24])
7. **偏见审计与治理**：WEAT/子空间检测 + 硬去偏（若用于应用）；论文中把偏见当作社会事实报告。 ([science.org][17])
8. **复现包**：发布词表、向量、训练脚本、评估脚本与可视化 notebook，确保可复跑。

---

如果你愿意，我可以把上述内容打包成一份 **GitHub README 模板或教学讲义（含 t-SNE/UMAP 可视化示例、超参表、评测脚本）**，并按你的研究语料（中文社媒/新闻/学术等）给出默认配置与基准结果。

[1]: https://arxiv.org/pdf/1301.3781?utm_source=chatgpt.com "Efficient Estimation of Word Representations in Vector Space"
[2]: https://aclanthology.org/D14-1162/?utm_source=chatgpt.com "GloVe: Global Vectors for Word Representation"
[3]: https://aclanthology.org/Q17-1010.pdf?utm_source=chatgpt.com "Enriching Word Vectors with Subword Information"
[4]: https://aclanthology.org/N18-1202/?utm_source=chatgpt.com "Deep Contextualized Word Representations"
[5]: https://arxiv.org/pdf/1810.04805?utm_source=chatgpt.com "arXiv:1810.04805v2 [cs.CL] 24 May 2019"
[6]: https://openreview.net/forum?id=SyK00v5xx&utm_source=chatgpt.com "A Simple but Tough-to-Beat Baseline for Sentence ..."
[7]: https://arxiv.org/pdf/1908.10084?utm_source=chatgpt.com "arXiv:1908.10084v1 [cs.CL] 27 Aug 2019"
[8]: https://arxiv.org/abs/1702.01417?utm_source=chatgpt.com "All-but-the-Top: Simple and Effective Postprocessing for Word Representations"
[9]: https://www.gabrilovich.com/resources/data/wordsim353/wordsim353.html?utm_source=chatgpt.com "The WordSimilarity-353 Test Collection - of Evgeniy Gabrilovich"
[10]: https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?utm_source=chatgpt.com "Visualizing Data using t-SNE"
[11]: https://arxiv.org/abs/1802.03426?utm_source=chatgpt.com "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"
[12]: https://distill.pub/2016/misread-tsne?utm_source=chatgpt.com "How to Use t-SNE Effectively"
[13]: https://arxiv.org/abs/1605.09096?utm_source=chatgpt.com "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change"
[14]: https://aclanthology.org/P18-1073/?utm_source=chatgpt.com "A robust self-learning method for fully unsupervised cross ..."
[15]: https://arxiv.org/abs/1805.11222?utm_source=chatgpt.com "Unsupervised Alignment of Embeddings with Wasserstein Procrustes"
[16]: https://cs.stanford.edu/people/jure/pubs/diachronic-acl16.pdf?utm_source=chatgpt.com "Diachronic Word Embeddings Reveal Statistical Laws of ..."
[17]: https://www.science.org/doi/10.1126/science.aal4230?utm_source=chatgpt.com "Semantics derived automatically from language corpora ..."
[18]: https://papers.neurips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf?utm_source=chatgpt.com "Man is to Computer Programmer as Woman is ..."
[19]: https://proceedings.neurips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf?utm_source=chatgpt.com "Neural Word Embedding as Implicit Matrix Factorization"
[20]: https://papers.nips.cc/paper/7181-attention-is-all-you-need?utm_source=chatgpt.com "Attention is All you Need"
[21]: https://aclanthology.org/J15-4004/?utm_source=chatgpt.com "SimLex-999: Evaluating Semantic Models With (Genuine) ..."
[22]: https://aclanthology.org/N13-1090.pdf?utm_source=chatgpt.com "Linguistic Regularities in Continuous Space Word ..."
[23]: https://arxiv.org/pdf/1802.03426?utm_source=chatgpt.com "UMAP: Uniform Manifold Approximation and Projection for ..."
[24]: https://arxiv.org/pdf/1702.03859?utm_source=chatgpt.com "offline bilingual word vectors, orthogonal"
[25]: https://openreview.net/pdf?id=SyK00v5xx&utm_source=chatgpt.com "A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SEN"
