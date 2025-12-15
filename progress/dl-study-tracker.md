# 深度学习学习进度追踪

**最后更新**: 2025-12-15
**总学习天数**: 1
**总学习时长**: ~2 小时
**当前阶段**: 神经网络基础

---

## 📊 快速统计

- **总主题数**: 56 (A-H 八个领域)
- **已掌握**: 13 个主题
- **进行中**: 0 个主题
- **待学习**: 43 个主题
- **整体进度**: 23% (13/56)

---

## 🎯 学习目标

- **短期目标** (1-2 个月)：掌握深度学习基础理论和 PyTorch 实现
- **中期目标** (3-6 个月)：完成经典论文复现，构建项目作品集
- **长期目标**：做 AI 研究 + 找算法工作

---

## 📚 领域进度总览

| 领域 | 权重 | 已掌握 | 总数 | 进度 | 状态 |
|------|------|--------|------|------|------|
| A. 数学基础 | 20% | 3/5 | 5 | 60% | 🟡 进行中 |
| B. 神经网络基础 | 18% | 10/8 | 8 | 125% | ✅ 超额完成 |
| C. 深度学习架构 | 16% | 0/8 | 8 | 0% | ⚪ 未开始 |
| D. 计算机视觉 | 12% | 0/6 | 6 | 0% | ⚪ 未开始 |
| E. 自然语言处理 | 12% | 0/7 | 7 | 0% | ⚪ 未开始 |
| F. 训练与优化 | 10% | 0/8 | 8 | 0% | ⚪ 未开始 |
| G. 框架与实现 | 8% | 0/7 | 7 | 0% | ⚪ 未开始 |
| H. 高级主题 | 4% | 0/7 | 7 | 0% | ⚪ 未开始 |

**说明**: B 领域超额是因为包含了一些实践技能（调试、实验）

---

## ✅ 已掌握的主题

### A. 数学基础 (3/5 完成)

#### A.1 线性代数 ✅
**掌握日期**: 2025-12-15
**置信度**: 中高
**关键知识点**:
- 矩阵乘法在神经网络中的意义
- 矩阵维度分析
- 转置操作
- 向量化计算的优势

**参考资源**: 本科课程基础 + Session 1 实践

#### A.2 微积分（链式法则）✅
**掌握日期**: 2025-12-15
**置信度**: 高
**关键知识点**:
- 链式法则：dz/dx = (dz/dy) × (dy/dx)
- 偏导数的计算
- 梯度的概念
- 反向传播中的应用

**参考资源**: 本科课程 + Session 1 推导

#### A.3 概率统计（基础）✅
**掌握日期**: 2025-12-15
**置信度**: 中
**关键知识点**:
- 基本概率概念
- 期望、方差（本科课程）

**备注**: 需要加强在深度学习中的具体应用

---

### B. 神经网络基础 (10/8 完成 - 超额)

#### B.6 感知机和激活函数 ✅
**掌握日期**: 2025-12-15
**置信度**: 高
**关键知识点**:
- 感知机数学定义：z = w^T x + b
- 几何理解：线性分类器
- 激活函数的必要性（非线性变换）
- Sigmoid: σ(z) = 1/(1+e^(-z))
- ReLU: max(0, z)
- Tanh 函数

**参考资源**: Session 1 详细讲解

#### B.7 前向传播 ✅
**掌握日期**: 2025-12-15
**置信度**: 高
**关键知识点**:
- 层与层之间的计算流程
- Z = WX + b（线性变换）
- A = σ(Z)（激活）
- 缓存中间值供反向传播使用

**代码实现**: `AI-study/project/backpropagation/example.py`

#### B.8 损失函数 ✅
**掌握日期**: 2025-12-15
**置信度**: 中高
**关键知识点**:
- 均方误差（MSE）：L = (ŷ - y)²
- 损失函数的作用：衡量预测误差
- 损失下降 = 学习进行

**备注**: 下节课学习交叉熵损失函数

#### B.9 反向传播算法 ✅
**掌握日期**: 2025-12-15
**置信度**: 高
**关键知识点**:
- 链式法则的应用
- 误差从后向前传播：δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ · δ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)
- 梯度计算：∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ
- 矩阵乘法顺序的重要性

**代码实现**: 从零实现反向传播 ✅
**关键 Bug 修复**: W2.T @ delta2（正确）vs delta2 @ W2.T（错误）

#### B.10 梯度下降 ✅
**掌握日期**: 2025-12-15
**置信度**: 高
**关键知识点**:
- 参数更新规则：W ← W - α · ∂L/∂W
- 学习率 α 的作用
- 学习率对训练的影响（实验验证）

**实验结果**:
| 学习率 | 最终损失 |
|--------|----------|
| 0.1    | 0.002973 |
| 1.0    | 0.000120 |
| 10.0   | 0.000013 |

#### B.Extra 1: 多层网络与特征学习 ✅
**掌握日期**: 2025-12-15
**置信度**: 高
**关键知识点**:
- XOR 问题：线性不可分的经典案例
- 多层网络的作用：特征变换
- 核心理解：**"将原本数据映射成线性可分的状态"**
- 隐藏层学习中间表示

**实验验证**: 手工权重解决 XOR

#### B.Extra 2: 从零实现神经网络 ✅
**掌握日期**: 2025-12-15
**置信度**: 高
**项目**: `/AI-study/project/backpropagation/example.py`
- 实现两层神经网络
- 解决 XOR 问题（100% 准确率）
- 完整的前向+反向传播
- 梯度下降优化

#### B.Extra 3: 矩阵维度分析与调试 ✅
**掌握日期**: 2025-12-15
**置信度**: 中高
**关键技能**:
- 检查矩阵运算的维度匹配
- 使用维度分析定位 Bug
- 理解 Broadcasting 机制（NumPy）

**实战案例**: 修复反向传播的矩阵乘法顺序错误

#### B.Extra 4: 超参数实验方法 ✅
**掌握日期**: 2025-12-15
**置信度**: 高
**实验设计**:
- 控制变量法
- 对比不同学习率（0.1 - 10.0）
- 对比不同隐藏层大小（2, 4, 8）
- 可视化损失曲线

**核心发现**:
- 简单问题可以用大学习率
- 复杂问题需要小心调参
- 理论最优 ≠ 实践最优

#### B.Extra 5: 数据集划分理解 ✅
**掌握日期**: 2025-12-15
**置信度**: 高
**关键区别**:
- 训练集：模型直接学习
- 验证集：人根据表现调整超参数
- 测试集：最终评估，绝对不能用来调整

**类比**: 课本练习题 / 模拟考 / 期末考

---

## 🚧 当前知识盲点

### 高优先级（下节课解决）
1. **梯度消失/爆炸** (B.9 扩展)
   - 为什么深层网络难训练？
   - Sigmoid 的梯度消失问题
   - 解决方案：ReLU、权重初始化、BatchNorm

2. **现代激活函数** (B.6 扩展)
   - ReLU 及其变种
   - LeakyReLU, ELU, GELU
   - 如何选择激活函数？

3. **权重初始化策略** (B.11)
   - Xavier 初始化
   - He 初始化
   - 为什么初始化重要？

### 中优先级
4. **NumPy/PyTorch 熟练度**
   - 矩阵操作的习惯养成
   - 向量化编程思维
   - 解决方案：多写代码，每次写维度注释

5. **Batch Normalization** (B.12)
   - 训练加速的秘密武器
   - Internal Covariate Shift

---

## 📅 学习计划

**课程体系参考**: 吴恩达深度学习专项课程 (Deep Learning Specialization) + Stanford CS231n/CS224n

### Week 1: 神经网络基础 (当前进度: Day 1/7 完成)
**对应**: 吴恩达课程 1 - Neural Networks and Deep Learning

- [x] **Day 1** (2025-12-15): 感知机、前向/反向传播、从零实现 ✅
  - 对应吴恩达 Course 1, Week 2-3
  - 完成项目：从零实现神经网络解决 XOR

- [ ] **Day 2** (2025-12-16): 梯度消失/爆炸、激活函数、权重初始化
  - 对应吴恩达 Course 1, Week 4 + Course 2, Week 1
  - ReLU vs Sigmoid vs Tanh
  - Xavier/He 初始化
  - BatchNorm 原理

- [ ] **Day 3**: PyTorch 基础入门
  - Tensor 操作、自动微分
  - nn.Module 的使用
  - 对应吴恩达编程作业的 PyTorch 版本

- [ ] **Day 4**: 浅层网络实战 - Planar Data Classification
  - 对应吴恩达 Course 1, Week 3 作业
  - 实现：分类螺旋数据集
  - 可视化决策边界

- [ ] **Day 5**: 深层网络实战 - MNIST 手写数字识别
  - 对应吴恩达 Course 1, Week 4 作业
  - 目标：95%+ 准确率
  - 实验：不同层数、不同激活函数的效果

- [ ] **Day 6**: 优化算法详解
  - 对应吴恩达 Course 2, Week 2
  - Mini-batch Gradient Descent
  - Momentum, RMSprop, Adam
  - 学习率衰减策略

- [ ] **Day 7**: 正则化 + 综合项目
  - 对应吴恩达 Course 2, Week 1
  - L2 正则化、Dropout
  - 梯度检验（Gradient Checking）
  - 综合项目：猫狗分类器（浅层网络版）

### Week 2-3: 卷积神经网络 (CNN)
**对应**: 吴恩达课程 4 - Convolutional Neural Networks + Stanford CS231n

- [ ] **Week 2, Day 1-2**: CNN 基础
  - 对应吴恩达 Course 4, Week 1
  - 卷积操作的数学原理（边缘检测、特征提取）
  - 池化层（Max pooling, Average pooling）
  - 从零实现卷积操作

- [ ] **Week 2, Day 3-4**: 经典 CNN 架构
  - 对应吴恩达 Course 4, Week 2
  - LeNet-5 (手写数字识别)
  - AlexNet (ImageNet 冠军)
  - VGG (简单但深)
  - ResNet (残差连接解决退化问题)
  - 实验：在 CIFAR-10 上对比不同架构

- [ ] **Week 2, Day 5-7**: CNN 实战项目 1
  - 对应吴恩达 Course 4, Week 2 作业
  - 项目：CIFAR-10 图像分类
  - 目标：85%+ 准确率
  - 数据增强技术（Data Augmentation）

- [ ] **Week 3, Day 1-3**: 目标检测
  - 对应吴恩达 Course 4, Week 3
  - 目标检测算法：R-CNN, Fast R-CNN, YOLO
  - 边界框回归（Bounding Box Regression）
  - 非极大值抑制（NMS）
  - 实验：YOLO 实现简单物体检测

- [ ] **Week 3, Day 4-5**: 人脸识别与风格迁移
  - 对应吴恩达 Course 4, Week 4
  - 人脸验证 vs 人脸识别
  - Siamese Network 和 Triplet Loss
  - Neural Style Transfer
  - 项目：实现风格迁移（把照片变成梵高画风）

- [ ] **Week 3, Day 6-7**: CNN 综合项目
  - 迁移学习（Transfer Learning）
  - 使用预训练模型（ResNet, VGG）
  - 项目：自定义图像分类器（选自己感兴趣的主题）

### Week 4-5: 序列模型 (RNN, LSTM, GRU)
**对应**: 吴恩达课程 5 - Sequence Models

- [ ] **Week 4, Day 1-2**: RNN 基础
  - 对应吴恩达 Course 5, Week 1
  - RNN 的前向传播和反向传播（BPTT）
  - 梯度消失问题在 RNN 中的体现
  - 从零实现简单 RNN

- [ ] **Week 4, Day 3-4**: LSTM 和 GRU
  - 对应吴恩达 Course 5, Week 1
  - LSTM 的门控机制（遗忘门、输入门、输出门）
  - GRU 的简化设计
  - 为什么 LSTM 能解决长期依赖问题？

- [ ] **Week 4, Day 5-7**: RNN 应用 1 - 字符级语言模型
  - 对应吴恩达 Course 5, Week 1 作业
  - 项目：训练莎士比亚文本生成器
  - 采样技术（温度采样）
  - 梯度裁剪（Gradient Clipping）

- [ ] **Week 5, Day 1-3**: Word Embeddings
  - 对应吴恩达 Course 5, Week 2
  - Word2Vec (Skip-gram, CBOW)
  - GloVe
  - 负采样（Negative Sampling）
  - 实验：可视化词向量（t-SNE）

- [ ] **Week 5, Day 4-5**: 序列到序列模型
  - 对应吴恩达 Course 5, Week 3
  - Seq2Seq 架构（Encoder-Decoder）
  - Beam Search
  - 项目：日期格式转换

- [ ] **Week 5, Day 6-7**: 注意力机制入门
  - 对应吴恩达 Course 5, Week 3
  - Attention Mechanism 的动机
  - Bahdanau Attention
  - 项目：机器翻译（英译法）

### Week 6-8: Transformer 和现代 NLP
**对应**: 吴恩达课程 5, Week 4 + Stanford CS224n + 论文阅读

- [ ] **Week 6, Day 1-3**: Transformer 架构详解
  - 论文："Attention is All You Need"
  - Self-Attention 机制的数学推导
  - Multi-Head Attention
  - Positional Encoding
  - 从零实现 Transformer（简化版）

- [ ] **Week 6, Day 4-7**: BERT 和 GPT
  - BERT: 双向编码器（Masked LM）
  - GPT: 自回归语言模型
  - 预训练 + 微调范式
  - 实验：使用 Hugging Face Transformers

- [ ] **Week 7-8**: NLP 综合项目
  - 文本分类（情感分析）
  - 命名实体识别（NER）
  - 问答系统
  - 选择一个深入实现

### Week 9-10: 优化与调试深度学习 (进阶)
**对应**: 吴恩达课程 2 - Improving Deep Neural Networks + 3 - Structuring ML Projects

- [ ] 超参数调优策略
- [ ] 批归一化（Batch Norm）深入
- [ ] 诊断偏差/方差问题
- [ ] 迁移学习最佳实践
- [ ] 端到端深度学习
- [ ] 错误分析（Error Analysis）
- [ ] 多任务学习

### Week 11-12: 高级主题与前沿研究
- [ ] 生成对抗网络（GAN）
- [ ] 变分自编码器（VAE）
- [ ] 图神经网络（GNN）
- [ ] 强化学习基础
- [ ] 模型压缩与部署
- [ ] 自监督学习
- [ ] 少样本学习（Few-shot Learning）

---

## 📖 学习资源清单

### 核心课程（按优先级）

#### 1. 吴恩达深度学习专项课程 ⭐⭐⭐⭐⭐ (最高优先级)
**平台**: [Coursera - Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

**为什么选吴恩达？**
- ✅ 讲解清晰，从零开始，适合初学者
- ✅ 数学推导详细但不过分复杂
- ✅ 配套编程作业（NumPy + TensorFlow）
- ✅ 覆盖完整的深度学习体系
- ✅ 工业界最佳实践（来自 Andrew Ng 在 Google Brain, Baidu 的经验）

**5 门课程结构**：
1. **Neural Networks and Deep Learning** (4周)
   - Week 1: 深度学习概述
   - Week 2: 神经网络基础（逻辑回归、梯度下降）
   - Week 3: 浅层神经网络
   - Week 4: 深层神经网络
   - 🎯 作业：平面数据分类、猫分类器

2. **Improving Deep Neural Networks** (3周)
   - Week 1: 正则化、优化、梯度检验
   - Week 2: 优化算法（Momentum, Adam）
   - Week 3: 超参数调优、BatchNorm、编程框架
   - 🎯 作业：正则化对比实验、优化算法对比

3. **Structuring Machine Learning Projects** (2周)
   - Week 1: ML 策略（正交化、单一数字指标）
   - Week 2: ML 策略（错误分析、迁移学习、端到端学习）
   - 🎯 这门课偏理论，没有编程作业但很重要！

4. **Convolutional Neural Networks** (4周)
   - Week 1: 卷积操作基础
   - Week 2: 经典网络（LeNet, AlexNet, VGG, ResNet, Inception）
   - Week 3: 目标检测（YOLO, R-CNN）
   - Week 4: 人脸识别、神经风格迁移
   - 🎯 作业：手势识别、ResNet 实现、YOLO、人脸识别、风格迁移

5. **Sequence Models** (3周)
   - Week 1: RNN, LSTM, GRU
   - Week 2: Word Embeddings (Word2Vec, GloVe)
   - Week 3: Seq2Seq, Attention
   - Week 4: Transformer (新增内容)
   - 🎯 作业：恐龙名字生成、爵士乐生成、机器翻译、触发词检测

**学习建议**：
- 每周学习时间：10-12 小时
- 所有编程作业都要做！
- 可以用 PyTorch 重新实现一遍作业（官方用 TensorFlow）
- 课程视频可以 1.5x 速度看

---

#### 2. Stanford CS231n - CNN for Visual Recognition ⭐⭐⭐⭐⭐
**网站**: [cs231n.stanford.edu](http://cs231n.stanford.edu/)

**为什么选 CS231n？**
- ✅ 斯坦福计算机视觉经典课程
- ✅ 数学推导更深入（比吴恩达更硬核）
- ✅ 涵盖最新研究进展
- ✅ 配套作业质量极高（PyTorch）

**课程特色**：
- Lecture 1: 计算机视觉简介
- Lecture 2: 图像分类
- Lecture 3: 损失函数与优化
- **Lecture 4: 反向传播与神经网络** ← 我们 Day 1 刚学的！
- Lecture 5-6: CNN 架构
- Lecture 7: 训练神经网络技巧（重要！）
- Lecture 8-9: 深度学习软件、CNN 架构进阶
- Lecture 10: RNN
- Lecture 11: 目标检测与分割
- Lecture 12: 可视化与理解
- Lecture 13: 生成模型

**3 个编程作业**：
- Assignment 1: kNN, SVM, Softmax, 两层神经网络
- Assignment 2: 全连接网络, BatchNorm, Dropout, CNN
- Assignment 3: RNN/LSTM, Network Visualization, Style Transfer, GAN

**学习建议**：
- 配合吴恩达 CNN 课程一起学
- 重点看 Lecture 4, 7, 11, 12
- 作业一定要做！

---

#### 3. Stanford CS224n - NLP with Deep Learning ⭐⭐⭐⭐
**网站**: [web.stanford.edu/class/cs224n/](http://web.stanford.edu/class/cs224n/)

**为什么选 CS224n？**
- ✅ NLP 领域最权威课程
- ✅ 涵盖 Transformer, BERT, GPT
- ✅ 配套作业质量极高

**学习建议**：
- 在学完 RNN 后开始
- 重点看 Transformer 相关讲座

---

### 已使用
- ✅ 本科数学课程（线代、微积分、概率）
- ✅ 机器学习课程（80/100）
- ✅ Session 1 定制化讲解

### 视频资源（补充理解）

#### 3Blue1Brown - 神经网络系列 ⭐⭐⭐⭐⭐
**链接**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

**特色**：
- ✅ 最佳可视化！动画超级直观
- ✅ 4 个视频，每个 15-20 分钟
- ✅ 完美的入门材料

**内容**：
- 视频 1: 神经网络是什么？
- 视频 2: 梯度下降
- 视频 3: 反向传播的计算
- 视频 4: 反向传播的微积分

**学习建议**：
- [ ] 视频 1-2：今天看（复习 Day 1 内容）
- [ ] 视频 3-4：明天看（配合 Day 2 学习）

---

### 书籍资源

#### Deep Learning Book (Goodfellow, Bengio, Courville) ⭐⭐⭐⭐
**在线阅读**: [deeplearningbook.org](https://www.deeplearningbook.org/)

**特色**：
- ✅ 深度学习"圣经"
- ✅ 数学推导严谨
- ✅ 免费在线阅读

**推荐章节**（不用全读）：
- Chapter 5: Machine Learning Basics
- Chapter 6: Deep Feedforward Networks（我们 Day 1 刚学的）
- Chapter 7: Regularization
- Chapter 8: Optimization
- Chapter 9: CNN
- Chapter 10: RNN
- Chapter 12: Applications

**学习建议**：
- 作为参考书，遇到不懂的概念再查
- 不要一开始就啃，会很枯燥

---

#### Dive into Deep Learning (d2l.ai) ⭐⭐⭐⭐⭐
**网站**: [d2l.ai](https://d2l.ai/)

**特色**：
- ✅ 互动式教材，代码 + 理论结合
- ✅ 支持 PyTorch, TensorFlow, MXNet
- ✅ 免费在线阅读
- ✅ 可以直接在浏览器运行代码（Jupyter Notebook）

**学习建议**：
- 非常适合动手实践
- 每一章都有配套代码
- 推荐和吴恩达课程并行学习

---

### 论文阅读（进阶）

#### 必读经典论文（按学习顺序）

**Week 1-2: 神经网络基础**
- [ ] Glorot & Bengio (2010): "Understanding the difficulty of training deep feedforward neural networks" (Xavier 初始化)
- [ ] He et al. (2015): "Delving Deep into Rectifiers" (He 初始化)
- [ ] Ioffe & Szegedy (2015): "Batch Normalization"

**Week 2-3: CNN**
- [ ] LeCun et al. (1998): "Gradient-Based Learning Applied to Document Recognition" (LeNet)
- [ ] Krizhevsky et al. (2012): "ImageNet Classification with Deep CNNs" (AlexNet)
- [ ] Simonyan & Zisserman (2014): "Very Deep Convolutional Networks" (VGG)
- [ ] He et al. (2015): "Deep Residual Learning for Image Recognition" (ResNet) ⭐⭐⭐

**Week 4-5: RNN & Attention**
- [ ] Hochreiter & Schmidhuber (1997): "Long Short-Term Memory" (LSTM)
- [ ] Cho et al. (2014): "Learning Phrase Representations using RNN Encoder-Decoder" (GRU)
- [ ] Bahdanau et al. (2014): "Neural Machine Translation by Jointly Learning to Align and Translate" (Attention)

**Week 6-8: Transformer**
- [ ] Vaswani et al. (2017): "Attention is All You Need" (Transformer) ⭐⭐⭐⭐⭐
- [ ] Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers" ⭐⭐⭐⭐⭐
- [ ] Radford et al. (2018): "Improving Language Understanding by Generative Pre-Training" (GPT)

---

### 实践资源

#### Kaggle ⭐⭐⭐⭐⭐
**网站**: [kaggle.com](https://www.kaggle.com/)

**推荐竞赛/数据集**（入门级）：
- [ ] Titanic: 经典入门竞赛
- [ ] Digit Recognizer (MNIST): 手写数字识别
- [ ] Dogs vs. Cats: 图像分类
- [ ] NLP Getting Started: 灾难推文分类

---

#### Papers with Code ⭐⭐⭐⭐⭐
**网站**: [paperswithcode.com](https://paperswithcode.com/)

**特色**：
- ✅ 论文 + 代码实现
- ✅ 按任务分类（图像分类、目标检测等）
- ✅ 可以看到 SOTA（state-of-the-art）结果

---

### PyTorch 学习资源

#### PyTorch 官方教程 ⭐⭐⭐⭐⭐
**网站**: [pytorch.org/tutorials/](https://pytorch.org/tutorials/)

**推荐教程**（按顺序）：
- [ ] 60 Minute Blitz（必看！）
- [ ] Learning PyTorch with Examples
- [ ] Neural Networks Tutorial
- [ ] Training a Classifier

**学习建议**：
- Day 3 集中学习
- 边看边敲代码

---

### 社区与博客

#### 优质技术博客
- [ ] [Andrej Karpathy's Blog](http://karpathy.github.io/): 深度学习大牛，讲解深入浅出
- [ ] [Distill.pub](https://distill.pub/): 最美的 ML 可视化博客
- [ ] [Lil'Log](https://lilianweng.github.io/): OpenAI 研究员的博客，质量极高
- [ ] [Jay Alammar's Blog](https://jalammar.github.io/): Transformer 可视化讲解（必看！）

#### 中文资源
- [ ] [动手学深度学习](https://zh.d2l.ai/): Dive into Deep Learning 中文版
- [ ] [Deep Learning 中文版](https://github.com/exacity/deeplearningbook-chinese): Deep Learning Book 中文翻译

---

## 🎯 吴恩达课程经典实验对照表

这是吴恩达课程中最经典的编程作业，我们会用 PyTorch 重新实现：

| 周次 | 吴恩达作业 | 我们的实现 | 数据集 | 难度 |
|------|-----------|-----------|--------|------|
| Week 1, Day 4 | Planar Data Classification | ✅ 待完成 | 生成的螺旋数据 | ⭐⭐ |
| Week 1, Day 5 | Deep Neural Network | ✅ 待完成 | MNIST | ⭐⭐⭐ |
| Week 1, Day 7 | Cat vs Non-Cat | ✅ 待完成 | 猫图片 | ⭐⭐ |
| Week 2, Day 5-7 | CIFAR-10 Classification | ✅ 待完成 | CIFAR-10 | ⭐⭐⭐⭐ |
| Week 3, Day 1-3 | YOLO Object Detection | ✅ 待完成 | COCO | ⭐⭐⭐⭐⭐ |
| Week 3, Day 4-5 | Neural Style Transfer | ✅ 待完成 | 艺术图片 | ⭐⭐⭐⭐ |
| Week 4, Day 5-7 | Dinosaur Name Generation | ✅ 待完成 | 恐龙名字列表 | ⭐⭐⭐ |
| Week 5, Day 4-5 | Machine Translation | ✅ 待完成 | 日期格式转换 | ⭐⭐⭐⭐ |

---

## 🎯 里程碑

### 已完成 ✅
- [x] 第一个完整的神经网络实现（XOR 问题）
- [x] 理解反向传播的数学原理
- [x] 独立调试并修复代码 Bug

### 即将完成
- [ ] PyTorch 实现第一个分类器
- [ ] MNIST 数据集达到 95%+ 准确率

### 长期目标
- [ ] 复现一篇经典 CV 论文
- [ ] 复现一篇经典 NLP 论文
- [ ] 完成 3-5 个项目作品集
- [ ] 准备算法面试

---

## 💡 学习心得

### 核心理解
> "通过引入隐藏层，将原本数据做一次映射，映射成线性可分的状态"
> —— 2025-12-15, 学生对深度学习本质的总结

### 关键技能
1. **维度分析**：每次写矩阵运算先写维度
2. **实验验证**：理论要用代码和实验检验
3. **主动提问**：不懂就问，不要硬憋

### 学习建议
- 理论和实践结合，每学一个概念都动手实现
- 多做实验，观察超参数的影响
- 遇到 Bug 先分析维度，再看公式

---

## 📈 进度可视化

```
神经网络基础: ████████████████░░  125% (超额完成)
数学基础:     ████████████░░░░░░   60%
深度学习架构: ░░░░░░░░░░░░░░░░░░    0%
计算机视觉:   ░░░░░░░░░░░░░░░░░░    0%
NLP:         ░░░░░░░░░░░░░░░░░░    0%
训练优化:     ░░░░░░░░░░░░░░░░░░    0%
框架实现:     ░░░░░░░░░░░░░░░░░░    0%
高级主题:     ░░░░░░░░░░░░░░░░░░    0%
```

---

**下次更新**: 2025-12-16（Session 2 后）
