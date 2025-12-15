# Deep Learning Study Repository

🚀 **My AI Learning Journey with Claude Code** 🚀

This is my personal deep learning study repository, built using the AI-powered guided learning methodology inspired by [chenran818's CFP-Study project](https://github.com/chenran818/CFP-Study). After seeing how effectively AI tutoring helped someone master complex financial planning concepts, I adapted this approach to learn deep learning from scratch.

**Current Progress**: 23% (13/56 topics mastered) after 1 study session
**Study Sessions**: 1 session (started Dec 15, 2025)
**Goal**: Master deep learning theory and practice for AI research & algorithm engineer positions

---

## 📚 What This Repository Is

This repository uses **Claude Code as an interactive Deep Learning tutor** that:
- Teaches using the **Socratic method** (asking what you know first)
- Provides concise (~200 word) explanations with practical examples
- Verifies your understanding with follow-up questions
- Adapts teaching style based on your responses
- **Tracks every learning session to personalize your study experience**
- **Never guesses** - always verifies technical details with authoritative sources

## 🎯 Learning Goals

- **Short-term** (1-2 months): Master deep learning fundamentals and PyTorch
- **Medium-term** (3-6 months): Reproduce classic papers, build portfolio projects
- **Long-term**: Conduct AI research + secure algorithm engineering position

---

## 📁 Repository Structure

```
/sessions/                    # Daily learning sessions documented
  /2025-12-15/               # Session 1: Neural network basics, backprop
    session-notes.md         # Detailed notes of what was learned
  SESSION-TEMPLATE.md        # Template for documenting sessions

/progress/                    # Single source of truth for learning progress
  dl-study-tracker.md        # Comprehensive tracker with:
                             # - All 56 deep learning topics mapped
                             # - Topics mastered (13/56 so far)
                             # - Knowledge gaps identified
                             # - Study plan with priorities

/project/                     # Hands-on coding projects
  /backpropagation/          # From-scratch neural network implementation
    example.py               # 2-layer NN solving XOR (100% accuracy!)
    compare_learning_rates.py # Hyperparameter experiments

CLAUDE.md                    # AI tutor instructions (Socratic method for DL)
README.md                    # This file
```

---

## 🧠 Deep Learning Topics Covered

### 8 Major Knowledge Domains (56 topics total)

| Domain | Weight | Progress | Status |
|--------|--------|----------|--------|
| **A. Mathematical Foundations** | 20% | 60% (3/5) | 🟡 In Progress |
| **B. Neural Network Fundamentals** | 18% | 125% (10/8) | ✅ Completed |
| **C. Deep Learning Architectures** | 16% | 0% (0/8) | ⚪ Not Started |
| **D. Computer Vision** | 12% | 0% (0/6) | ⚪ Not Started |
| **E. Natural Language Processing** | 12% | 0% (0/7) | ⚪ Not Started |
| **F. Training and Optimization** | 10% | 0% (0/8) | ⚪ Not Started |
| **G. Frameworks and Implementation** | 8% | 0% (0/7) | ⚪ Not Started |
| **H. Advanced Topics** | 4% | 0% (0/7) | ⚪ Not Started |

**Topics Mastered So Far**:
- ✅ Linear algebra for neural networks
- ✅ Calculus & chain rule (backpropagation math)
- ✅ Perceptrons and activation functions
- ✅ Forward propagation
- ✅ Loss functions (MSE)
- ✅ **Backpropagation algorithm** (complete mathematical derivation)
- ✅ Gradient descent optimization
- ✅ Multi-layer networks & feature learning
- ✅ From-scratch neural network implementation
- ✅ Matrix dimension analysis & debugging
- ✅ Hyperparameter experimentation

---

## 🚀 How to Use This Repository

### Daily Study Sessions

1. **Open Claude Code** in this repository
2. **Ask questions** about deep learning topics naturally - like talking to a tutor
3. **Answer comprehension check questions** Claude asks
4. **Code along** - implement concepts from scratch
5. After each session, Claude automatically documents:
   - What you learned
   - What you struggled with (knowledge gaps)
   - What you mastered (with confidence levels)
   - What to study next

### Example Questions to Ask

- "Explain how backpropagation works"
- "Why do we need activation functions?"
- "Help me implement a CNN from scratch"
- "What's the difference between ReLU and Sigmoid?"
- "Show me my progress and what I should focus on next"

### Review Your Progress

Check your comprehensive tracker at [progress/dl-study-tracker.md](progress/dl-study-tracker.md) to see:
- Overall learning progress (currently 23%)
- Which topics are mastered
- Current knowledge gaps
- Prioritized study plan

---

## 💡 Study Philosophy

**Guided Learning Approach** (inspired by Google Gemini's methodology):
- **Conversational and judgment-free** - safe space to explore
- **Builds on your existing knowledge** - starts where you are
- **Checks understanding before moving forward** - no blind spots
- **Adapts to your learning style** - theory or code-first
- **Focuses on deep understanding** - not just memorization
- **Evidence-based** - verifies all technical claims with authoritative sources

**Two-Step Progress Tracking**:
1. **Session Notes** - detailed record of each learning session
2. **Progress Tracker** - big picture view of overall learning journey

---

## 🎓 Key Features

### Personalized Learning
- Socratic teaching method (asks what you know first)
- Adaptive explanations based on your responses
- Hands-on coding exercises tailored to your level
- Practice problems targeting your weak areas

### Comprehensive Tracking
- Every session automatically documented
- Knowledge gaps systematically identified
- Topics mastered with confidence levels
- Progress measured across 8 domains

### Evidence-Based Approach
- All technical details verified with authoritative sources:
  - PyTorch/TensorFlow official docs
  - Research papers (arXiv, NeurIPS, ICML, CVPR)
  - Stanford CS231n/CS224n courses
  - Deep Learning Book (Goodfellow et al.)
- **No guessing on technical questions** - accuracy is essential
- Citations provided for complex concepts

---

## 📖 Recommended Learning Resources

### Core Courses (Top Priority)

#### 1. Andrew Ng's Deep Learning Specialization ⭐⭐⭐⭐⭐
**Platform**: [Coursera](https://www.coursera.org/specializations/deep-learning)

**Why Andrew Ng?**
- Clear explanations from scratch, perfect for beginners
- Detailed but not overly complex math derivations
- High-quality programming assignments (NumPy + TensorFlow)
- Complete deep learning curriculum
- Industry best practices from his experience at Google Brain and Baidu

**5 Courses**:
1. Neural Networks and Deep Learning (4 weeks)
2. Improving Deep Neural Networks (3 weeks)
3. Structuring Machine Learning Projects (2 weeks)
4. Convolutional Neural Networks (4 weeks)
5. Sequence Models (3 weeks)

**Key Projects**:
- Planar data classification
- Cat classifier
- CIFAR-10 with CNN
- YOLO object detection
- Neural style transfer
- Dinosaur name generator
- Machine translation

**My Plan**: Follow this as the main curriculum, reimplementing all assignments in PyTorch.

---

#### 2. Stanford CS231n - CNN for Visual Recognition ⭐⭐⭐⭐⭐
**Website**: [cs231n.stanford.edu](http://cs231n.stanford.edu/)

Classic computer vision course with deeper math than Andrew Ng. **Lecture 4 (Backpropagation)** covers exactly what we learned in Day 1! Outstanding assignments using PyTorch.

---

#### 3. Stanford CS224n - NLP with Deep Learning ⭐⭐⭐⭐
**Website**: [web.stanford.edu/class/cs224n/](http://web.stanford.edu/class/cs224n/)

The authoritative NLP course covering Transformers, BERT, and GPT. Will study after mastering RNNs.

---

### Video Resources

#### 3Blue1Brown - Neural Networks Series ⭐⭐⭐⭐⭐
**Link**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

Best visualizations of neural networks and backpropagation! 4 videos, ~15-20 min each. Perfect complement to Day 1-2 learning.

---

### Books

#### Deep Learning Book (Goodfellow, Bengio, Courville) ⭐⭐⭐⭐
**Free online**: [deeplearningbook.org](https://www.deeplearningbook.org/)

The "bible" of deep learning with rigorous math. Use as a reference, not for cover-to-cover reading.

#### Dive into Deep Learning (d2l.ai) ⭐⭐⭐⭐⭐
**Free online**: [d2l.ai](https://d2l.ai/)

Interactive textbook with code + theory. Supports PyTorch, TensorFlow, MXNet. Can run code directly in browser!

---

### Practice Platforms
- [Kaggle](https://www.kaggle.com/) - Competitions and datasets
- [Papers with Code](https://paperswithcode.com/) - Latest research + implementations

---

## 🛠️ Current Projects

### Project 1: From-Scratch Neural Network ✅
**Location**: `/project/backpropagation/`
- Implemented 2-layer neural network without frameworks (pure NumPy)
- Solved XOR problem with 100% accuracy
- Complete forward and backward propagation
- Experimented with different learning rates and architectures

**Key Learning**:
- Deep understanding of backpropagation math
- Matrix dimension analysis skills
- Hyperparameter tuning experience

### Upcoming Projects
- [ ] PyTorch implementation of neural network
- [ ] MNIST handwritten digit classification (target: 95%+ accuracy)
- [ ] CIFAR-10 image classification with CNN
- [ ] Text generation with RNN
- [ ] Transformer for sentiment analysis

---

## 📊 Progress Snapshot

```
Overall Progress: 23% (13/56 topics)
Study Sessions: 1
Study Time: ~2 hours

Domain Progress:
Mathematical Foundations: ████████████░░░░░░   60%
Neural Network Basics:   █████████████████░   125% (超额完成!)
CNN Architectures:       ░░░░░░░░░░░░░░░░░░    0%
RNN Architectures:       ░░░░░░░░░░░░░░░░░░    0%
Computer Vision:         ░░░░░░░░░░░░░░░░░░    0%
NLP:                     ░░░░░░░░░░░░░░░░░░    0%
Training & Optimization: ░░░░░░░░░░░░░░░░░░    0%
Advanced Topics:         ░░░░░░░░░░░░░░░░░░    0%
```

---

## 🎯 Next Steps

### Session 2 (Planned for 2025-12-16)
1. Gradient vanishing/exploding problems
2. ReLU and its variants (LeakyReLU, ELU, GELU)
3. Weight initialization strategies (Xavier, He)
4. Batch Normalization

### Weekly Goals
- **Week 1**: Complete neural network fundamentals
- **Week 2**: Start CNN (Convolutional Neural Networks)
- **Week 3-4**: RNN and LSTM
- **Week 5-7**: Transformers and Attention

---

## 🙏 Acknowledgments

This learning system is inspired by **[chenran818's CFP-Study repository](https://github.com/chenran818/CFP-Study)**, which demonstrates the power of AI-assisted learning. After seeing how effectively someone used Claude Code to pass a professional certification exam (from failure to 82% mastery in 23 sessions!), I adapted this methodology for deep learning.

**Key inspiration**:
- The Socratic teaching method
- Systematic progress tracking (session notes + overall tracker)
- Evidence-based learning (no guessing, always verify)
- Adaptive teaching based on student responses

Thank you to Chen Ran for open-sourcing this brilliant learning framework! 🙏

**Original CFP-Study repository**: [github.com/chenran818/CFP-Study](https://github.com/chenran818/CFP-Study)

---

## 🚀 Want to Use This for Your Own Deep Learning Journey?

This system is designed to be reusable! Here's how:

1. **Clone or fork this repository**
2. **Clear my study history** (start fresh):
   ```bash
   rm -rf progress/ sessions/ project/
   ```
3. **Run Claude Code** in the repository:
   ```bash
   claude-code
   ```
4. **Start learning!** Just ask your first deep learning question

The [CLAUDE.md](CLAUDE.md) file contains all the instructions for how Claude should tutor you. It works magically! ✨

---

## 📞 Connect

If you're also learning deep learning or using AI-assisted study methods, let's connect and share insights!

---

**Last Updated**: 2025-12-15
**Status**: Active learning 🔥
**Current Focus**: Neural network fundamentals & backpropagation
