```markdown
# Author : Mohamed Elfaki
# IMDB Sentiment Analysis - RNN in PyTorch

This is a sentiment analysis project trained on IMDB movie reviews. It uses a word-level RNN built with PyTorch to classify each review as either positive or negative.

Originally, I built an RNN from scratch using NumPy to get a deeper understanding of how things work under the hood. That version reached about 51% accuracy - barely better than random guessing. After switching to PyTorch and implementing an LSTM-based architecture, I hit around 84.7% accuracy.

This was part of my learning journey in NLP and PyTorch — and it gave me a much stronger grasp of embeddings, LSTMs, and training pipelines.

---

## Highlights
- Word-level tokenization and vocabulary limited to top 5000 words
- Embedding layer + LSTM built with PyTorch
- Binary classification using BCEWithLogitsLoss
- Optimized using Adam
- Clean, padded input sequences and manual batching

---

## Accuracy
| Model           | Accuracy |
|------------------|----------|
| NumPy RNN (scratch) | ~51%     |
| PyTorch LSTM        | ~84.7%     |

**Failed NumPy version:** [See scrapped attempt](https://github.com/SyntaxNomad/Sentiment-RNNbyscratch)

---

## Run it yourself
```bash
git clone https://github.com/SyntaxNomad/PyTorchRNN
cd PyTorchRNN
pip install -r requirements.txt
python main.py
```

**Built during AI internship at Cloud Solutions (July-August 2024)**
```
