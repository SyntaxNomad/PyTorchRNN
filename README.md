 IMDB Sentiment Analysis with PyTorch RNN

This is a project where I built a sentiment analysis model for IMDB movie reviews using an RNN in PyTorch. The model predicts whether a review is positive or negative, and it gets around 84% accuracy on the test set.

  Background

Before using PyTorch, I actually built a basic RNN from scratch with just NumPy. That version taught me a lot — even though it only got ~51% accuracy. After that, I recreated the logic in PyTorch to make it more scalable, cleaner, and efficient.

  What This Does
	•	Loads IMDB movie reviews
	•	Preprocesses the data:
	•	Tokenization
	•	Padding
	•	Builds vocabulary (top 5000 words)
	•	Uses a custom PyTorch RNN with LSTM + Embedding
	•	Trains using BCEWithLogitsLoss and Adam optimizer
	•	Outputs sentiment predictions (positive / negative)

  Accuracy
	•	PyTorch model: ~84% accuracy
	•	From-scratch NumPy model: ~51% 

  Why I Did This

I wanted to:
	•	Learn RNNs properly, not just use them
	•	Practice PyTorch by implementing everything manually
	•	Understand how embeddings, padding, and vocab all connect
	•	See the improvement from scratch code to framework usage

  Tools Used
	•	Python
	•	PyTorch
	•	NumPy (for scratch version)
	•	IMDB Dataset
	•	Matplotlib (for some optional plots)

  Should You Try This?

If you’re learning PyTorch, RNNs, or NLP — this is a good place to start. It walks through all the steps clearly, and comparing both versions gives you insight into what PyTorch automates under the hood.
