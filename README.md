# LARGE-LANGUAGE-MODELS
Sentiment analysis on Twitter data using the TweetEval – Sentiment dataset.
📌 Project Overview

This project investigates sentiment analysis on Twitter data using the TweetEval – Sentiment dataset.
The goal is to compare a traditional machine learning baseline with a medium-sized Large Language Model (BERT) using different fine-tuning strategies and to evaluate their performance, efficiency, and practicality.

The study focuses on:
	•	Understanding the impact of fine-tuning
	•	Comparing full fine-tuning vs parameter-efficient LoRA
	•	Evaluating the effect of hyperparameter tuning
	•	Selecting the best model based on performance and efficiency

⸻

🧠 Task Description
	•	Task: Sentiment Classification
	•	Classes:
	•	Negative
	•	Neutral
	•	Positive

This is a multi-class text classification problem on short, informal Twitter text.

⸻

📊 Dataset

Dataset: TweetEval – Sentiment
	•	Public benchmark for Twitter NLP tasks
	•	Pre-split into train, validation, and test sets
	•	Tweets are short, informal, and slightly class-imbalanced (Neutral dominates)

📌 Due to class imbalance, macro-averaged Precision, Recall, and F1-score are used for evaluation.

⸻

🏗️ Models Implemented

1️⃣ Baseline Model
	•	TF-IDF vectorisation (uni-grams + bi-grams)
	•	Logistic Regression
	•	Serves as a reference to quantify improvements from transformer-based models

2️⃣ BERT (No Fine-Tuning)
	•	Pre-trained BERT evaluated directly on the task
	•	Demonstrates that task-specific adaptation is necessary

3️⃣ Fully Fine-Tuned BERT
	•	BERT-base model
	•	All transformer parameters updated
	•	Training and validation loss recorded
	•	Learning curves analysed for convergence and overfitting

4️⃣ LoRA Fine-Tuned BERT
	•	Parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA)
	•	Only small adapter layers trained
	•	Base BERT weights remain frozen
	•	Significantly fewer trainable parameters

⸻

⚙️ Hyperparameter Tuning
	•	Hyperparameter optimisation performed using Optuna
	•	Tuned parameters include:
	•	Learning rate
	•	Weight decay
	•	Batch size
	•	Applied to:
	•	Fully fine-tuned BERT
	•	LoRA fine-tuned BERT

This ensures a fair and optimised comparison.

⸻

📈 Evaluation Metrics

All models are evaluated on the held-out test set using:
	•	Accuracy
	•	Macro-averaged Precision
	•	Macro-averaged Recall
	•	Macro-averaged F1-score

Additional analysis includes:
	•	Learning curves (training vs validation loss)
	•	Confusion matrices
	•	Metric comparison bar charts

⸻

🏆 Key Results

Effect of Fine-Tuning
	•	Baseline model performs moderately but struggles with nuanced sentiment
	•	BERT without fine-tuning performs poorly
	•	Fine-tuned BERT models significantly outperform both baselines

Full Fine-Tuning vs LoRA
	•	Fully fine-tuned BERT achieves the highest macro F1-score
	•	LoRA fine-tuned BERT performs very close to full fine-tuning
	•	LoRA uses far fewer trainable parameters and is more computationally efficient

Learning Curve Analysis
	•	Stable reduction in training and validation loss
	•	No significant overfitting observed
	•	Performance plateaus after a small number of epochs

⸻

✅ Best Model Decision
	•	Best Performance: Fully fine-tuned BERT (after hyperparameter tuning)
	•	Best Efficiency: LoRA fine-tuned BERT (after hyperparameter tuning)
	•	Best Trade-Off: LoRA fine-tuned BERT, due to near-identical performance with much lower computational cost

⸻

⚠️ Limitations
	•	Limited hyperparameter search space
	•	Short training duration
	•	Single dataset (TweetEval only)
	•	No cross-domain or multilingual evaluation

⸻

🔮 Future Work
	•	Larger hyperparameter search
	•	Evaluate stronger transformer models (RoBERTa, DeBERTa)
	•	Explore different LoRA ranks
	•	Test generalisation on other Twitter datasets

⸻

📚 References
	1.	Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019).
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
https://arxiv.org/abs/1810.04805
	2.	Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L. and Neves, L. (2020).
TweetEval: Unified Benchmark for Tweet Classification.
https://arxiv.org/abs/2010.12421
	3.	Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W. (2021).
LoRA: Low-Rank Adaptation of Large Language Models.
https://arxiv.org/abs/2106.09685
