This project includes extensive feature engineering, data scaling, model training, evaluation, and advanced techniques for handling class imbalance and optimizing misclassification costs. Key components of the project are:

	•	Data Preprocessing & Feature Engineering: Conversion of categorical variables (e.g., term, interest rate) into numeric formats, calculation of additional features like average FICO score and credit history duration, and handling of missing data.
	•	Model Training & Evaluation: Training logistic regression models on both the original and extended feature sets, evaluating performance using accuracy, precision, recall, F1-score, and ROC-AUC, and visualizing the results through feature importance bar charts and ROC curves.
	•	Handling Imbalanced Data: Application of SMOTE to balance the training data, followed by retraining and evaluation to improve model robustness.
	•	Feature Selection & Custom Cost Function: Removal of low-impact features based on their importance scores and implementation of a custom misclassification cost function to assess the financial impact of prediction errors.
