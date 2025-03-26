# Checked Feature Importance

coefs = model.coef_[0]

feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefs,
    'AbsCoefficient': np.abs(coefs)
})
feature_importance = feature_importance.sort_values(by='AbsCoefficient', ascending=False)
print(feature_importance)

plt.figure(figsize=(8, 6))
plt.bar(feature_importance['Feature'], feature_importance['AbsCoefficient'], color='skyblue')
plt.xlabel('Feature')
plt.ylabel('Absolute Coefficient')
plt.title('Feature Importance (Logistic Regression)')
plt.show()

# 1. Extended Feature Set:
# Added new variables such as term_numeric,int_rate_numeric, avg_fico, and credit_hist to the original features (fico_range_high, fico_range_low, annual_inc, dti, etc.).

features_2 = ['delinq_2yrs', 'open_acc', 'pub_rec', 'total_acc',
              'chargeoff_within_12_mths', 'loan_amnt', 'term',
              'int_rate', 'installment', 'annual_inc']

print(df[features_2].info())
print(df[features_2].head())

# Convert term: remove " months" and convert to int
df['term_numeric'] = df['term'].str.extract('(\d+)').astype(int)
# Convert int_rate: remove % and convert to float
df['int_rate_numeric'] = df['int_rate'].str.replace('%', '').astype(float)

print(df[['term', 'term_numeric', 'int_rate', 'int_rate_numeric']].info())

print(df[['term', 'term_numeric', 'int_rate', 'int_rate_numeric']].head())

print(df[['earliest_cr_line', 'issue_d']].info())

#2. Feature Engineering:
# Converted string columns (e.g., 'term', 'int_rate') to numeric, computed average FICO score, and calculated credit history duration.

# Feature ingineering

# Convert to datetime
df['issue_d_dt'] = pd.to_datetime(df['issue_d'], errors='coerce')
df['earliest_cr_line_dt'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')

# 1. Average FICO score
df['avg_fico'] = (df['fico_range_high'] + df['fico_range_low']) / 2

# 2. Duration of credit history
df['credit_hist'] = (df['issue_d_dt'] - df['earliest_cr_line_dt']).dt.days / 365


print(df[['avg_fico', 'credit_hist']].head())

df[['fico_range_high', 'fico_range_low', 'annual_inc', 'dti',
    'delinq_2yrs', 'open_acc', 'pub_rec', 'total_acc', 'chargeoff_within_12_mths',
    'loan_amnt', 'term_numeric', 'int_rate_numeric', 'installment', 'avg_fico', 'credit_hist']].info()

features_2 = ['fico_range_high', 'fico_range_low', 'annual_inc', 'dti',
              'delinq_2yrs', 'open_acc', 'pub_rec', 'total_acc', 'chargeoff_within_12_mths',
              'loan_amnt', 'term_numeric', 'int_rate_numeric', 'installment', 'avg_fico', 'credit_hist']

X_new = df[features_2]
y = df['loan_default']

# 3. Missing Data Handling:
# Filled NaN values with the mean of each column.

X_new = X_new.fillna(X_new.mean())
print(X_new.isnull().sum())
X_new.info()

# Split the dataset into training and testing sets
X_new_train, X_new_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
for dframe in [X_new_train, X_new_test, y_train, y_test]:
    print(f"Shape: {dframe.shape}")

# Data Scaling & Model_2 Training using the expanded feature set.

# Scale the training and testing data
scaler = StandardScaler()
X_new_train_scaled = scaler.fit_transform(X_new_train)
X_new_test_scaled = scaler.transform(X_new_test)

# Convert scaled arrays back to DataFrame for easier handling
X_new_train_scaled = pd.DataFrame(X_new_train_scaled, columns=X_new_train.columns)
X_new_test_scaled = pd.DataFrame(X_new_test_scaled, columns=X_new_test.columns)

# Train logistic regression model on the scaled training data
model_2 = LogisticRegression(random_state=42, max_iter=1000)
model_2.fit(X_new_train_scaled, y_train)

# Get predictions on the scaled test data
predictions_2 = model_2.predict(X_new_test_scaled)
print(predictions_2)

# Evaluation: Calculated accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

# ----- Cross-Validation on the entire dataset -----

# 1. Create a pipeline: scaling followed by logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42, max_iter=1000))
])

# 2. Define a 5-fold cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Obtain aggregated cross-validated predictions
y_pred_cv = cross_val_predict(pipeline, X_new, y, cv=cv)

# 4. Evaluate aggregated metrics using the cross-validated predictions
acc_cv = accuracy_score(y, y_pred_cv)
prec_cv = precision_score(y, y_pred_cv)
rec_cv = recall_score(y, y_pred_cv)
f1_cv = f1_score(y, y_pred_cv)

print("=== Cross-Val (Out-of-Fold) Metrics ===")
print(f"Accuracy:  {acc_cv:.4f}")
print(f"Precision: {prec_cv:.4f}")
print(f"Recall:    {rec_cv:.4f}")
print(f"F1-Score:  {f1_cv:.4f}")

# 5. For ROC-AUC, obtain predicted probabilities using cross-validation
y_proba_cv = cross_val_predict(pipeline, X_new, y, cv=cv, method='predict_proba')
roc_auc_cv = roc_auc_score(y, y_proba_cv[:, 1])
print(f"ROC-AUC:   {roc_auc_cv:.4f}")

# 6. Display the confusion matrix for the aggregated predictions
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_cv))

# Get predict on test with trained model_2
predictions_test = model_2.predict(X_new_test_scaled)

accuracy_test = accuracy_score(y_test, predictions_test)
precision_test = precision_score(y_test, predictions_test)
recall_test = recall_score(y_test, predictions_test)
f1_test = f1_score(y_test, predictions_test)
roc_auc_test = roc_auc_score(y_test, model_2.predict_proba(X_new_test_scaled)[:, 1])

print("=== Test Metrics ===")
print(f"Accuracy:  {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall:    {recall_test:.4f}")
print(f"F1-Score:  {f1_test:.4f}")
print(f"ROC-AUC:   {roc_auc_test:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions_test))

# Feature Importance

coefs = model_2.coef_[0]


feature_importance = pd.DataFrame({
    'Feature': features_2,
    'Coefficient': coefs,
    'AbsCoefficient': np.abs(coefs)
})
feature_importance = feature_importance.sort_values(by='AbsCoefficient', ascending=False)
print(feature_importance)

plt.figure(figsize=(15, 6))
plt.bar(feature_importance['Feature'], feature_importance['AbsCoefficient'], color='skyblue')
plt.xlabel('Feature')
plt.ylabel('Absolute Coefficient')
plt.title('Feature Importance (Logistic Regression)')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()

plt.show()

# Compute predicted probabilities (for the positive class)
y_pred_prob = model_2.predict_proba(X_new_test_scaled)[:, 1]

# Compute ROC curve and ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line (random prediction)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Additional stats
print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred_prob):.2f}')

#Applying SMOTE

sm = SMOTE(random_state=42)

# Apply SMOTE to train only
X_train_res, y_train_res = sm.fit_resample(X_new_train_scaled, y_train)

# Train logistic regression on balanced data
model_2_smote = LogisticRegression(random_state=42, max_iter=1000)
model_2_smote.fit(X_train_res, y_train_res)

# Predict on test
predictions_smote = model_2_smote.predict(X_new_test_scaled)
proba_smote = model_2_smote.predict_proba(X_new_test_scaled)[:, 1]
accuracy = accuracy_score(y_test, predictions_smote)

accuracy = accuracy_score(y_test, predictions_smote)
precision = precision_score(y_test, predictions_smote)
recall = recall_score(y_test, predictions_smote)
f1 = f1_score(y_test, predictions_smote)
roc_auc = roc_auc_score(y_test, proba_smote)

print("After SMOTE:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_smote))



# Compute predicted probabilities (for the positive class)

y_pred_prob_smote = model_2_smote.predict_proba(X_new_test_scaled)[:, 1]

# Compute ROC curve and ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random guess')  # Диагональ
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Additional stats
print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred_prob):.2f}')

coefs = model_2_smote.coef_[0]

feature_importance = pd.DataFrame({
    'Feature': features_2,
    'Coefficient': coefs,
    'AbsCoefficient': np.abs(coefs)
})


feature_importance.sort_values(by='AbsCoefficient', ascending=False, inplace=True)
print(feature_importance)


plt.figure(figsize=(15, 6))
plt.bar(feature_importance['Feature'], feature_importance['AbsCoefficient'], color='skyblue')
plt.xlabel('Feature')
plt.ylabel('Absolute Coefficient')
plt.title('Feature Importance (Logistic Regression)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Feature selection with Removal the low-importance features from the full dataset

# Choose features with AbsCoefficient < 0.09
threshold = 0.09
lowest_impact_features = feature_importance[
    feature_importance['AbsCoefficient'] < threshold
]['Feature'].tolist()

print("Features to remove (below threshold):", lowest_impact_features)

# Delete them from X_new
X_new_filtered = X_new.drop(columns=lowest_impact_features)

print("Shape before deleting:", X_new.shape)
print("Shape after deleting:", X_new_filtered.shape)

print(X_new_filtered.head())
print(X_new_filtered.info())

# Train/Test Split and Model Training

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_new_filtered, y, test_size=0.2, random_state=42
)

# Scale the training and testing data
scaler = StandardScaler()
X_train_f_scaled = scaler.fit_transform(X_train_f)
X_test_f_scaled = scaler.transform(X_test_f)

# Create and train a Logistic Regression model with balanced class weights
model_3 = LogisticRegression(
    max_iter=3000,
    class_weight='balanced',
    random_state=42
)
model_3.fit(X_train_f_scaled, y_train_f)

# Predict on the test set using the default threshold of 0.5
preds_default = model_3.predict(X_test_f_scaled)
proba_default = model_3.predict_proba(X_test_f_scaled)[:, 1]

# Calculate evaluation metrics for the default threshold
accuracy_d = accuracy_score(y_test_f, preds_default)
precision_d = precision_score(y_test_f, preds_default)
recall_d = recall_score(y_test_f, preds_default)
f1_d = f1_score(y_test_f, preds_default)
auc_d = roc_auc_score(y_test_f, proba_default)

print("=== Default Threshold (0.5) ===")
print(f"Accuracy:  {accuracy_d:.4f}")
print(f"Precision: {precision_d:.4f}")
print(f"Recall:    {recall_d:.4f}")
print(f"F1-score:  {f1_d:.4f}")
print(f"ROC-AUC:   {auc_d:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test_f, preds_default))

# Evaluate model using a custom threshold (0.3)
threshold = 0.3
preds_custom = (proba_default >= threshold).astype(int)

accuracy_c = accuracy_score(y_test_f, preds_custom)
precision_c = precision_score(y_test_f, preds_custom)
recall_c = recall_score(y_test_f, preds_custom)
f1_c = f1_score(y_test_f, preds_custom)
auc_c = roc_auc_score(y_test_f, proba_default)

print("\n=== Custom Threshold (0.3) ===")
print(f"Accuracy:  {accuracy_c:.4f}")
print(f"Precision: {precision_c:.4f}")
print(f"Recall:    {recall_c:.4f}")
print(f"F1-score:  {f1_c:.4f}")
print(f"ROC-AUC:   {auc_c:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test_f, preds_custom))

# ----------------- Cross-Validation Evaluation -----------------
# Create a pipeline that scales data and then applies Logistic Regression
pipeline_f = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42))
])

# Define a 5-fold stratified cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get aggregated predictions across the folds using cross_val_predict
y_pred_cv = cross_val_predict(pipeline_f, X_new_filtered, y, cv=cv)

# Obtain predicted probabilities (for the positive class) for ROC AUC computation
y_proba_cv = cross_val_predict(pipeline_f, X_new_filtered, y, cv=cv, method='predict_proba')[:, 1]

# Calculate cross-validation metrics
accuracy_cv = accuracy_score(y, y_pred_cv)
precision_cv = precision_score(y, y_pred_cv)
recall_cv = recall_score(y, y_pred_cv)
f1_cv = f1_score(y, y_pred_cv)
auc_cv = roc_auc_score(y, y_proba_cv)

print("\n=== Cross-Validation Metrics ===")
print(f"Accuracy:  {accuracy_cv:.4f}")
print(f"Precision: {precision_cv:.4f}")
print(f"Recall:    {recall_cv:.4f}")
print(f"F1-score:  {f1_cv:.4f}")
print(f"ROC-AUC:   {auc_cv:.4f}")
print("Confusion Matrix (CV):\n", confusion_matrix(y, y_pred_cv))

def custom_cost_loss(y_true, y_pred, fp_cost=100, fn_cost=1000):
    """
    Custom loss function to compute misclassification cost.

    Parameters:
    - y_true: array-like of true binary labels (0 or 1)
    - y_pred: array-like of predicted binary labels (0 or 1)
    - fp_cost: cost of a false positive prediction (default=100)
    - fn_cost: cost of a false negative prediction (default=1000)

    Returns:
    - average_cost: average misclassification cost per sample
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate false positives and false negatives
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate total misclassification cost
    total_cost = false_positives * fp_cost + false_negatives * fn_cost

    # Optionally, return the average cost per sample
    average_cost = total_cost / len(y_true)
    return average_cost

# a scorer for use in cross-validation or grid search with greater_is_better=False because lower cost is better
custom_cost_scorer = make_scorer(custom_cost_loss, greater_is_better=False)

#fit the pipeline on the training data
pipeline_f.fit(X_train_f, y_train_f)

# Predict on the test set
y_test_pred = pipeline_f.predict(X_test_f)

# Compute the custom cost on the test set using the custom loss function
test_cost = custom_cost_loss(y_test_f, y_test_pred)
print("Custom misclassification cost on test set:", test_cost)



