# Let's extend the experiment to the real Credit Card Fraud dataset (<1% positives)

# We'll train 5 Decision Trees of increasing complexity and compare ROC AUC vs Average Precision.

# We'll include the dashed baseline (precision = positive prevalence) in PR curves.



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve



# 1) Load dataset

url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"

df = pd.read_csv(url)



X, y = df.drop("Class", axis=1), df["Class"]

X_train, X_test, y_train, y_test = train_test_split(

    X, y, stratify=y, test_size=0.3, random_state=42

)



# Positive class prevalence

pos_rate = y_test.mean()



# 2) Define models of increasing complexity

models = {

    "Tree depth=2": make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=2, random_state=42)),

    "Tree depth=4": make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=4, random_state=42)),

    "Tree depth=6": make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=6, random_state=42)),

    "Tree depth=8": make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=8, random_state=42)),

    "Tree depth=12": make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=12, random_state=42)),

}



# 3) Train and collect metrics

records = []

roc_curves = {}

pr_curves = {}



for name, model in models.items():

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)

    ap = average_precision_score(y_test, probs)



    fpr, tpr, _ = roc_curve(y_test, probs)

    prec, rec, _ = precision_recall_curve(y_test, probs)



    roc_curves[name] = (fpr, tpr)

    pr_curves[name] = (rec, prec)



    records.append({"model": name, "ROC_AUC": auc, "Average_Precision": ap})



metrics_df = pd.DataFrame(records).sort_values("model").reset_index(drop=True)



# 4) Plot ROC Curves

plt.figure(figsize=(6,5))

for name, (fpr, tpr) in roc_curves.items():

    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curves – Credit Card Fraud (<1% Positives)")

plt.legend()

plt.tight_layout()

plt.show()



# 5) Plot Precision–Recall Curves with baseline dashed line

plt.figure(figsize=(6,5))

for name, (rec, prec) in pr_curves.items():

    plt.plot(rec, prec, label=name)

plt.hlines(y=pos_rate, xmin=0, xmax=1, linestyles="--", color="gray", label=f"Baseline ({pos_rate:.3f})")

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision–Recall Curves – Credit Card Fraud (<1% Positives)")

plt.legend()

plt.tight_layout()

plt.show()



# 6) Display table of metrics

#from caas_jupyter_tools import display_dataframe_to_user
#
#display_dataframe_to_user("ROC AUC vs Average Precision – Credit Card Fraud (<1%)", metrics_df)



# Save CSV for reference

csv_path = "credit_fraud_auc_ap.csv"

metrics_df.to_csv(csv_path, index=False)



metrics_df

