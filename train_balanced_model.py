# train_balanced_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import joblib
import os

# 1️⃣ Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=["label", "message"])
print("Original class distribution:\n", df['label'].value_counts(), "\n")

# 2️⃣ Balance dataset (upsample spam)
df_majority = df[df.label == 'ham']
df_minority = df[df.label == 'spam']

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
print("Balanced class distribution:\n", df_balanced['label'].value_counts(), "\n")

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['message'], df_balanced['label'],
    test_size=0.2, random_state=42
)

# 4️⃣ TF-IDF + Naive Bayes pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB(alpha=0.1))
])

# 5️⃣ Train model
model.fit(X_train, y_train)

# 6️⃣ Evaluate
preds = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

# 7️⃣ Save improved model
joblib.dump(model, "sms_spam_model.joblib")
print("\n🎉 Model retrained and saved as sms_spam_model.joblib")
print(f"Saved at: {os.getcwd()}\\sms_spam_model.joblib")
