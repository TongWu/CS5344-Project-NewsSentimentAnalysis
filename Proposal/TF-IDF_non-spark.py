#%%
import pandas as pd
import numpy as np

df = pd.read_csv("../ggg_sg.csv", nrows=10000)

#%%
df.head()
#%%
# Check cols type
for column in df.columns:
    print(f"Column '{column}' data types:")
    print(df[column].apply(type).value_counts())
    print("\n")

#%%
selected_cols = ['Title', 'DateTime', 'DocTone', 'ContextualText']
df_cleaned = df[selected_cols]
df_cleaned = df_cleaned.dropna()
# %%
df_cleaned["DocTone"].describe()
#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution of DocTone scores
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned["DocTone"], bins=50, kde=True)
plt.title('Distribution of DocTone Scores (Sampled Data)')
plt.xlabel('DocTone Score')
plt.ylabel('Frequency')
plt.show()
# %%
# Create sentiment label: Positive (2), Neutral (1), Negative (0)
def sentiment_label(score):
    if score > 1.9910:
        return 2
    elif score < -2.0202:
        return 0
    else:
        return 1
df_cleaned["label"] = df_cleaned['DocTone'].apply(sentiment_label)
#%%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df_cleaned["ContextualText"])

print(vectorizer.get_feature_names_out())
print(X.toarray())
print(X.shape)

y = df_cleaned["label"]
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
# %%
