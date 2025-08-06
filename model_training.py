import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# ----------------------------
# Text cleaning function
# ----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ----------------------------
# Load and prepare dataset
# ----------------------------
df = pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
df = df[['reviews.text', 'reviews.rating']]
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ----------------------------
# Map ratings to sentiment labels
# ----------------------------
def map_sentiment(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df['label'] = df['reviews.rating'].apply(map_sentiment)

# ----------------------------
# Upsample all classes to 200 each
# ----------------------------
pos = df[df['label'] == 2]
neu = df[df['label'] == 1]
neg = df[df['label'] == 0]

pos_upsampled = resample(pos, replace=True, n_samples=200, random_state=42)
neu_upsampled = resample(neu, replace=True, n_samples=200, random_state=42)
neg_upsampled = resample(neg, replace=True, n_samples=200, random_state=42)

df_upsampled = pd.concat([pos_upsampled, neu_upsampled, neg_upsampled])
df_upsampled = df_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------------
# Clean the text
# ----------------------------
df_upsampled['cleaned_text'] = df_upsampled['reviews.text'].apply(clean_text)

# ----------------------------
# TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df_upsampled['cleaned_text'])
y = df_upsampled['label']

# ----------------------------
# Train Logistic Regression
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=300)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# ----------------------------
# Evaluation
# ----------------------------
print("Logistic Regression Results:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Save Model and Vectorizer
# ----------------------------
joblib.dump(lr, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
