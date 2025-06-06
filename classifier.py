import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

data = pd.read_csv("fake_or_real_news.csv")
data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)

data = data.drop("label", axis = 1)
X,y = data['text'], data['fake']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#print(len(X_train))
#5068

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vectorized= vectorizer.fit_transform(X_train)
X_test_vectorized=vectorizer.transform(X_test)

clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)
#print(clf.score(X_test_vectorized, y_test))
# --> 0.9376479873717443
print((len(y_test) * 0.9376479873717443), " / ", len(y_test))

