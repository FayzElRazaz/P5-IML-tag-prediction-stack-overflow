import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from xgboost import XGBClassifier

ftags = pd.read_csv('tags.csv')
df_clean = pd.read_csv('df_clean.csv')

vectorizer = CountVectorizer(min_df=0.01)
X = vectorizer.fit_transform(df_clean["sentence_title_bow"])
vectorizer.get_feature_names_out()
df_bow_sklearn = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())

multilabel_binarizer = MultiLabelBinarizer().fit(ftags.values.astype(str))
y = multilabel_binarizer.transform(ftags.values.astype(str))
X_train, X_test, y_train, y_test = train_test_split(df_bow_sklearn, y, test_size=0.3, random_state=42)

gb_clf = OneVsRestClassifier(XGBClassifier(n_jobs = -1))
gb_clf = gb_clf.fit(X_train, y_train)

#y_pred = gb_clf.predict(X_test)
#print(jaccard_score(y_test,y_pred, average = "macro"))

pickle.dump(gb_clf, open('model.pyc','wb'))
pickle.dump(multilabel_binarizer, open('multilabel_binarizer.pyc','wb'))
pickle.dump(vectorizer, open("vectorizer.pyc", 'wb'))