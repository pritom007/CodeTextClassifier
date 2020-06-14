import matplotlib as matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pickle
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier


# Data processing

file_dir = "...\\Web search and mining\\"
df = pd.read_csv(file_dir + 'code_text_title_tag.csv', encoding="ISO-8859-1")
df = df[pd.notnull(df['tags'])]
print("First 10 data")
print(df.head(10))

my_tags = ['title', 'title_text', 'code', 'tag']

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = str(text)  # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    return text


df['post'] = df['post'].apply(clean_text)

X = df.post
y = df.tags
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

# Algorithm implementation
dnn = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MLPClassifier(hidden_layer_sizes=3, activation='logistic', solver='adam', alpha=0.001, shuffle=True))
                ])
print("Started Training")
dnn.fit(X_train, y_train)
print("Started Predicting")
y_pred = dnn.predict(X_test)
print('accuracy %s' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=my_tags))
print(y_pred)
print(confusion_matrix(y_test, y_pred, labels=my_tags))

filename = 'title_text_code_tag.sav'
pickle.dump(dnn, open(filename, 'wb'))
report = open("report.txt", 'w')
report.write(classification_report(y_test, y_pred, target_names=my_tags) + "\n")
report.write('accuracy %s' % accuracy_score(y_test, y_pred) + "\n")
report.write(str(confusion_matrix(y_test, y_pred, labels=my_tags)))
report.close()
data = ["Also worked for me, ISO 8859-1 is going to save a lot, hahaha, mainly if using Speech Recognition API's"]
print(dnn.predict(data))
