import pickle
filename = 'd_title_text_code_tag.sav'
loaded_model = pickle.load(open(filename, 'rb'))
data = ["Save classifier to disk in scikit-learn",".net","from sklearn.metrics import classification_report y_pred = nb.predict(X_test)", "14 This question already has an answer here: How to input a regex in string.replace? 7 answers I am trying to do a grab everything after the \"</html>\" tag and delete it, but my code doesn't seem to be doing anything. Does .replace() not support regex?"]
#for i in data:
result = loaded_model.predict(data)
print(result)