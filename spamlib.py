import re, joblib
from nltk.corpus import stopwords
STOP = frozenset(stopwords.words('english'))
_P = re.compile(r'[^a-z0-9\s]')
def clean_tokens(t):
    return [w for w in _P.sub(' ', t.lower()).split() if w not in STOP]
def clean(t): return ' '.join(clean_tokens(t))
def risk(p): return 'Low' if p < 0.30 else ('Medium' if p < 0.70 else 'High')
vec, lr = joblib.load('spam_model.pkl')
def predict(text):
    p = float(lr.predict_proba(vec.transform([clean(text)]))[0][1])
    return {'label': 'SPAM' if p >= 0.5 else 'SAFE', 'probability': round(p, 4), 'risk': risk(p)}
