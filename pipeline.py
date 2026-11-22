import re, json, time, string, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
ENGLISH_STOP_WORDS = frozenset(stopwords.words('english'))
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SEED = 42
df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label','text'], quoting=3, encoding='utf-8')
df['y'] = (df.label == 'spam').astype(int)

PUNCT = re.compile(r'[^a-z0-9\s]')
def clean(t, stop=True):
    t = PUNCT.sub(' ', t.lower())
    toks = t.split()
    if stop:
        toks = [w for w in toks if w not in ENGLISH_STOP_WORDS]
    return ' '.join(toks)

df['clean'] = df.text.map(clean)
df['clean_nostop'] = df.text.map(lambda t: clean(t, stop=False))

tr, te = train_test_split(df.index, test_size=0.2, stratify=df.y, random_state=SEED)
Xtr_txt, Xte_txt = df.loc[tr,'clean'], df.loc[te,'clean']
ytr, yte = df.loc[tr,'y'], df.loc[te,'y']

def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return dict(acc=accuracy_score(y,p), prec=precision_score(y,p), rec=recall_score(y,p), f1=f1_score(y,p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), fpr=fp/(fp+tn))

def fit_eval(model, ngram=(1,2), col='clean'):
    a, b = df.loc[tr,col], df.loc[te,col]
    pipe = make_pipeline(TfidfVectorizer(ngram_range=ngram), model)
    pipe.fit(a, ytr)
    return pipe, metrics(yte, pipe.predict(b))

R = {}
R['dataset'] = dict(total=len(df), ham=int((df.y==0).sum()), spam=int(df.y.sum()), dupes=int(df.text.duplicated().sum()),
                    train=len(tr), test=len(te), train_spam=int(ytr.sum()), test_spam=int(yte.sum()),
                    train_ham=int((ytr==0).sum()), test_ham=int((yte==0).sum()))

nb_pipe, R['nb'] = fit_eval(MultinomialNB())
svm_pipe, R['svm'] = fit_eval(LinearSVC(C=1.0, random_state=SEED))
lr_pipe, R['lr'] = fit_eval(LogisticRegression(C=1.0, max_iter=1000))

# parameter tests (spam-class F1)
R['ng11'] = fit_eval(LogisticRegression(C=1.0, max_iter=1000), ngram=(1,1))[1]
R['ng12'] = R['lr']
R['stop_on'] = R['lr']
R['stop_off'] = fit_eval(LogisticRegression(C=1.0, max_iter=1000), col='clean_nostop')[1]
R['C'] = {c: fit_eval(LogisticRegression(C=c, max_iter=1000))[1] for c in (0.1, 1, 10)}

# 5-fold CV on the training set (spam F1 + accuracy) for LR
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
R['cv_f1'] = cross_val_score(make_pipeline(TfidfVectorizer(ngram_range=(1,2)), LogisticRegression(C=1.0,max_iter=1000)), Xtr_txt, ytr, cv=skf, scoring='f1').tolist()
R['cv_acc'] = cross_val_score(make_pipeline(TfidfVectorizer(ngram_range=(1,2)), LogisticRegression(C=1.0,max_iter=1000)), Xtr_txt, ytr, cv=skf, scoring='accuracy').tolist()

# the slide-18 variant: vectoriser fitted on ALL data before the split (to quantify leakage)
vec_all = TfidfVectorizer(ngram_range=(1,2)); Xall = vec_all.fit_transform(df.clean)
m = LogisticRegression(C=1.0, max_iter=1000).fit(Xall[df.index.get_indexer(tr)], ytr)
R['lr_slide18_variant'] = metrics(yte, m.predict(Xall[df.index.get_indexer(te)]))

# risk levels
def risk(p): return 'Low' if p < 0.30 else ('Medium' if p < 0.70 else 'High')
vec, lr = lr_pipe.steps[0][1], lr_pipe.steps[1][1]
joblib.dump((vec, lr), 'spam_model.pkl')
import os; R['model_kb'] = round(os.path.getsize('spam_model.pkl')/1024, 1)
R['vocab'] = len(vec.vocabulary_)
R['train_matrix_shape'] = list(vec.transform(Xtr_txt).shape)
R['nnz_per_doc'] = float(vec.transform(Xtr_txt).getnnz(axis=1).mean())

# test-set risk distribution
pt = lr_pipe.predict_proba(Xte_txt)[:,1]
R['risk_dist'] = {k: int(sum(1 for p in pt if risk(p)==k)) for k in ('Low','Medium','High')}
R['risk_spam_in_high'] = int(sum(1 for p,y in zip(pt,yte) if risk(p)=='High' and y==1))

# top features
names = np.array(vec.get_feature_names_out()); co = lr.coef_[0]
R['top_spam'] = [(names[i], float(co[i])) for i in np.argsort(-co)[:10]]
R['top_ham']  = [(names[i], float(co[i])) for i in np.argsort(co)[:10]]

# slide-19 sample messages
s_spam = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
s_ham = "Ok lar... Joking wif u oni..."
R['slide19'] = {}
for k, t in (('spam', s_spam), ('ham', s_ham)):
    p = float(lr.predict_proba(vec.transform([clean(t)]))[0][1])
    R['slide19'][k] = dict(p=p, risk=risk(p), in_train=bool(t in set(df.loc[tr,'text'])), in_test=bool(t in set(df.loc[te,'text'])))

# latency: end-to-end per SMS (clean + vectorise + predict + risk), 500 calls after warm-up
msgs = df.loc[te,'text'].sample(500, random_state=SEED).tolist()
def predict(t):
    p = float(lr.predict_proba(vec.transform([clean(t)]))[0][1]); return ('SPAM' if p>=0.5 else 'SAFE', p, risk(p))
for t in msgs[:20]: predict(t)
t0 = time.perf_counter()
for t in msgs: predict(t)
R['latency_inproc_s'] = (time.perf_counter()-t0)/len(msgs)

# save preprocessing examples + 50-message test set for the API page
rng = np.random.RandomState(SEED)
sp = df.loc[te][df.loc[te,'y']==1].sample(25, random_state=SEED); hm = df.loc[te][df.loc[te,'y']==0].sample(25, random_state=SEED)
t50 = pd.concat([sp, hm]).sample(frac=1, random_state=SEED)
t50[['text','label']].to_json('test50.json', orient='records')
df.loc[tr[:0]]  # noop
df.to_pickle('df.pkl'); pd.Series(list(tr)).to_pickle('tr.pkl'); pd.Series(list(te)).to_pickle('te.pkl')
json.dump(R, open('results.json','w'), indent=1, default=float)

f = lambda m: f"acc {m['acc']*100:.2f} | P {m['prec']*100:.2f} | R {m['rec']*100:.2f} | F1 {m['f1']*100:.2f} | TN {m['tn']} FP {m['fp']} FN {m['fn']} TP {m['tp']} | FPR {m['fpr']*100:.2f}%"
for k in ('nb','svm','lr','ng11','stop_off','lr_slide18_variant'): print(k, f(R[k]))
for c,m in R['C'].items(): print('C',c,f(m))
print('CV f1', np.round(R['cv_f1'],4), 'CV acc', np.round(R['cv_acc'],4))
print({k:R[k] for k in ('dataset','vocab','train_matrix_shape','model_kb','latency_inproc_s','risk_dist','risk_spam_in_high','slide19')})
print(R['top_spam'][:5], R['top_ham'][:5])
