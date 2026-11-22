import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
import json, os, time
import pandas as pd, numpy as np
from flask import Flask, request, jsonify, render_template_string
from spamlib import predict, clean, clean_tokens, risk, vec, lr

app = Flask(__name__)
R = json.load(open('results.json'))
df = pd.read_pickle('df.pkl'); tr = pd.read_pickle('tr.pkl').tolist(); te = pd.read_pickle('te.pkl').tolist()

@app.post('/predict')
def api_predict():
    msg = (request.get_json(silent=True) or {}).get('message', '')
    return jsonify(predict(msg))

# ---------- shared look ----------
BASE = """<!doctype html><html><head><meta charset="utf-8"><title>{{title}}</title><style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1280px;height:720px;overflow:hidden;font-family:Arial,'Liberation Sans',sans-serif;color:#262626;background:#dee1e6}
.chrome{height:64px;background:#dee1e6}
.tabs{height:28px;display:flex;align-items:flex-end;padding-left:14px;gap:8px}
.dots{display:flex;gap:7px;align-self:center;margin-right:12px}.dots i{width:11px;height:11px;border-radius:50%;display:block}
.tab{background:#fff;border-radius:8px 8px 0 0;height:24px;padding:0 16px;font-size:11.5px;display:flex;align-items:center;color:#333;width:250px}
.tab b{display:inline-block;width:9px;height:9px;background:#c00000;border-radius:2px;margin-right:8px}
.bar{height:36px;background:#fff;display:flex;align-items:center;padding:0 14px;gap:14px;border-bottom:1px solid #d0d3d8;color:#666;font-size:15px}
.url{flex:1;height:24px;border-radius:12px;background:#eef0f3;display:flex;align-items:center;padding:0 14px;font-size:12.5px;color:#444}
.url span{color:#888;margin-right:1px}
.page{height:656px;background:#f3f4f6;overflow:hidden}
.hdr{height:50px;background:#262626;color:#fff;display:flex;align-items:center;padding:0 26px;justify-content:space-between}
.hdr h1{font-size:19px;letter-spacing:.2px}.hdr h1 em{font-style:normal;color:#ff6b6b}
.nav{display:flex;gap:6px}.nav a{font-size:12.5px;color:#cfcfcf;padding:7px 13px;border-radius:6px;text-decoration:none}
.nav a.on{background:#c00000;color:#fff;font-weight:bold}
.body{padding:16px 26px;height:606px}
h2{font-size:15px;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.card{background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.12);padding:14px 16px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:14px}
.stat .k{font-size:11.5px;color:#777;text-transform:uppercase;letter-spacing:.5px}
.stat .v{font-size:27px;font-weight:bold;margin-top:2px}.stat .s{font-size:12px;color:#777;margin-top:1px}
.red{color:#c00000}.green{color:#2e7d32}.amber{color:#b26a00}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{background:#262626;color:#fff;text-align:left;padding:7px 9px;font-weight:bold}
td{padding:7px 9px;border-bottom:1px solid #ececec;vertical-align:top}
.chip{display:inline-block;padding:2px 9px;border-radius:10px;color:#fff;font-size:11px;font-weight:bold}
.chip.ham,.chip.SAFE,.chip.Low{background:#2e7d32}.chip.spam,.chip.SPAM,.chip.High{background:#c00000}.chip.Medium{background:#e08a00}
.tok{display:inline-block;background:#eef0f3;border-radius:4px;padding:1px 6px;margin:1px 2px 1px 0;font-size:11px;font-family:'DejaVu Sans Mono',monospace}
.ok{color:#2e7d32;font-weight:bold}
code,pre{font-family:'DejaVu Sans Mono',monospace}
</style></head><body>
<div class="chrome"><div class="tabs"><div class="dots"><i style="background:#ff5f57"></i><i style="background:#febc2e"></i><i style="background:#28c840"></i></div><div class="tab"><b></b>{{title}}</div></div>
<div class="bar"><span>&#8592;</span><span>&#8594;</span><span>&#8635;</span><div class="url"><span>http://</span>127.0.0.1:5000/module{{n}}</div></div></div>
<div class="page"><div class="hdr"><h1>SMS Spam <em>Detector</em> &nbsp;&middot;&nbsp; {{head|safe}}</h1>
<div class="nav">{% for i,l in [(1,'1 Preprocessing'),(2,'2 Features &amp; Model'),(3,'3 Risk &amp; API'),(4,'4 Android App')] %}<a class="{{'on' if i==n else ''}}">{{l|safe}}</a>{% endfor %}</div></div>
<div class="body">{{body|safe}}</div></div></body></html>"""

def page(n, title, head, body):
    return render_template_string(BASE, n=n, title=title, head=head, body=body)
fmt = lambda x: f"{x:,}"
pc = lambda x, d=2: f"{x*100:.{d}f}%"

# ---------- Module 1 ----------
@app.get('/module1')
def m1():
    d = R['dataset']
    raw_words = df.text.str.split().str.len().mean()
    tok_n = df.text.map(lambda t: len(clean_tokens(t))).mean()
    rows = ''
    for i in range(6):
        r = df.iloc[i]; toks = clean_tokens(r.text)
        raw = r.text if len(r.text) <= 78 else r.text[:75] + '...'
        chips = ''.join(f'<span class="tok">{t}</span>' for t in toks[:9]) + ('<span class="tok">&hellip;</span>' if len(toks) > 9 else '')
        rows += f'<tr><td><span class="chip {r.label}">{r.label}</span></td><td style="width:36%">{raw}</td><td>{chips}</td></tr>'
    def bar(name, ham, spam):
        tot = ham + spam; sp = spam / tot * 100
        return (f'<div style="margin-bottom:11px"><div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px"><b>{name}</b><span>{fmt(tot)} msgs &middot; spam {sp:.1f}%</span></div>'
                f'<div style="display:flex;height:20px;border-radius:5px;overflow:hidden;font-size:10.5px;color:#fff;font-weight:bold"><div style="width:{100-sp:.2f}%;background:#2e7d32;padding:3px 7px">ham {fmt(ham)}</div><div style="width:{sp:.2f}%;background:#c00000;padding:3px 4px">{spam}</div></div></div>')
    bars = bar('Full dataset', d['ham'], d['spam']) + bar('Train (80%)', d['train_ham'], d['train_spam']) + bar('Test (20%)', d['test_ham'], d['test_spam'])
    steps = [('Load dataset', f"{fmt(d['total'])} rows, 2 columns (label, text)"), ('Clean text', 'lowercase, punctuation removed, NLTK stop-words dropped'),
             ('Tokenise', f'avg {raw_words:.1f} words &rarr; {tok_n:.1f} tokens per message'), ('Stratified split', f"80 / 20, seed 42 &rarr; {fmt(d['train'])} train / {fmt(d['test'])} test")]
    st = ''.join(f'<div style="display:flex;gap:9px;font-size:12.5px;margin-bottom:7px"><span class="ok">&#10003;</span><div><b>{a}</b><div style="color:#666;font-size:11.5px">{b}</div></div></div>' for a, b in steps)
    body = f"""<div class="stats">
<div class="card stat"><div class="k">Total messages</div><div class="v">{fmt(d['total'])}</div><div class="s">UCI SMS Spam Collection</div></div>
<div class="card stat"><div class="k">Ham (SAFE)</div><div class="v green">{fmt(d['ham'])}</div><div class="s">{d['ham']/d['total']*100:.1f}% of dataset</div></div>
<div class="card stat"><div class="k">Spam</div><div class="v red">{fmt(d['spam'])}</div><div class="s">{d['spam']/d['total']*100:.1f}% of dataset</div></div>
<div class="card stat"><div class="k">Train / Test</div><div class="v">{fmt(d['train'])} / {fmt(d['test'])}</div><div class="s">stratified 80 / 20 split</div></div></div>
<div style="display:grid;grid-template-columns:1.45fr 1fr;gap:14px">
<div class="card"><h2>Raw SMS &rarr; cleaned tokens <span style="font-weight:normal;color:#777;font-size:12px">(first 6 rows)</span></h2>
<table><tr><th>Label</th><th>Raw message</th><th>Cleaned tokens</th></tr>{rows}</table></div>
<div style="display:flex;flex-direction:column;gap:14px"><div class="card"><h2>Class ratio preserved by the split</h2>{bars}</div>
<div class="card"><h2>Pipeline status</h2>{st}</div></div></div>"""
    return page(1, 'Module 1 - Data Preprocessing', 'Module 1: Data Preprocessing', body)

# ---------- Module 2 ----------
@app.get('/module2')
def m2():
    cv = np.mean(R['cv_f1']) * 100; cvs = np.std(R['cv_f1']) * 100
    def row(name, m, hl=False):
        s = ' style="background:#fdecec;font-weight:bold"' if hl else ''
        return f"<tr{s}><td>{name}</td><td>{pc(m['acc'])}</td><td>{pc(m['prec'])}</td><td>{pc(m['rec'])}</td><td>{pc(m['f1'])}</td></tr>"
    tbl = row('Naive Bayes', R['nb']) + row('SVM (linear)', R['svm']) + row('Logistic Regression (proposed)', R['lr'], True)
    def feats(lst, color, sign):
        mx = max(abs(w) for _, w in lst[:8])
        return ''.join(f'<div style="display:flex;align-items:center;gap:8px;font-size:12px;margin-bottom:5px"><span style="width:62px;text-align:right;font-family:\'DejaVu Sans Mono\',monospace">{n}</span><div style="height:13px;width:{abs(w)/mx*150:.0f}px;background:{color};border-radius:3px"></div><span style="color:#666">{w:+.2f}</span></div>' for n, w in lst[:8])
    m = R['lr']
    body = f"""<div class="stats">
<div class="card stat"><div class="k">TF-IDF vocabulary</div><div class="v">{fmt(R['vocab'])}</div><div class="s">1&ndash;2 gram features</div></div>
<div class="card stat"><div class="k">Training matrix</div><div class="v">{fmt(R['train_matrix_shape'][0])} &times; {fmt(R['train_matrix_shape'][1])}</div><div class="s">sparse, ~{R['nnz_per_doc']:.0f} non-zeros / message</div></div>
<div class="card stat"><div class="k">5-fold CV F1 (LR)</div><div class="v">{cv:.1f}%</div><div class="s">&plusmn; {cvs:.1f} on training set</div></div>
<div class="card stat"><div class="k">Saved model</div><div class="v" style="font-size:22px;margin-top:7px">spam_model.pkl</div><div class="s"><span class="ok">&#10003;</span> {R['model_kb']:,.0f} KB (joblib)</div></div></div>
<div style="display:grid;grid-template-columns:1.25fr 1fr;gap:14px">
<div style="display:flex;flex-direction:column;gap:14px"><div class="card"><h2>Model comparison &mdash; same 80/20 split, test set ({fmt(R['dataset']['test'])} msgs)</h2>
<table><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{tbl}</table>
<div style="font-size:11px;color:#777;margin-top:7px">Precision, recall and F1 are for the spam class.</div></div>
<div class="card"><h2>Confusion matrix &mdash; Logistic Regression</h2>
<div style="display:grid;grid-template-columns:90px 1fr 1fr;gap:6px;font-size:12px;text-align:center;align-items:center">
<div></div><b>Pred SAFE</b><b>Pred SPAM</b>
<b style="text-align:right">Actual SAFE</b><div style="background:#262626;color:#fff;border-radius:6px;padding:9px"><div style="font-size:11px">TN</div><div style="font-size:20px;font-weight:bold">{m['tn']}</div></div><div style="background:#eef0f3;border-radius:6px;padding:9px"><div style="font-size:11px">FP</div><div style="font-size:20px;font-weight:bold">{m['fp']}</div></div>
<b style="text-align:right">Actual SPAM</b><div style="background:#eef0f3;border-radius:6px;padding:9px"><div style="font-size:11px">FN</div><div style="font-size:20px;font-weight:bold">{m['fn']}</div></div><div style="background:#262626;color:#fff;border-radius:6px;padding:9px"><div style="font-size:11px">TP</div><div style="font-size:20px;font-weight:bold">{m['tp']}</div></div></div></div></div>
<div class="card"><h2>Top LR features (log-odds weight)</h2>
<div style="font-size:11.5px;color:#c00000;font-weight:bold;margin-bottom:6px">&uarr; pushes towards SPAM</div>{feats(R['top_spam'], '#c00000', 1)}
<div style="font-size:11.5px;color:#2e7d32;font-weight:bold;margin:12px 0 6px">&darr; pushes towards SAFE</div>{feats(R['top_ham'], '#2e7d32', -1)}</div></div>"""
    return page(2, 'Module 2 - Features & Model', 'Module 2: Features &amp; Classification', body)

# ---------- Module 3 ----------
def band_stats():
    Xte = vec.transform(df.loc[te, 'text'].map(clean)); p = lr.predict_proba(Xte)[:, 1]; y = df.loc[te, 'y'].values
    out = {}
    for k in ('Low', 'Medium', 'High'):
        mask = np.array([risk(v) == k for v in p]); out[k] = (int(mask.sum()), int(y[mask].sum()))
    return out

@app.get('/module3')
def m3():
    t50 = json.load(open('api_test.json'))
    s = t50['summary']; demo = t50['demo']
    bands = band_stats(); rng = {'Low': 'P &lt; 30%', 'Medium': '30% &le; P &lt; 70%', 'High': 'P &ge; 70%'}
    brow = ''.join(f"<tr><td><span class='chip {k}'>{k}</span></td><td>{rng[k]}</td><td>{fmt(n)}</td><td>{sp}</td></tr>" for k, (n, sp) in bands.items())
    calls = ''.join(f"<tr><td style='max-width:230px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{c['text'][:44]}{'&hellip;' if len(c['text'])>44 else ''}</td><td><span class='chip {c['label']}'>{c['label']}</span></td><td>{c['probability']*100:.1f}%</td><td><span class='chip {c['risk']}'>{c['risk']}</span></td><td>{c['ms']:.1f} ms</td></tr>" for c in t50['calls'][:5])
    req = json.dumps({'message': demo['text']}, indent=2, ensure_ascii=False)
    resp = json.dumps(demo['response'], indent=2, ensure_ascii=False)
    body = f"""<div style="display:grid;grid-template-columns:1.05fr 1fr;gap:14px">
<div style="display:flex;flex-direction:column;gap:14px">
<div class="card"><h2><span class="chip High" style="background:#1565c0">POST</span> <code style="font-size:13px">/predict</code><span style="margin-left:auto;font-weight:normal;font-size:11.5px;color:#777">Content-Type: application/json</span></h2>
<pre style="background:#1e1e1e;color:#d4d4d4;border-radius:7px;padding:12px;font-size:11.5px;line-height:1.5;white-space:pre-wrap">{req}</pre></div>
<div class="card" style="flex:1"><h2>Response <span class="chip Low" style="background:#2e7d32">{demo['status']} OK</span><span style="margin-left:auto;font-weight:normal;font-size:11.5px;color:#777">{demo['ms']:.1f} ms</span></h2>
<pre style="background:#1e1e1e;color:#d4d4d4;border-radius:7px;padding:12px;font-size:13px;line-height:1.55">{resp}</pre>
<div style="font-size:11.5px;color:#666;margin-top:8px">Flow: receive JSON &rarr; clean + vectorise &rarr; predict P(spam) &rarr; assign risk &rarr; reply</div></div></div>
<div style="display:flex;flex-direction:column;gap:14px">
<div class="card"><h2>Risk mapping (P(spam) &ge; 0.5 &rarr; SPAM) &mdash; test set</h2>
<table><tr><th>Risk</th><th>Probability band</th><th>Messages</th><th>Actual spam</th></tr>{brow}</table></div>
<div class="card"><h2>Batch test &mdash; {s['n']} messages sent to POST /predict</h2>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center">
<div><div style="font-size:21px;font-weight:bold" class="green">{s['ok']}/{s['n']}</div><div style="font-size:11px;color:#777">HTTP 200</div></div>
<div><div style="font-size:21px;font-weight:bold">{s['correct']}/{s['n']}</div><div style="font-size:11px;color:#777">correct labels</div></div>
<div><div style="font-size:21px;font-weight:bold">{s['mean_ms']:.1f} ms</div><div style="font-size:11px;color:#777">mean latency</div></div>
<div><div style="font-size:21px;font-weight:bold">{s['max_ms']:.1f} ms</div><div style="font-size:11px;color:#777">max latency</div></div></div></div>
<div class="card" style="flex:1"><h2>Recent requests</h2><table><tr><th>Message</th><th>Label</th><th>P(spam)</th><th>Risk</th><th>Time</th></tr>{calls}</table></div></div></div>"""
    return page(3, 'Module 3 - Risk & Backend API', 'Module 3: Risk Assessment &amp; Backend API', body)

# ---------- Module 4 ----------
PHONE = """<div style="width:268px;height:468px;background:#fff;border-radius:26px;border:7px solid #1b1b1b;overflow:hidden;position:relative;box-shadow:0 6px 18px rgba(0,0,0,.25)">
<div style="height:24px;background:#f4f4f4;display:flex;justify-content:space-between;padding:5px 14px;font-size:11px"><b>10:01</b><span>5G &nbsp;86%</span></div>
<div style="background:#c00000;color:#fff;font-weight:bold;font-size:15px;padding:16px 14px">SMS Spam Detector</div>
<div style="padding:10px 12px"><div style="font-size:10.5px;color:#777;margin-bottom:5px">Paste or type an SMS message</div>
<div style="height:112px;border:1.5px solid #c9c9c9;border-radius:9px;background:#fafafa;padding:8px 9px;font-size:12px;line-height:1.4;color:{tc}">{msg}</div>
<div style="margin:11px 0;background:#262626;color:#fff;text-align:center;font-weight:bold;font-size:12.5px;padding:11px;border-radius:9px">CHECK MESSAGE</div>{result}</div></div>"""
RESULT = """<div style="border:2px solid {c};border-radius:11px;padding:10px 11px;background:#f7f7f7"><span style="background:{c};color:#fff;font-weight:bold;font-size:13px;padding:5px 15px;border-radius:7px;display:inline-block">{label}</span>
<div style="font-size:10.5px;color:#777;margin-top:9px">Spam probability</div><div style="font-size:25px;font-weight:bold">{p:.1f}%</div>
<div style="font-size:10.5px;color:#777;margin-top:2px">Risk level</div><div style="font-size:15px;font-weight:bold;color:{c}">{risk}</div></div>"""

@app.get('/module4')
def m4():
    demos = json.load(open('demo_msgs.json'))
    cols = {'High': '#c00000', 'Medium': '#e08a00', 'Low': '#2e7d32'}
    phones = [(PHONE.format(msg='Type your message here...', tc='#9a9a9a', result=''), '1  Enter SMS')]
    for i, d in enumerate(demos):
        r = predict(d['text']); c = cols[r['risk']]
        res = RESULT.format(c=c, label=r['label'], p=r['probability'] * 100, risk=r['risk'])
        phones.append((PHONE.format(msg=d['text'], tc='#222', result=res), f"{i+2}  {r['label']} &middot; {r['risk']} risk"))
    cells = ''.join(f'<div style="display:flex;flex-direction:column;align-items:center;gap:9px">{p}<div style="font-size:13px;font-weight:bold">{c}</div></div>' for p, c in phones)
    body = f"""<div style="text-align:center;font-size:12.5px;color:#666;margin:-2px 0 12px">Android app screens &mdash; web preview. Each result is returned live by <code>POST /predict</code>.</div>
<div style="display:flex;justify-content:center;gap:28px">{cells}</div>"""
    return page(4, 'Module 4 - Android App', 'Module 4: Android Mobile App', body)

if __name__ == '__main__':
    app.run(port=5000)
