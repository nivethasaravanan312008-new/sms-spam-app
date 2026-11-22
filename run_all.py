import json, subprocess, time, sys, requests, numpy as np, pandas as pd
from spamlib import predict

# ---- demo messages (held-out test set, short, one per risk band) ----
df=pd.read_pickle('df.pkl'); te=pd.read_pickle('te.pkl').tolist(); t=df.loc[te].copy()
t['r']=t.text.map(predict); t['p']=t.r.map(lambda r:r['probability']); t['risk']=t.r.map(lambda r:r['risk']); t['n']=t.text.str.len()
s=t[t.n.between(55,125)]
hi=s[(s.y==1)&(s.risk=='High')].sort_values('p',ascending=False); me=s[(s.y==1)&(s.risk=='Medium')].sort_values('p'); lo=s[(s.y==0)&(s.risk=='Low')]
hi=hi.iloc[min(1,len(hi)-1)]; me=me.iloc[len(me)//2]; lo=lo.sample(1,random_state=7).iloc[0]
demo=[dict(text=x.text,label=x.label) for x in (hi,me,lo)]
json.dump(demo,open('demo_msgs.json','w'),indent=1)

# ---- start the real Flask server ----
srv=subprocess.Popen([sys.executable,'app.py'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for _ in range(40):
    try: requests.get('http://127.0.0.1:5000/module1',timeout=1); break
    except Exception: time.sleep(0.5)

# ---- 50-message API test over HTTP ----
msgs=json.load(open('test50.json')); S=requests.Session()
for m in msgs[:5]: S.post('http://127.0.0.1:5000/predict',json={'message':m['text']})   # warm-up (not counted)
calls=[]
for m in msgs:
    t0=time.perf_counter(); r=S.post('http://127.0.0.1:5000/predict',json={'message':m['text']}); ms=(time.perf_counter()-t0)*1000
    j=r.json(); calls.append(dict(text=m['text'],truth=m['label'],status=r.status_code,ms=ms,**j))
ok=sum(c['status']==200 for c in calls); correct=sum((c['label']=='SPAM')==(c['truth']=='spam') for c in calls)
dt=demo[0]; t0=time.perf_counter(); r=S.post('http://127.0.0.1:5000/predict',json={'message':dt['text']}); dms=(time.perf_counter()-t0)*1000
out=dict(summary=dict(n=len(calls),ok=ok,correct=correct,mean_ms=float(np.mean([c['ms'] for c in calls])),max_ms=float(np.max([c['ms'] for c in calls])),median_ms=float(np.median([c['ms'] for c in calls]))),
         demo=dict(text=dt['text'],response=r.json(),status=r.status_code,ms=dms),calls=calls)
json.dump(out,open('api_test.json','w'),indent=1)
print(json.dumps(out['summary'],indent=1)); print(out['demo']['response'])

# ---- screenshots ----
from playwright.sync_api import sync_playwright
import os; os.makedirs('shots',exist_ok=True)
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={'width':1280,'height':720},device_scale_factor=2)
    for n in (1,2,3,4):
        pg.goto(f'http://127.0.0.1:5000/module{n}'); pg.wait_for_load_state('networkidle'); pg.screenshot(path=f'shots/module{n}.png')
    b.close()
srv.terminate()
