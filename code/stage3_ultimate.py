#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: 终极测试 — 真正多模态 (视觉+听觉同时输入)
===================================================
【可直接复现】

场景：5类"影音片段"，每个片段同时有画面+声音
  0=红色闪光+高音警报  (警报)
  1=蓝色平静+低音嗡鸣  (海洋)
  2=绿色闪烁+中频脉冲  (森林)
  3=黄色渐变+和弦       (日落)
  4=黑白交替+静音       (老电影)

SNN用block_map：视觉通道→左脑，听觉通道→右脑

对比4条路径：
  A) 多模态联合 SNN+LUT （视觉+听觉→SNN→LUT）
  B) 仅视觉 SNN+LUT
  C) 仅听觉 SNN+LUT
  D) LLM读原始数据（伪全模态）

通过标准：A ≥ B, A ≥ C, A ≥ D
"""
import sys, os, time, json, re
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch, numpy as np, requests
from pathlib import Path
from collections import Counter, defaultdict

MODEL_DIR = Path(__file__).parent.parent / "model"
sys.path.insert(0, str(MODEL_DIR.parent))
from model.prism_visual_v2 import PrismVisualEncoderV2
from model.real_cochlea import RealCochlea

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG = []
def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"; print(s, flush=True); LOG.append(s)
torch.manual_seed(42); np.random.seed(42)

def ollama_classify(text, cats, timeout=30):
    cs = "\n".join(f"{i+1}. {c}" for i,c in enumerate(cats))
    p = f"分类任务。判断以下信号数据属于哪个场景。\n数据：\n{text[:500]}\n类别：\n{cs}\n只回复编号数字：\n"
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_URL, json={"model":OLLAMA_MODEL,
            "messages":[{"role":"user","content":p}], "stream":False,
            "options":{"temperature":0.1,"num_predict":10}}, timeout=timeout)
        lat = time.time()-t0
        c = r.json()["message"]["content"].strip()
        nums = re.findall(r'\d+', c)
        if nums:
            idx = int(nums[0])-1
            if 0 <= idx < len(cats): return idx, lat, "ok"
        return -1, lat, "parse"
    except Exception as e: return -1, time.time()-t0, str(e)


# ═══════════════ 影音场景生成 ═══════════════
def make_av_scenes():
    SR = 16000; dur = 0.3
    t = np.linspace(0, dur, int(SR*dur))
    samples = []; names = ['警报', '海洋', '森林', '日落', '老电影']

    for _ in range(30):  # 警报：红闪+高音
        img = np.zeros((8,8,3),np.float32)
        img[:,:,0] = 0.8+np.random.uniform(-0.1,0.1)
        for r,c in zip(np.random.randint(0,8,2), np.random.randint(0,8,2)):
            img[r,c,:] = [1,1,0.5]
        img = np.clip(img+np.random.uniform(-0.03,0.03,img.shape),0,1)
        wav = (np.sin(2*np.pi*(1500+np.random.uniform(-100,100))*t)*0.8).astype(np.float32)
        samples.append((img, wav, SR, 0))

    for _ in range(30):  # 海洋：蓝底+低音
        img = np.zeros((8,8,3),np.float32)
        img[:,:,2] = 0.7+np.random.uniform(-0.1,0.1)
        img[:,:,1] = 0.2+np.random.uniform(-0.05,0.05)
        img = np.clip(img+np.random.uniform(-0.03,0.03,img.shape),0,1)
        wav = (np.sin(2*np.pi*(200+np.random.uniform(-20,20))*t)*0.5).astype(np.float32)
        samples.append((img, wav, SR, 1))

    for _ in range(30):  # 森林：绿底+中频脉冲
        img = np.zeros((8,8,3),np.float32)
        img[:,:,1] = 0.6+np.random.uniform(-0.15,0.15,(8,8))
        img[:,:,0] = 0.1+np.random.uniform(-0.05,0.05,(8,8))
        img = np.clip(img,0,1)
        f = 600+np.random.uniform(-50,50)
        pulse = (np.sin(2*np.pi*8*t)>0).astype(np.float32)
        wav = (np.sin(2*np.pi*f*t)*pulse*0.6).astype(np.float32)
        samples.append((img, wav, SR, 2))

    for _ in range(30):  # 日落：黄色渐变+和弦
        img = np.zeros((8,8,3),np.float32)
        for row in range(8):
            img[row,:,0]=0.9-row*0.08; img[row,:,1]=0.7-row*0.06; img[row,:,2]=0.1+row*0.03
        img = np.clip(img+np.random.uniform(-0.03,0.03,img.shape),0,1)
        wav = (np.sin(2*np.pi*260*t)*0.4+np.sin(2*np.pi*330*t)*0.3+np.sin(2*np.pi*390*t)*0.2).astype(np.float32)
        samples.append((img, wav, SR, 3))

    for _ in range(30):  # 老电影：黑白棋盘+静音
        img = np.zeros((8,8,3),np.float32)
        for r in range(8):
            for c in range(8): img[r,c,:]=0.8 if (r+c)%2==0 else 0.1
        img = np.clip(img+np.random.uniform(-0.05,0.05,img.shape),0,1)
        wav = np.random.uniform(-0.02,0.02,len(t)).astype(np.float32)
        samples.append((img, wav, SR, 4))

    np.random.shuffle(samples)
    return samples, names


class CCSNN:
    def __init__(self, dim, K):
        self.K,self.dim = K,dim
        self.block_size = dim//K
        self.block_map = [(k*self.block_size,min((k+1)*self.block_size,dim)) for k in range(K)]
        self.beta = np.exp(-1e-3/10e-3); self.v_threshold = 1.0
        self.block_weights = [torch.ones(e-s)*1.5 for s,e in self.block_map]
    def compress(self, X):
        bids,confs=[],[]
        for i in range(X.shape[0]):
            x=X[i]; en=torch.zeros(self.K)
            for k in range(self.K):
                s,e=self.block_map[k]; loc=x[s:e]; v=0.0; sp=0
                for t in range(10):
                    v=self.beta*v+(1-self.beta)*(loc*self.block_weights[k]).sum()
                    if v>=self.v_threshold: sp+=1; v=0
                en[k]=sp
            bid=en.argmax().item(); confs.append((en[bid]/(en.sum()+1e-8)).item()); bids.append(bid)
        return torch.tensor(bids),torch.tensor(confs)

class LookupTable:
    def __init__(self): self.table={}; self.sub_table={}
    def learn(self,bids,labels,feats):
        bc=defaultdict(list)
        for i,(b,l) in enumerate(zip(bids,labels)): bc[b].append((feats[i],l))
        for bid,items in bc.items():
            cls=set(s[1] for s in items)
            if len(cls)==1: self.table[bid]=list(cls)[0]
            else:
                ct={}
                for c in cls: ct[c]=torch.stack([s[0] for s in items if s[1]==c]).mean(0)
                self.sub_table[bid]=ct
                self.table[bid]=Counter(s[1] for s in items).most_common(1)[0][0]
    def query(self,bid,feat=None):
        if bid in self.sub_table and feat is not None:
            bc,bd=-1,float('inf')
            for c,ct in self.sub_table[bid].items():
                d=torch.norm(feat-ct).item()
                if d<bd: bd=d; bc=c
            return bc
        return self.table.get(bid,-1)


def eval_snn(name, feats, labels, n_test, K):
    N=feats.shape[0]; idx=list(range(N)); np.random.shuffle(idx)
    ti,tri = idx[:n_test],idx[n_test:]
    X_tr=feats[tri]; y_tr=[labels[i] for i in tri]
    X_te=feats[ti]; y_te=[labels[i] for i in ti]
    snn=CCSNN(feats.shape[1],K); tr_bids,_=snn.compress(X_tr)
    lut=LookupTable(); lut.learn(tr_bids.tolist(),y_tr,X_tr)
    te_bids,_=snn.compress(X_te)
    ok=0; lats=[]
    for i in range(n_test):
        t0=time.time(); pred=lut.query(te_bids[i].item(),X_te[i]); lats.append(time.time()-t0)
        if pred==y_te[i]: ok+=1
    acc=ok/n_test; lat=np.mean(lats)*1000
    log(f"  {name}: {acc:.1%} ({ok}/{n_test}) {lat:.3f}ms")
    return acc, lat, ti


def run():
    log("="*60)
    log("Stage 3: 终极测试 — 真正多模态 (视觉+听觉)")
    log("="*60)

    vis_enc = PrismVisualEncoderV2(grid_h=8, grid_w=8)
    cochlea = RealCochlea(n_channels=32, freq_range=(80, 8000))
    scenes, names = make_av_scenes()
    C=len(names); K=C; n_test=30

    vis_feats=[]; aud_feats=[]; labels=[]; llm_descs=[]
    for img,wav,sr,lbl in scenes:
        vf=vis_enc.encode(img,timesteps=10).mean(dim=0)
        af=cochlea.encode_waveform(wav,sr).mean(dim=0)
        vis_feats.append(vf); aud_feats.append(af); labels.append(lbl)
        r_vals=",".join(f"{img[r,4,0]:.1f}" for r in range(8))
        info=cochlea.describe(wav,sr)
        ch_str=",".join(f"{e:.1f}" for e in info['channel_energy_norm'][:8])
        llm_descs.append(f"视觉:R列值=[{r_vals}] 听觉:耳蜗=[{ch_str}] 主频={info['dominant_freq']:.0f}Hz")

    vis_feats=torch.stack(vis_feats); aud_feats=torch.stack(aud_feats)
    multi_feats=torch.cat([vis_feats, aud_feats], dim=1)
    log(f"特征: 视觉={vis_feats.shape[1]} 听觉={aud_feats.shape[1]} 联合={multi_feats.shape[1]}")

    log("\n[A] 多模态联合")
    acc_multi,lat_multi,test_idx = eval_snn("多模态联合", multi_feats, labels, n_test, K)

    log("\n[B] 仅视觉")
    acc_vis,lat_vis,_ = eval_snn("仅视觉", vis_feats, labels, n_test, K)

    log("\n[C] 仅听觉")
    acc_aud,lat_aud,_ = eval_snn("仅听觉", aud_feats, labels, n_test, K)

    log("\n[D] LLM伪全模态")
    llm_ok=0; llm_fail=0; llm_lats=[]
    for i in test_idx:
        pred,lat,st=ollama_classify(llm_descs[i],names)
        llm_lats.append(lat)
        if st=="ok" and pred==labels[i]: llm_ok+=1
        elif st!="ok": llm_fail+=1
    acc_llm=llm_ok/max(1,n_test-llm_fail); lat_llm=np.mean(llm_lats)*1000
    log(f"  LLM: {acc_llm:.1%} {lat_llm:.1f}ms")

    multi_win = acc_multi>=acc_vis and acc_multi>=acc_aud
    beat_llm = acc_multi>=acc_llm

    log(f"\n{'='*60}")
    log("Stage 3 结果")
    log("="*60)
    log(f"  A)多模态SNN+LUT  {acc_multi:.1%}  {lat_multi:.3f}ms")
    log(f"  B)仅视觉SNN+LUT  {acc_vis:.1%}  {lat_vis:.3f}ms")
    log(f"  C)仅听觉SNN+LUT  {acc_aud:.1%}  {lat_aud:.3f}ms")
    log(f"  D)LLM伪全模态     {acc_llm:.1%}  {lat_llm:.1f}ms")
    log(f"\n  多模态≥单模态: {'✅' if multi_win else '❌'}")
    log(f"  多模态≥LLM:     {'✅' if beat_llm else '❌'}")
    log(f"  Stage 3: {'✅ ALL PASS' if multi_win and beat_llm else '❌ FAIL'}")

    results = {"stage":3,"acc_multi":acc_multi,"acc_vis":acc_vis,"acc_aud":acc_aud,
               "acc_llm":acc_llm,"lat_multi_ms":lat_multi,"lat_llm_ms":lat_llm,
               "multi_wins":multi_win,"beats_llm":beat_llm,
               "all_pass":multi_win and beat_llm,
               "timestamp":time.strftime('%Y-%m-%d %H:%M:%S')}
    out = RESULTS_DIR/"stage3_results.json"
    with open(out,'w',encoding='utf-8') as f: json.dump(results,f,ensure_ascii=False,indent=2)
    log(f"保存: {out}")

if __name__ == '__main__':
    run()
