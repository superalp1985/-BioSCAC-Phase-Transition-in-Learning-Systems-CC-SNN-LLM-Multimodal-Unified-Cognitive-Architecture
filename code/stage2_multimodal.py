#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: 单模态分别验证 — 视觉(8×8像素图) + 听觉(合成声音)
============================================================
【可直接复现】数据全部代码生成，无需外部文件

核心对比：
  SNN+LUT（真感知）：原始信号→SNN编码→桶→LUT查表
  LLM裸跑（伪全模态）：原始信号→文字化→LLM分类

关键区别：
  LLM收到的是像素数值或耳蜗频谱的文字版，不是人写的描述
  这模拟了"伪全模态"的本质——把非语言信号转成token
"""
import sys, os, time, json, re
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch, numpy as np, requests
from pathlib import Path
from collections import Counter, defaultdict

# 导入模型
MODEL_DIR = Path(__file__).parent.parent / "model"
sys.path.insert(0, str(MODEL_DIR.parent))
from model.prism_visual_v2 import PrismVisualEncoderV2
from model.real_cochlea import RealCochlea

# ═══════════════ 配置 ═══════════════
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
    p = f"分类任务。判断描述属于哪个类别。\n描述：{text[:500]}\n类别：\n{cs}\n只回复编号数字：\n"
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


# ═══════════════ 数据生成：8×8像素图 ═══════════════
def make_pixel_art():
    """
    手绘8×8 RGB像素图，5类简单几何图形：
      0=横线  1=竖线  2=十字  3=方框  4=对角线
    每类30个样本，带随机颜色和微小噪声
    """
    samples = []
    names = ['横线', '竖线', '十字', '方框', '对角线']

    for _ in range(30):
        img = np.zeros((8,8,3), dtype=np.float32)
        row = np.random.randint(2, 6)
        color = np.random.rand(3) * 0.5 + 0.5
        img[row, :, :] = color
        img += np.random.uniform(-0.05, 0.05, img.shape)
        samples.append((np.clip(img, 0, 1), 0))

    for _ in range(30):
        img = np.zeros((8,8,3), dtype=np.float32)
        col = np.random.randint(2, 6)
        color = np.random.rand(3) * 0.5 + 0.5
        img[:, col, :] = color
        img += np.random.uniform(-0.05, 0.05, img.shape)
        samples.append((np.clip(img, 0, 1), 1))

    for _ in range(30):
        img = np.zeros((8,8,3), dtype=np.float32)
        color = np.random.rand(3) * 0.5 + 0.5
        img[3:5, :, :] = color
        img[:, 3:5, :] = color
        img += np.random.uniform(-0.05, 0.05, img.shape)
        samples.append((np.clip(img, 0, 1), 2))

    for _ in range(30):
        img = np.zeros((8,8,3), dtype=np.float32)
        color = np.random.rand(3) * 0.5 + 0.5
        img[1, 1:7, :] = color
        img[6, 1:7, :] = color
        img[1:7, 1, :] = color
        img[1:7, 6, :] = color
        img += np.random.uniform(-0.05, 0.05, img.shape)
        samples.append((np.clip(img, 0, 1), 3))

    for _ in range(30):
        img = np.zeros((8,8,3), dtype=np.float32)
        color = np.random.rand(3) * 0.5 + 0.5
        for i in range(8): img[i, i, :] = color
        img += np.random.uniform(-0.05, 0.05, img.shape)
        samples.append((np.clip(img, 0, 1), 4))

    np.random.shuffle(samples)
    return samples, names


# ═══════════════ 数据生成：合成声音 ═══════════════
def make_sounds():
    """
    合成简单声音，5类：
      0=低音(200Hz)  1=中音(600Hz)  2=高音(1500Hz)
      3=和弦(200+400+600Hz叠加)  4=脉冲(方波调制)
    """
    SR = 16000; dur = 0.3
    t = np.linspace(0, dur, int(SR * dur))
    samples = []
    names = ['低音', '中音', '高音', '和弦', '脉冲']

    for _ in range(30):
        f = 200 + np.random.uniform(-20, 20)
        w = np.sin(2*np.pi*f*t) * (0.7 + np.random.uniform(-0.1, 0.1))
        samples.append((w.astype(np.float32), 0, SR))

    for _ in range(30):
        f = 600 + np.random.uniform(-50, 50)
        w = np.sin(2*np.pi*f*t) * (0.7 + np.random.uniform(-0.1, 0.1))
        samples.append((w.astype(np.float32), 1, SR))

    for _ in range(30):
        f = 1500 + np.random.uniform(-100, 100)
        w = np.sin(2*np.pi*f*t) * (0.7 + np.random.uniform(-0.1, 0.1))
        samples.append((w.astype(np.float32), 2, SR))

    for _ in range(30):
        w = (np.sin(2*np.pi*200*t) + 0.5*np.sin(2*np.pi*400*t) +
             0.3*np.sin(2*np.pi*600*t)) * 0.5
        w += np.random.uniform(-0.05, 0.05, w.shape)
        samples.append((w.astype(np.float32), 3, SR))

    for _ in range(30):
        pulse_freq = 10 + np.random.uniform(-2, 2)
        carrier = np.sin(2*np.pi*400*t)
        envelope = (np.sin(2*np.pi*pulse_freq*t) > 0).astype(np.float32)
        w = carrier * envelope * 0.7
        samples.append((w.astype(np.float32), 4, SR))

    np.random.shuffle(samples)
    return samples, names


# ═══════════════ CC-SNN + LUT（同Stage 1） ═══════════════
class CCSNN:
    def __init__(self, dim, K):
        self.K, self.dim = K, dim
        self.block_size = dim // K
        self.block_map = [(k*self.block_size, min((k+1)*self.block_size, dim)) for k in range(K)]
        self.beta = np.exp(-1e-3/10e-3)
        self.v_threshold = 1.0
        self.block_weights = [torch.ones(e-s)*1.5 for s,e in self.block_map]

    def compress(self, X):
        bids, confs = [], []
        for i in range(X.shape[0]):
            x = X[i]; energies = torch.zeros(self.K)
            for k in range(self.K):
                s,e = self.block_map[k]; local = x[s:e]; v=0.0; sp=0
                for t in range(10):
                    v = self.beta*v + (1-self.beta)*(local*self.block_weights[k]).sum()
                    if v >= self.v_threshold: sp+=1; v=0
                energies[k] = sp
            bid = energies.argmax().item()
            confs.append((energies[bid]/(energies.sum()+1e-8)).item())
            bids.append(bid)
        return torch.tensor(bids), torch.tensor(confs)

class LookupTable:
    def __init__(self):
        self.table = {}; self.sub_table = {}
    def learn(self, bids, labels, feats):
        bc = defaultdict(list)
        for i,(bid,lbl) in enumerate(zip(bids,labels)): bc[bid].append((feats[i],lbl))
        for bid,items in bc.items():
            classes = set(s[1] for s in items)
            if len(classes)==1: self.table[bid]=list(classes)[0]
            else:
                centroids={}
                for cls in classes:
                    cf=torch.stack([s[0] for s in items if s[1]==cls])
                    centroids[cls]=cf.mean(dim=0)
                self.sub_table[bid]=centroids
                self.table[bid]=Counter(s[1] for s in items).most_common(1)[0][0]
        pure=sum(1 for b in self.table if b not in self.sub_table)
        log(f"    LUT: {len(self.table)}桶 (纯{pure} 混{len(self.sub_table)})")
    def query(self, bid, feat=None):
        if bid in self.sub_table and feat is not None:
            best_c,best_d=-1,float('inf')
            for cls,cent in self.sub_table[bid].items():
                d=torch.norm(feat-cent).item()
                if d<best_d: best_d=d; best_c=cls
            return best_c
        return self.table.get(bid,-1)


# ═══════════════ 通用评估 ═══════════════
def evaluate(name, feats, labels, descs, cat_names, n_test=30):
    N = feats.shape[0]; C = len(cat_names); K = C
    idx = list(range(N)); np.random.shuffle(idx)
    ti, tri = idx[:n_test], idx[n_test:]
    X_tr = feats[tri]; y_tr = [labels[i] for i in tri]
    X_te = feats[ti]; y_te = [labels[i] for i in ti]
    te_descs = [descs[i] for i in ti]

    log(f"\n{'='*50}")
    log(f"模态: {name} (C={C}, N={N}, K={K})")

    snn = CCSNN(feats.shape[1], K)
    tr_bids,_ = snn.compress(X_tr)
    log(f"  分桶: {len(Counter(tr_bids.tolist()))}/{K}")

    lut = LookupTable()
    lut.learn(tr_bids.tolist(), y_tr, X_tr)

    te_bids,_ = snn.compress(X_te)
    ok=0; lats=[]
    for i in range(n_test):
        t0=time.time(); pred=lut.query(te_bids[i].item(),X_te[i]); lats.append(time.time()-t0)
        if pred==y_te[i]: ok+=1
    acc_lut=ok/n_test; lat_lut=np.mean(lats)*1000
    log(f"  LUT: {acc_lut:.1%} ({ok}/{n_test}) {lat_lut:.3f}ms")

    llm_ok=0; llm_fail=0; llm_lats=[]
    for i in range(n_test):
        pred,lat,st=ollama_classify(te_descs[i],cat_names)
        llm_lats.append(lat)
        if st=="ok" and pred==y_te[i]: llm_ok+=1
        elif st!="ok": llm_fail+=1
    acc_llm=llm_ok/max(1,n_test-llm_fail); lat_llm=np.mean(llm_lats)*1000
    log(f"  LLM: {acc_llm:.1%} {lat_llm:.1f}ms")

    passed = acc_lut >= acc_llm
    log(f"  >>> {'✅ PASS' if passed else '❌ FAIL'}: LUT={acc_lut:.1%} vs LLM={acc_llm:.1%}")
    return {"modality":name,"C":C,"K":K,"acc_lut":acc_lut,"acc_llm":acc_llm,
            "lat_lut_ms":lat_lut,"lat_llm_ms":lat_llm,"passed":passed}


def run():
    log("="*60)
    log("Stage 2: 单模态验证 (视觉 + 听觉)")
    log("="*60)

    # ═══ 视觉 ═══
    vis_enc = PrismVisualEncoderV2(grid_h=8, grid_w=8)
    vis_samples, vis_names = make_pixel_art()
    vis_feats=[]; vis_labels=[]; vis_descs=[]
    for img, lbl in vis_samples:
        spike = vis_enc.encode(img, timesteps=10).mean(dim=0)
        vis_feats.append(spike); vis_labels.append(lbl)
        # 伪全模态：给LLM原始像素值（不是描述）
        rows = []
        for row in range(8):
            vals = ",".join(f"{img[row,c,0]:.1f}" for c in range(8))
            rows.append(f"行{row}:[{vals}]")
        vis_descs.append("8x8亮度图:\n"+"\n".join(rows))
    vis_feats = torch.stack(vis_feats)
    res_vis = evaluate("视觉(8×8像素图)", vis_feats, vis_labels, vis_descs, vis_names)

    # ═══ 听觉 ═══
    cochlea = RealCochlea(n_channels=32, freq_range=(80, 8000))
    snd_samples, snd_names = make_sounds()
    snd_feats=[]; snd_labels=[]; snd_descs=[]
    for wav, lbl, sr in snd_samples:
        spike = cochlea.encode_waveform(wav, sr); feat = spike.mean(dim=0)
        snd_feats.append(feat); snd_labels.append(lbl)
        info = cochlea.describe(wav, sr)
        ch_str = ",".join(f"{e:.2f}" for e in info['channel_energy_norm'][:16])
        snd_descs.append(f"耳蜗32通道响应(前16):[{ch_str}] 主频={info['dominant_freq']:.0f}Hz")
    snd_feats = torch.stack(snd_feats)
    res_snd = evaluate("听觉(合成声音)", snd_feats, snd_labels, snd_descs, snd_names)

    # ═══ 汇总 ═══
    all_pass = res_vis['passed'] and res_snd['passed']
    log(f"\n{'='*60}")
    log(f"Stage 2: {'✅ ALL PASS' if all_pass else '❌ HAS FAILURES'}")
    results = {"stage":2,"vision":res_vis,"audio":res_snd,"all_pass":all_pass,
               "timestamp":time.strftime('%Y-%m-%d %H:%M:%S')}
    out = RESULTS_DIR/"stage2_results.json"
    with open(out,'w',encoding='utf-8') as f: json.dump(results,f,ensure_ascii=False,indent=2)
    log(f"保存: {out}")

if __name__ == '__main__':
    run()
