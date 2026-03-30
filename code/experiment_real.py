#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BioSCAC 论文核心验证 — 真Ollama + 真D-LIF SNN + SCAC"""
import sys, os, time, json, math, warnings, re
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
import torch, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.cluster import MiniBatchKMeans
import requests

RESULTS_DIR = Path(r'E:\BioSCAC\theory_validation')
RESULTS_DIR.mkdir(exist_ok=True)
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"
LOG = []
def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"; print(s, flush=True); LOG.append(s)
torch.manual_seed(42); np.random.seed(42)

def ollama_classify(text, cats, timeout=30):
    cs = "\n".join(f"{i+1}. {c}" for i,c in enumerate(cats))
    p = f"文本分类。归入最合适的类别。\n文本：「{text[:300]}」\n类别：\n{cs}\n只回复编号数字：\n"
    try:
        r = requests.post(OLLAMA_URL, json={"model":OLLAMA_MODEL,
            "messages":[{"role":"user","content":p}], "stream":False,
            "options":{"temperature":0.1,"num_predict":10}}, timeout=timeout)
        c = r.json()["message"]["content"].strip()
        nums = re.findall(r'\d+', c)
        if nums:
            idx = int(nums[0])-1
            if 0 <= idx < len(cats): return idx, "ok"
        return -1, "parse"
    except Exception as e: return -1, str(e)

def load_texts(d, n=250):
    docs = []
    for f in sorted(Path(d).glob("*.txt"))[:n]:
        try:
            t = f.read_text(encoding='utf-8')[:3000]
            if len(t) > 50: docs.append((t, f.stem))
        except: pass
    return docs

def text_features(docs, dim=256):
    ac = Counter(); dng = []
    for t,_ in docs:
        d = Counter()
        for i in range(len(t)-1):
            ng = t[i:i+2]
            if not ng.isspace(): d[ng]+=1; ac[ng]+=1
        dng.append(d)
    top = [ng for ng,_ in ac.most_common(dim)]
    m = {ng:i for i,ng in enumerate(top)}
    F = np.zeros((len(docs),dim), dtype=np.float32)
    for j,d in enumerate(dng):
        s = sum(d.values()) or 1
        for ng,c in d.items():
            if ng in m: F[j,m[ng]] = c/s
    return torch.tensor(F)

def cluster_data(docs, feats, C):
    km = MiniBatchKMeans(n_clusters=C, random_state=42, n_init=3)
    labels = km.fit_predict(feats.numpy())
    names = []
    for c in range(C):
        cd = "".join(docs[i][0] for i in range(len(docs)) if labels[i]==c)
        ch = Counter(x for x in cd if '\u4e00'<=x<='\u9fff')
        names.append("".join(x for x,_ in ch.most_common(5))+"类" if ch else f"类{c+1}")
    return labels, names

class RealCCSNN:
    """
    CC-SNN: D-LIF + WTA竞争 + 自适应阈值 + SCAC约束STDP
    支持block_map局部连接（视觉模态用）
    """
    def __init__(self, dim, K, block_map=None):
        self.K, self.dim = K, dim
        # 多样初始化：每个桶偏好不同输入子空间
        self.W = torch.zeros(dim, K)
        chunk = dim // K
        for k in range(K):
            lo = k * chunk
            hi = min((k+1) * chunk, dim)
            self.W[lo:hi, k] = torch.randn(hi - lo) * 1.5 + 1.0  # 强偏好
            # 其余维度弱连接
            other = torch.randn(dim, 1) * 0.1
            self.W[:, k] += other.squeeze()
        self.W.clamp_(0.01, 3.0)

        # 连接掩码（局部连接支持）
        if block_map is not None:
            self.mask = torch.zeros(dim, K)
            for i0,i1,n0,n1 in block_map:
                self.mask[i0:i1, n0:n1] = 1.0
            self.W *= self.mask
        else:
            self.mask = torch.ones(dim, K)

        self.beta = np.exp(-1e-3 / 10e-3)
        self.kappa = 0.7
        # 自适应阈值（homeostatic）
        self.thresholds = torch.ones(K) * 1.0
        self.fire_counts = torch.zeros(K)
        self.target_rate = 1.0 / K  # 每个桶目标发放率

    def spike(self, x):
        """D-LIF + WTA: 只有winner发放，loser被抑制"""
        v = torch.zeros(self.K)
        sc = torch.zeros(self.K)
        Weff = self.W * self.mask
        T = 10
        for t in range(T):
            x_t = x * (1.0 + 0.05 * (t / T))  # 时间渐强
            i_syn = x_t @ Weff
            v = self.beta * v + (1 - self.beta) * i_syn

            # WTA: 只有最强的过阈值
            above = v - self.thresholds
            if above.max() > 0:
                winner = above.argmax()
                sp = torch.zeros(self.K)
                sp[winner] = 1.0
                sc += sp
                v[winner] = 0.0  # reset winner
                v -= 0.3 * sp.sum()  # 侧向抑制（全局抑制）
                v.clamp_(min=0.0)
        return sc

    def compress(self, X):
        bids, confs = [], []
        for i in range(X.shape[0]):
            sc = self.spike(X[i])
            if sc.sum() == 0:
                # 无发放→取膜电位最大
                Weff = self.W * self.mask
                v = X[i] @ Weff
                b = v.argmax().item()
                confs.append(0.1)
            else:
                b = sc.argmax().item()
                confs.append((sc[b] / (sc.sum() + 1e-8)).item())
            bids.append(b)
            self.fire_counts[b] += 1
        return torch.tensor(bids), torch.tensor(confs)

    def _adapt_thresholds(self):
        """自适应阈值：多发的升，少发的降"""
        total = self.fire_counts.sum().clamp(min=1)
        rates = self.fire_counts / total
        for k in range(self.K):
            if rates[k] > self.target_rate * 1.5:
                self.thresholds[k] *= 1.05  # 太活跃→升阈值
            elif rates[k] < self.target_rate * 0.5:
                self.thresholds[k] *= 0.95  # 太沉默→降阈值
        self.thresholds.clamp_(0.3, 5.0)
        self.fire_counts.zero_()

    def train_scac(self, X, y, ep=80):
        prev = float('inf')
        for e in range(ep):
            self.fire_counts.zero_()
            bids, _ = self.compress(X)

            # 自适应阈值
            if (e + 1) % 5 == 0:
                self._adapt_thresholds()

            # 损失：类内桶散度
            loss = 0; nc = 0
            for c in torch.unique(y):
                m = (y == c); cb = bids[m]
                if m.sum() < 2: continue
                loss += (cb != cb.mode().values).float().mean().item(); nc += 1
            loss /= max(1, nc)

            # 竞争STDP：同类强化到同桶，异类推离
            improvement = prev - loss
            if improvement > -0.01:  # 容忍微小退步
                for c in torch.unique(y):
                    m = (y == c)
                    cx = X[m]
                    # 类中心的目标桶
                    center = cx.mean(0)
                    tb = self.spike(center).argmax() if self.spike(center).sum() > 0 else (center @ (self.W * self.mask)).argmax()

                    # Hebbian: 强化 center→target_bucket
                    delta = torch.zeros_like(self.W)
                    delta[:, tb] = center * 0.08
                    # Anti-Hebbian: 弱化到其他桶
                    for k in range(self.K):
                        if k != tb:
                            delta[:, k] = -center * 0.02

                    self.W += self.kappa * delta * self.mask
                    self.W.clamp_(0.01, 3.0)

                self.kappa = max(0.15, self.kappa - 0.003)
            else:
                self.kappa = min(0.85, self.kappa + 0.015)
            prev = loss

        bids, _ = self.compress(X)
        n_used = len(torch.unique(bids))
        # 桶分布
        bc = Counter(bids.tolist())
        dist = " ".join(f"b{k}:{bc.get(k,0)}" for k in range(min(self.K, 10)))
        log(f"    SNN: loss={loss:.3f} 桶={n_used}/{self.K} k={self.kappa:.2f} 阈值={self.thresholds[:min(self.K,6)].tolist()}")
        log(f"    分布: {dist}")

def run():
    log("="*60)
    log("BioSCAC 真家伙验证: Ollama + D-LIF SNN + SCAC")
    log("="*60)

    docs = load_texts(r'E:\BioSCAC\data\curriculum_text')
    log(f"文本: {len(docs)}篇")
    feats = text_features(docs)
    log(f"特征: {feats.shape}")

    # Ollama连通测试
    _,st = ollama_classify("测试", ["A类","B类"])
    log(f"Ollama: {st}")

    N_test = 40
    C_vals = [4, 8, 16, 32, 64, 128]
    sqN = int(math.sqrt(len(docs)))
    log(f"N={len(docs)} √N={sqN} 测试={N_test}/组")

    results = {"config":{"N":len(docs),"sqrtN":sqN,"Cvals":C_vals,"model":OLLAMA_MODEL},"data":[]}

    for C in C_vals:
        if C > len(docs): continue
        log(f"\n{'='*50}")
        log(f"C={C} ({'亚' if C<=sqN else '超'}临界)")

        labels, cnames = cluster_data(docs, feats, C)
        idx = list(range(len(docs))); np.random.shuffle(idx)
        ti, tri = idx[:N_test], idx[N_test:]
        Xtr,ytr = feats[tri], torch.tensor(labels[tri])
        Xte,yte = feats[ti], torch.tensor(labels[ti])
        tdocs = [docs[i] for i in ti]

        # A: LLM直接
        log(f"  [A] Ollama直分 ({C}类)")
        cA=0; fA=0
        for i in range(N_test):
            pi,s = ollama_classify(tdocs[i][0][:200], cnames)
            if s=="ok" and pi==yte[i].item(): cA+=1
            elif s!="ok": fA+=1
            if (i+1)%10==0: log(f"    {i+1}/{N_test} acc={cA/max(1,i+1-fA):.0%}")
        aA = cA/max(1,N_test-fA)
        log(f"  A={aA:.1%} (fail={fA})")

        # SNN
        K = max(2, min(C//2, 32))
        if C > sqN: K = max(2, round(C**(2/3)/(len(docs)/C)**(1/3)))
        K = min(K, C, 32)
        log(f"  [SNN] K={K}")
        snn = RealCCSNN(feats.shape[1], K)
        snn.train_scac(Xtr, ytr)
        tr_bids,_ = snn.compress(Xtr)
        te_bids, te_conf = snn.compress(Xte)

        # B: SNN多数投票
        bc = {}
        for b,l in zip(tr_bids.tolist(), ytr.tolist()):
            bc.setdefault(b, Counter())[l] += 1
        cB = sum(1 for i in range(N_test)
                 if te_bids[i].item() in bc and
                 bc[te_bids[i].item()].most_common(1)[0][0]==yte[i].item())
        aB = cB/N_test
        log(f"  B={aB:.1%}")

        # C: SNN→桶内LLM→SCAC
        log(f"  [C] SNN→桶内LLM→SCAC")
        bcat = defaultdict(set)
        for b,l in zip(tr_bids.tolist(), ytr.tolist()): bcat[b].add(l)

        cC=0; scac=0; fC=0
        for i in range(N_test):
            b = te_bids[i].item(); gt = yte[i].item()
            if b not in bcat or not bcat[b]:
                scac+=1
                d=((Xtr-Xte[i:i+1])**2).sum(1)
                if ytr[d.argmin()].item()==gt: cC+=1
                continue
            sc = sorted(bcat[b])
            if len(sc)==1:
                if sc[0]==gt: cC+=1
                continue
            sn = [cnames[c] for c in sc]
            pi,s = ollama_classify(tdocs[i][0][:200], sn)
            if s=="ok" and 0<=pi<len(sc):
                if sc[pi]==gt: cC+=1
                elif te_conf[i]<0.3:
                    scac+=1; d=((Xtr-Xte[i:i+1])**2).sum(1)
                    if ytr[d.argmin()].item()==gt: cC+=1
            else:
                fC+=1; scac+=1; d=((Xtr-Xte[i:i+1])**2).sum(1)
                if ytr[d.argmin()].item()==gt: cC+=1
            if (i+1)%10==0: log(f"    {i+1}/{N_test}")
        aC = cC/N_test
        avg_bc = np.mean([len(v) for v in bcat.values()]) if bcat else C
        log(f"  C={aC:.1%} SCAC={scac}/{N_test} 桶内均类={avg_bc:.1f}")

        r = {"C":C,"regime":"sub" if C<=sqN else "super",
             "acc_A":aA,"acc_B":aB,"acc_C":aC,"K":K,
             "scac":scac,"avg_bucket_cats":avg_bc,"fail_A":fA,"fail_C":fC}
        results["data"].append(r)
        log(f"  >>> A={aA:.1%} B={aB:.1%} C={aC:.1%} {'A胜' if aA>aC else 'C≥A' if aC>=aA else '≈'}")

    # 保存
    results["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(RESULTS_DIR/"real_validation.json",'w',encoding='utf-8') as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    with open(RESULTS_DIR/"real_log.txt",'w',encoding='utf-8') as f:
        f.write('\n'.join(LOG))
    log(f"\n保存: {RESULTS_DIR/'real_validation.json'}")

    log(f"\n{'='*60}")
    log("汇总")
    for r in results["data"]:
        log(f"  C={r['C']:3d}({r['regime']}): A={r['acc_A']:.1%} B={r['acc_B']:.1%} C={r['acc_C']:.1%} SCAC={r['scac']}")
    log("="*60)

if __name__ == '__main__':
    run()
