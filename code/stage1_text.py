#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: 文本分类 — SNN→LUT查表 vs LLM裸跑
============================================
【可直接复现】无需外部数据，脚本自带课程文本生成器

环境要求：
  pip install torch numpy requests
  Ollama运行中（用于LLM基线对比）：
    ollama pull qwen2.5:3b-instruct-q4_K_M
    ollama serve

配置项：
  OLLAMA_URL — Ollama地址（默认本地11434）
  OLLAMA_MODEL — 模型名（可换成任意Ollama模型）
"""
import sys, os, time, json, math, re
sys.stdout.reconfigure(encoding='utf-8')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch, numpy as np, requests
from pathlib import Path
from collections import Counter, defaultdict

# ═══════════════ 配置（按需修改） ═══════════════
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"   # 换模型改这里
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG = []
def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"; print(s, flush=True); LOG.append(s)
torch.manual_seed(42); np.random.seed(42)


# ═══════════════ 内置数据集：课程文本 ═══════════════
def generate_curriculum_texts():
    """
    生成5类课程教学文本，每类25篇（共125篇）
    类别：认知发展、语言发展、运动发展、精细动作、社会情感
    每篇由领域特征词+随机组合生成，模拟真实课程材料
    """
    templates = {
        '认知发展': {
            'keywords': ['思维', '逻辑', '推理', '认知', '概念', '理解', '分析',
                         '判断', '记忆', '注意力', '观察', '想象', '创造',
                         '分类', '排序', '比较', '抽象', '归纳', '空间感知'],
            'sentences': [
                '儿童在{age}岁时开始发展{kw}能力，通过{activity}来促进{kw}的发展。',
                '培养{kw}的关键在于提供适当的{activity}，让儿童在实践中学习。',
                '{kw}是认知发展的重要里程碑，教师应当关注每个孩子的{kw}水平。',
                '通过游戏化的方式训练{kw}，能够有效提升儿童的认知能力。',
                '研究表明，早期{kw}训练对后续学业成绩有显著正向影响。',
            ],
            'activities': ['拼图游戏', '积木搭建', '数学练习', '科学实验', '观察记录',
                          '逻辑推理题', '分类整理', '图案匹配', '迷宫探索'],
        },
        '语言发展': {
            'keywords': ['语言', '词汇', '表达', '阅读', '书写', '发音', '语法',
                         '叙事', '倾听', '对话', '理解力', '口语', '文字',
                         '故事', '朗读', '识字', '拼音', '语感', '沟通'],
            'sentences': [
                '儿童的{kw}发展遵循从简单到复杂的规律，{age}岁是{kw}发展的关键期。',
                '通过{activity}可以有效促进儿童{kw}能力的提升。',
                '{kw}教学应当注重情境化，在真实语境中培养{kw}能力。',
                '教师在{kw}教学中应采用多元化策略，结合{activity}来激发兴趣。',
                '良好的{kw}基础是未来学术成功的重要预测因素。',
            ],
            'activities': ['绘本阅读', '故事讲述', '角色扮演', '儿歌学唱', '看图说话',
                          '亲子对话', '戏剧表演', '字母游戏', '诗歌朗诵'],
        },
        '运动发展': {
            'keywords': ['运动', '体能', '平衡', '协调', '跑步', '跳跃', '攀爬',
                         '力量', '耐力', '灵活性', '体操', '球类', '游泳',
                         '节奏', '反应', '速度', '柔韧性', '大肌肉'],
            'sentences': [
                '大肌肉{kw}发展是幼儿体能教育的核心，{age}岁儿童应掌握基本{kw}技能。',
                '{activity}是促进{kw}发展的有效方式。',
                '每天保证充足的户外活动时间，对{kw}发展至关重要。',
                '{kw}训练要循序渐进，避免过度训练导致运动损伤。',
                '通过{activity}培养{kw}能力，同时发展社交技能。',
            ],
            'activities': ['障碍跑', '跳绳', '球类运动', '体操动作', '户外游戏',
                          '骑行训练', '攀爬活动', '舞蹈律动', '接力比赛'],
        },
        '精细动作': {
            'keywords': ['手指', '精细', '握笔', '剪纸', '捏', '拧', '穿珠',
                         '绘画', '手眼协调', '操控', '书写准备', '折纸',
                         '扣纽扣', '使用筷子', '手工', '粘贴', '描画'],
            'sentences': [
                '{kw}能力的发展需要大量实践机会，通过{activity}可以有效训练。',
                '{age}岁儿童的{kw}能力快速发展，教师应提供丰富的操作材料。',
                '{kw}训练不仅促进动作发展，还能增强儿童的自信心和专注力。',
                '通过日常生活中的{activity}来发展{kw}能力。',
                '{kw}发展与认知发展密切相关，不可忽视。',
            ],
            'activities': ['剪纸手工', '穿珠子', '画画涂色', '橡皮泥造型', '折纸',
                          '拼豆创作', '编织活动', '螺丝拧转', '夹豆子游戏'],
        },
        '社会情感': {
            'keywords': ['情绪', '社交', '合作', '分享', '同理心', '自我管理',
                         '友谊', '冲突解决', '规则', '自信', '归属感',
                         '情感表达', '人际关系', '集体', '礼貌', '关爱'],
            'sentences': [
                '培养儿童的{kw}能力是幼儿教育的重要目标。',
                '通过{activity}帮助儿童学习{kw}技能。',
                '{age}岁是{kw}发展的敏感期，教师应创设支持性环境。',
                '{kw}教育应融入日常活动中，而非单独进行。',
                '良好的{kw}能力是儿童终身幸福的基石。',
            ],
            'activities': ['小组合作游戏', '角色扮演', '情绪卡片识别', '冲突调解练习',
                          '分享圈活动', '班级会议', '社区服务', '互助配对活动'],
        }
    }

    docs = []
    ages = ['3', '4', '5', '6', '3-4', '4-5', '5-6']
    cat_names = list(templates.keys())

    for cat_idx, (cat_name, data) in enumerate(templates.items()):
        for i in range(25):
            paragraphs = []
            for _ in range(3):  # 每篇3段
                tmpl = np.random.choice(data['sentences'])
                kw = np.random.choice(data['keywords'])
                act = np.random.choice(data['activities'])
                age = np.random.choice(ages)
                para = tmpl.format(kw=kw, activity=act, age=age)
                paragraphs.append(para)
            text = '\n'.join(paragraphs)
            docs.append((text, cat_idx, cat_name))

    np.random.shuffle(docs)
    return docs, cat_names


# ═══════════════ LLM调用 ═══════════════
def ollama_classify(text, cats, timeout=30):
    """调用Ollama对文本进行分类"""
    cs = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cats))
    prompt = (f"文本分类任务。将文本归入最合适的类别。\n"
              f"文本：「{text[:300]}」\n类别：\n{cs}\n只回复编号数字：\n")
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 10}
        }, timeout=timeout)
        lat = time.time() - t0
        content = r.json()["message"]["content"].strip()
        nums = re.findall(r'\d+', content)
        if nums:
            idx = int(nums[0]) - 1
            if 0 <= idx < len(cats):
                return idx, lat, "ok"
        return -1, lat, "parse_error"
    except Exception as e:
        return -1, time.time() - t0, str(e)


# ═══════════════ 特征提取（字符n-gram） ═══════════════
def text_features(docs, dim=256):
    """字符二元组(bigram)频率特征"""
    all_counts = Counter()
    doc_ngrams = []
    for text, _, _ in docs:
        d = Counter()
        for i in range(len(text) - 1):
            ng = text[i:i+2]
            if not ng.isspace():
                d[ng] += 1
                all_counts[ng] += 1
        doc_ngrams.append(d)

    top = [ng for ng, _ in all_counts.most_common(dim)]
    ngram_map = {ng: i for i, ng in enumerate(top)}
    F = np.zeros((len(docs), dim), dtype=np.float32)
    for j, d in enumerate(doc_ngrams):
        total = sum(d.values()) or 1
        for ng, count in d.items():
            if ng in ngram_map:
                F[j, ngram_map[ng]] = count / total
    return torch.tensor(F)


# ═══════════════ CC-SNN 局部连接分桶 ═══════════════
class CCSNN:
    """
    收敛约束脉冲神经网络 — 局部连接版
    ─────────────────────────────────
    核心机制：
    1. 将输入特征等分为K块（block_map）
    2. 每块仅连接到对应的SNN神经元子群（局部连接，无跨块串扰）
    3. 每个子群运行D-LIF脉冲神经元仿真
    4. 总脉冲最多的子群 = 桶ID

    无需训练：block_map物理结构本身就是路由机制

    参数说明：
      dim — 输入特征维度（如256）
      K — 桶数（通常=类别数C）
      beta — D-LIF膜电位衰减系数 exp(-dt/tau_m)
      v_threshold — 发放阈值
    """
    def __init__(self, dim, K):
        self.K = K
        self.dim = dim
        self.block_size = dim // K
        self.beta = np.exp(-1e-3 / 10e-3)   # tau_m=10ms, dt=1ms
        self.v_threshold = 1.0

        # block_map: 输入块 → 神经元子群（严格隔离）
        self.block_map = []
        for k in range(K):
            i0 = k * self.block_size
            i1 = min((k + 1) * self.block_size, dim)
            self.block_map.append((i0, i1))

        # 块内权重（均匀正值）
        self.block_weights = []
        for k in range(K):
            bsz = self.block_map[k][1] - self.block_map[k][0]
            self.block_weights.append(torch.ones(bsz) * 1.5)

    def compress(self, X):
        """
        将N个样本压缩为桶ID
        X: (N, dim)
        返回: bucket_ids (N,), confidences (N,)
        """
        bids, confs = [], []
        for i in range(X.shape[0]):
            x = X[i]
            block_energies = torch.zeros(self.K)
            for k in range(self.K):
                i0, i1 = self.block_map[k]
                local_input = x[i0:i1]
                # D-LIF 脉冲仿真（10个时间步）
                v = 0.0
                spikes = 0
                for t in range(10):
                    i_syn = (local_input * self.block_weights[k]).sum()
                    v = self.beta * v + (1 - self.beta) * i_syn
                    if v >= self.v_threshold:
                        spikes += 1
                        v = 0.0  # 复位
                block_energies[k] = spikes
            bid = block_energies.argmax().item()
            total = block_energies.sum() + 1e-8
            conf = (block_energies[bid] / total).item()
            bids.append(bid)
            confs.append(conf)
        return torch.tensor(bids), torch.tensor(confs)


# ═══════════════ 查找表（LUT） ═══════════════
class LookupTable:
    """
    查找表 — SNN分桶后的确定性输出
    ─────────────────────────────
    纯桶（桶内只有一个类别）→ O(1) 直接查
    混合桶（桶内多个类别）→ 桶内质心最近邻

    学习阶段：遍历训练集，统计每个桶的类别分布
    推理阶段：零LLM调用，纯查表
    """
    def __init__(self):
        self.table = {}        # bid → 默认类别
        self.sub_table = {}    # bid → {cls: centroid_vector}

    def learn(self, bucket_ids, labels, features):
        """从训练数据建表"""
        bc = defaultdict(list)
        for i, (bid, lbl) in enumerate(zip(bucket_ids, labels)):
            bc[bid].append((features[i], lbl))

        for bid, items in bc.items():
            classes = set(s[1] for s in items)
            if len(classes) == 1:
                # 纯桶：O(1)
                self.table[bid] = list(classes)[0]
            else:
                # 混合桶：存每个类的质心
                centroids = {}
                for cls in classes:
                    cf = torch.stack([s[0] for s in items if s[1] == cls])
                    centroids[cls] = cf.mean(dim=0)
                self.sub_table[bid] = centroids
                counts = Counter(s[1] for s in items)
                self.table[bid] = counts.most_common(1)[0][0]

        pure = sum(1 for b in self.table if b not in self.sub_table)
        mixed = len(self.sub_table)
        log(f"    LUT建表完成: {len(self.table)}桶 (纯{pure} 混{mixed})")

    def query(self, bucket_id, feat=None):
        """查表：纯桶O(1)，混合桶最近邻"""
        if bucket_id in self.sub_table and feat is not None:
            best_cls, best_dist = -1, float('inf')
            for cls, centroid in self.sub_table[bucket_id].items():
                d = torch.norm(feat - centroid).item()
                if d < best_dist:
                    best_dist = d
                    best_cls = cls
            return best_cls
        return self.table.get(bucket_id, -1)


# ═══════════════ 主实验 ═══════════════
def run():
    log("=" * 60)
    log("Stage 1: SNN→LUT查表 vs LLM裸跑 (文本分类)")
    log("=" * 60)
    log("原理：SNN局部连接物理分桶 + LUT质心最近邻 vs LLM每次推理")

    # 生成数据
    docs, cat_names = generate_curriculum_texts()
    C = len(cat_names)
    log(f"数据: {len(docs)}篇, {C}类: {cat_names}")

    # 特征提取
    feats = text_features(docs)
    labels = [d[1] for d in docs]
    log(f"特征: {feats.shape[1]}维 字符bigram")

    # 划分训练/测试
    n_test = 31
    X_test, y_test = feats[:n_test], labels[:n_test]
    X_train, y_train = feats[n_test:], labels[n_test:]
    test_texts = [docs[i][0][:300] for i in range(n_test)]
    N = len(y_train)
    log(f"训练={N} 测试={n_test}")

    # K* 公式计算
    sqN = int(math.sqrt(N))
    K = C  # 每类一个桶
    log(f"N={N} √N={sqN} C={C} K={K}")

    # ════ SNN + LUT ════
    log(f"\n--- SNN局部连接分桶 ---")
    snn = CCSNN(feats.shape[1], K)
    train_bids, _ = snn.compress(X_train)
    bc = Counter(train_bids.tolist())
    log(f"  分桶分布: 使用{len(bc)}/{K}个桶 {dict(bc)}")

    lut = LookupTable()
    lut.learn(train_bids.tolist(), y_train, X_train)

    # LUT推理
    log(f"\n--- LUT查表推理（零LLM调用） ---")
    test_bids, test_confs = snn.compress(X_test)
    lut_ok = 0
    lut_lats = []
    for i in range(n_test):
        t0 = time.time()
        pred = lut.query(test_bids[i].item(), feat=X_test[i])
        lut_lats.append(time.time() - t0)
        if pred == y_test[i]:
            lut_ok += 1
    acc_lut = lut_ok / n_test
    lat_lut = np.mean(lut_lats) * 1000
    log(f"  LUT准确率: {acc_lut:.1%} ({lut_ok}/{n_test})")
    log(f"  LUT延迟:   {lat_lut:.3f}ms/样本")

    # ════ LLM基线 ════
    log(f"\n--- LLM裸跑基线 ---")
    llm_ok = 0
    llm_fail = 0
    llm_lats = []
    for i in range(n_test):
        pred, lat, st = ollama_classify(test_texts[i], cat_names)
        llm_lats.append(lat)
        if st == "ok" and pred == y_test[i]:
            llm_ok += 1
        elif st != "ok":
            llm_fail += 1
        if (i + 1) % 10 == 0:
            log(f"    进度 {i+1}/{n_test} acc={llm_ok/max(1,i+1-llm_fail):.0%}")
    acc_llm = llm_ok / max(1, n_test - llm_fail)
    lat_llm = np.mean(llm_lats) * 1000
    log(f"  LLM准确率: {acc_llm:.1%}")
    log(f"  LLM延迟:   {lat_llm:.1f}ms/样本")

    # ════ 汇总 ════
    passed = acc_lut >= acc_llm
    log(f"\n{'='*60}")
    log(f"Stage 1 结果")
    log(f"{'='*60}")
    log(f"  {'指标':<12} {'LUT查表':<16} {'LLM裸跑':<16}")
    log(f"  {'准确率':<12} {acc_lut:<16.1%} {acc_llm:<16.1%}")
    log(f"  {'延迟':<12} {f'{lat_lut:.3f}ms':<16} {f'{lat_llm:.1f}ms':<16}")
    log(f"  {'LLM调用':<12} {'0次':<16} {f'{n_test}次':<16}")
    log(f"\n  通过标准(LUT≥LLM): {'✅ PASS' if passed else '❌ FAIL'}")

    results = {
        "stage": 1, "task": "文本分类",
        "C": C, "K": K, "N": N, "n_test": n_test,
        "acc_lut": acc_lut, "acc_llm": acc_llm,
        "lat_lut_ms": lat_lut, "lat_llm_ms": lat_llm,
        "buckets_used": len(bc), "passed": passed,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    out = RESULTS_DIR / "stage1_results.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"结果保存: {out}")

if __name__ == '__main__':
    run()
