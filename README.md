# Penalty Monitor

足球 VAR 点球预判研究系统。验证"疑似点球/VAR 信号是否能早于市场充分反应"这个研究假设。

系统只做**纸面交易**和审计，不会提交真钱订单。

## 一句话理解

你不是在"猜比赛输赢"，而是在做这件事：

1. **感知**：观察比赛中疑似点球的场景（禁区内接触、倒地、裁判听 VAR、走向屏幕等）。
2. **证据**：把这些信号 + 盘口快照 + 比赛上下文打包成 `EvidenceEvent`。
3. **预测**：用可解释规则计算"这会不会判点球"。
4. **纸面交易**：达到阈值时，模拟一笔"买 YES"的纸面订单。
5. **赛后审计**：根据真实判罚，判断这次预测是对是错，并记录 PnL。

```
sensor -> evidence -> prediction -> paper_trade -> audit -> evolution_candidate
感知   -> 证据包   -> 预测       -> 纸面交易     -> 审计  -> 进化候选
```

---

## 一场比赛完整 workflow

假设今晚 20:00 有 Arsenal vs Chelsea，你想边看盘边监控疑似点球。

### 赛前（开球前 5 分钟）

**1. 创建实时状态文件，填入比赛基本信息**

```bash
python main.py update-state \
  --file /tmp/match_state.json \
  --home Arsenal --away Chelsea \
  --market-id demo_market --token-id demo_token \
  --best-ask 0.50 --liquidity-usd 2500 \
  --attacking-side home
```

这会创建 `/tmp/match_state.json`，后续你可以持续覆盖它。

**2. 验证配置文件没问题**

```bash
python main.py validate --input data/samples/replay_events.json
```

（验证你之前积累的历史事件文件格式是否正确。）

---

### 赛中（开球后，两个终端配合）

**终端 1：启动实时监控**

```bash
python main.py live \
  --video-source 0 \
  --match-id arsenal-chelsea-2026-05-08 \
  --state-file /tmp/match_state.json \
  --sample-interval-sec 2 \
  --only-alerts
```

- `--video-source 0`：摄像头编号，或换成视频文件路径 `/path/to/match.mp4`
- `--state-file`：系统每秒读取这个文件里的信号
- `--only-alerts`：只打印达到阈值的事件，减少刷屏
- `--sample-interval-sec 2`：每 2 秒抽一帧

启动后，终端会安静等待。当状态文件里的信号达到阈值时，它会：
- **终端打印**高亮提醒
- **桌面弹窗通知**（Linux: notify-send / macOS: osascript）
- **终端响铃**（`\a`）

**终端 2：边看盘边快速更新信号**

当比赛中出现疑似点球场景时（禁区内接触、倒地、裁判捂耳机、走向 VAR 屏幕），在另一个终端快速输入：

```bash
python main.py update-state \
  --file /tmp/match_state.json \
  --box-contact 0.92 --fall 0.86 --protest 0.72 \
  --ref-earpiece 0.90 --ref-var-walk 0.84 --stoppage 0.70 \
  --minute 72
```

`live` 进程会在下一次抽帧时读取到这些信号，立即输出概率判断。

> **为什么需要手动输入信号？**
> 当前 MVP 阶段还没有接入真正的足球动作识别 AI 模型（后续可以替换 `VisionSensorAgent`）。
> 现在你的角色是"人肉传感器"：看到什么场景，就把对应的信号分数写进去。
> 这正好帮助你**积累高质量标注样本**——你每写一次信号，系统就记录一次证据包，赛后可以回放。

---

### 赛后（比赛结束）

**1. 查看运行结果**

```bash
# 找到刚才的运行目录
ls -lt data/runs/ | head -5

# 分析该场比赛的指标
python main.py analyze --run-dir data/runs/20260508_204500 --format html --output match_report.html
```

**2. 对未标注事件进行人工标注**

```bash
python main.py annotate --run-dir data/runs/20260508_204500 --output annotated.json
```

交互式输入 `y`（点球）/ `n`（无点球），把真实结果写回去。

**3. 积累足够样本后，参数调优**

```bash
python main.py tune --input annotated.json --top-k 10
```

找到在你数据集上表现最好的概率阈值和置信度组合。

**4. 批量策略对比**

```bash
python main.py batch-replay \
  --input annotated.json \
  --overrides '{"decision.probability_threshold":[0.7,0.75,0.8],"paper.cooldown_sec":[0,30]}' \
  --output-dir data/runs/batch
```

---

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

需要 `opencv-python` 才能使用 `live` 模式。

---

## 常用命令速查

### 实时模式（核心）

```bash
# 读取摄像头，持续监控
python main.py live --video-source 0 --match-id demo-match

# 读取本地视频文件，只处理 20 帧（快速测试）
python main.py live --video-source match.mp4 --match-id demo --max-frames 20

# 只打印达到阈值的事件，并启用桌面通知
python main.py live \
  --video-source 0 \
  --match-id demo \
  --state-file /tmp/match_state.json \
  --only-alerts
```

### 状态文件快速更新（赛中必用）

```bash
python main.py update-state \
  --file /tmp/match_state.json \
  --box-contact 0.9 --fall 0.8 --ref-earpiece 0.9 \
  --minute 72 --attacking-side home
```

### 离线回放（测试/验证）

```bash
python main.py replay --input data/samples/replay_events.json
python main.py sample                              # 打印内置样例事件
```

### 分析与调优

```bash
python main.py analyze --run-dir data/runs/smoke
python main.py review --run-dir data/runs/smoke --only-candidates
python main.py tune --input data/samples/replay_events.json --top-k 10
python main.py batch-replay \
  --input data/samples/replay_events.json \
  --overrides '{"decision.probability_threshold":[0.7,0.75]}'
```

### 数据质量

```bash
python main.py validate --input data/samples/replay_events.json
```

### Polymarket 研究

```bash
python main.py markets --tag-id 82
python main.py geoblock-check
```

---

## 关键字段说明

### `signals`（实时状态文件里的信号）

所有分数范围 `0.0 ~ 1.0`，越高代表该信号越强。

| 字段 | 含义 |
|---|---|
| `box_contact_score` | 禁区内身体接触强度 |
| `fall_score` | 进攻方倒地/失衡强度 |
| `protest_score` | 球员抗议强度 |
| `ref_earpiece_score` | 裁判捂耳机/听 VAR 通讯的置信度 |
| `ref_var_walk_score` | 裁判走向 VAR 屏幕的置信度 |
| `whistle_or_stoppage_score` | 哨声、比赛暂停或异常停顿信号 |

### `market_snapshot`（盘口）

| 字段 | 含义 |
|---|---|
| `market_id` / `token_id` | 市场映射；缺失时不会创建纸面订单 |
| `best_ask` | 当前卖一价（系统用它作为参考买入价） |
| `liquidity_usd` | 流动性估计；低于配置阈值时阻止订单 |

### `match_context`（比赛上下文）

| 字段 | 含义 |
|---|---|
| `home` / `away` | 球队名称 |
| `score_home` / `score_away` | 比分 |
| `minute` | 比赛分钟 |
| `attacking_side` | 疑似获利方：`home`、`away`、`unknown` |
| `var_history_penalty_rate` | 裁判历史 VAR 点球倾向 |

---

## 配置说明

配置文件 [`config.yaml`](config.yaml) 默认保持研究模式：

```yaml
# 决策层
mode: paper
enable_real_trading: false

decision:
  probability_threshold: 0.75    # 概率达到此值才考虑交易
  min_confidence: 0.55           # 最低置信度
  model_version: rule-v0.1
  # 信号权重：可以调整来实验不同策略
  weight_box_contact: 0.25
  weight_fall: 0.15
  weight_protest: 0.15
  weight_ref_earpiece: 0.20
  weight_ref_var_walk: 0.20
  weight_stoppage: 0.05

# 纸面交易
paper:
  simulated_size_usd: 100.0
  max_loss_usd: 25.0
  min_liquidity_usd: 0.0
  cooldown_sec: 30.0             # 同一比赛 30 秒内不重复下单

# 通知
notifications:
  enabled: true
  desktop: true                  # Linux/macOS 桌面弹窗
  sound: true                    # 终端响铃
  min_probability: 0.75
```

---

## 输出文件

一次 `live` / `replay` 会在运行目录下生成：

- `evidence.jsonl`：原始证据包
- `predictions.jsonl`：预测结果
- `paper_orders.jsonl`：纸面订单
- `audit.jsonl`：赛后审计记录
- `evolution_candidates.jsonl`：待人工审核的策略改进候选
- `summary.json`：运行摘要（含 threshold、PnL 等元信息）
- `frames/`：仅 `live` 模式保存的抽帧图片

---

## 安全默认值

- `mode: paper` — 只模拟，不真钱交易
- `enable_real_trading: false` — 不可绕过
- 策略进化只创建 `status=pending_human_review` 的候选项
- 缺少视频、盘口或比赛上下文时，系统会降级记录，不会强行交易

---

## 常见问题

**Q: `live` 模式说需要 opencv-python，怎么装？**

```bash
pip install opencv-python
```

**Q: 我不想保存抽帧图片，怎么关？**

```bash
python main.py live ... --no-save-frames
```

**Q: 我没有摄像头，怎么测试 live 模式？**

用本地视频文件代替：

```bash
python main.py live --video-source /path/to/match.mp4 --match-id test --max-frames 20
```

**Q: 桌面通知不工作？**

- Linux: 确保安装了 `notify-send`（通常在 `libnotify-bin` 包中）
- macOS: 默认支持，无需额外安装
- Windows: 当前仅支持终端响铃，桌面通知需安装 `win10toast`（后续支持）

**Q: 什么时候接入真正的 AI 视觉模型？**

当前阶段先用手动输入信号来**积累标注样本**和**验证工作流**。当样本量足够、指标可信后，再把 `VisionSensorAgent` 替换成真实模型，用同一套 `replay` 审计框架验证它是否真的比人工规则更好。
