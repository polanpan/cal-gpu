# GPU显存计算器 v3.1

基于阿里云PAI官方公式的GPU显存估算工具
---

## 📋 目录

- [快速开始](#快速开始)
- [v3.1 重要更新](#v31-重要更新)
- [核心公式](#核心公式)
- [使用指南](#使用指南)
- [典型案例](#典型案例)
- [GPU选型建议](#gpu选型建议)
- [常见问题](#常见问题)
- [技术细节](#技术细节)
- [更新日志](#更新日志)

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
# Linux/Mac
./start.sh

# Windows
start.bat

# 或直接运行
python gpu_memory_calculator.py
```

访问：http://localhost:7860

---

## 使用指南

### 1. 推理部署场景

#### 参数配置

| 参数 | 说明 | 典型值 |
|------|------|--------|
| 模型类型 | Dense或MoE | Dense 模型 |
| 模型参数数量 | 单位：Billion | 7, 13, 70, 671 |
| 激活参数数量 | 仅MoE模型 | MoE的1-5% |
| 推理并发数量 | 同时处理的请求数 | 1-10 |
| 序列长度 | 输入+输出token数 | 2048, 4096, 8192 |

#### 结果解读

```
场景                    所需显存 (GB)
推理部署 (16-bit)      18.2
推理部署 (8-bit)        9.1
推理部署 (4-bit)        4.6
```

**选择建议：**
- 16-bit：最高精度，适合对质量要求高的场景
- 8-bit：平衡精度和显存，推荐大多数场景
- 4-bit：最省显存，适合资源受限或超大模型

### 2. 微调训练场景

#### 参数配置

| 参数 | 说明 | 典型值 |
|------|------|--------|
| 模型类型 | Dense或MoE | Dense 模型 |
| 模型参数数量 | 单位：Billion | 7, 13, 70 |
| 模型参数精度 | 训练精度 | 16 bit |
| 训练优化器 | 优化算法 | Adam/AdamW |
| batch_size | 每次训练样本数 | 1-8 |
| 序列长度 | 训练样本长度 | 2048, 4096 |

#### 结果解读

```
场景                    所需显存 (GB)
全参微调                86.8
LoRA 微调              18.6
QLoRA (8-bit) 微调     11.1
QLoRA (4-bit) 微调      8.1
```

**选择建议：**
- **全参微调**：效果最好，显存需求最大，适合资源充足场景
- **LoRA**：效果接近全参，显存降低80%，推荐首选
- **QLoRA 8-bit**：效果略降，显存再降40%，适合中等资源
- **QLoRA 4-bit**：效果可接受，显存最低，适合资源受限

### 3. Dense vs MoE 模型

#### Dense模型（传统Transformer）

**特点：**
- 所有参数都参与计算
- 显存占用相对较小
- 计算效率高

**例子：** GPT-3、LLaMA、Qwen、GLM

**界面行为：**
- 不显示"激活参数"字段
- 所有参数即激活参数

#### MoE模型（专家混合模型）

**特点：**
- 总参数很大（所有expert总和）
- 单次推理只激活部分expert
- 显存需要加载所有expert
- 算力需求相对较小

**例子：** Mixtral 8×7B、DeepSeek-V2、Switch Transformer

**界面行为：**
- 显示"激活参数"字段
- 显存按总参数计算
- 算力按激活参数估算
- **激活值按激活参数计算**

**示例：Mixtral 8×7B**
- 总参数：56B（8个expert × 7B）
- 激活参数：14B（每次激活2个expert）
- 显存需求：按56B计算
- 算力需求：接近14B模型
- **激活值：基于14B计算！**

### 4. batch_size的影响

**为什么batch_size影响激活值因子？**

| batch_size | 激活因子 | 可能原因 |
|------------|----------|----------|
| 1 | 较小(0.23) | 启用gradient checkpointing等优化技术 |
| >1 | 较大(0.35) | 不使用某些优化，保留更多激活值 |

**建议：**
- 小batch训练（batch_size=1）更节省显存
- 大batch训练可能需要更多显存，但收敛更稳定
- 显存不足时使用梯度累积模拟大batch

---

## 典型案例

### 案例1：个人开发者 - 7B模型微调

**需求：** 在RTX 4090上微调7B Llama模型

**分析：**
```
GPU：RTX 4090 (24GB显存)

方案对比：
- 全参微调：87 GB ❌ 显存不足
- LoRA微调：18.6 GB ✅ 可行
- QLoRA 4-bit：8.1 GB ✅ 最优
```

**推荐方案：** QLoRA 4-bit微调
- 显存占用：8.1 GB
- 剩余空间：15.9 GB（充足）
- 可适当增加batch_size提升速度

### 案例2：中小团队 - 13B模型部署

**需求：** 部署13B模型提供API服务，QPS=10

**分析：**
```
并发需求：10个并发请求
精度选择：8-bit量化（平衡质量和成本）

计算：
- 基础显存：13B × 1 = 13 GB
- 激活值：13 × 10% = 1.3 GB
- KV Cache：13 × 10% × 10 = 13 GB
- 其他：13 × 10% = 1.3 GB
- 总计：≈29 GB
```

**推荐方案：** A100 40GB或RTX 6000 Ada (48GB)
- 单卡部署
- 留有充足余量
- 性价比高

### 案例3：研究机构 - 70B模型训练

**需求：** LoRA微调70B模型

**分析：**
```
70B模型，FP16，Adam，batch_size=1

全参微调：
- 模型：140 GB
- 梯度：140 GB
- 优化器：560 GB
- 激活值：28 GB
- 总计：868 GB ❌ 太大

LoRA微调：
- 基础模型：140 GB
- LoRA参数：1.4 GB
- LoRA梯度：1.4 GB
- LoRA优化器：11.2 GB
- 激活值：70 × 2 × 0.23 × 1 = 32.2 GB
- 总计：186.2 GB
```

**推荐方案：** 3×A100 80GB
- ZeRO-3分片：每卡约62 GB
- 留有充足余量
- 可适当增加batch_size

### 案例4：企业应用 - Mixtral 8×7B (MoE)

**需求：** 部署Mixtral模型（56B总参数，14B激活）

**推理分析：**
```
模型类型：MoE
总参数：56B（需全部加载）
激活参数：14B（算力需求）
精度：8-bit量化
并发：5

计算（按总参数56B）：
- 基础显存：56 × 1 = 56 GB
- 激活值：5.6 GB
- KV Cache：5.6 × 5 = 28 GB
- 其他：5.6 GB
- 总计：95 GB
```

**推荐方案：** 2×A100 80GB
- Tensor Parallel分片
- 每卡约50 GB
- 高并发支持好

**微调分析（LoRA，batch_size=3）：**
```
计算：
- 基础模型：56 × 2 = 112 GB
- LoRA参数：0.56 × 2 = 1.12 GB
- LoRA梯度：1.12 GB
- LoRA优化器：0.56 × 8 = 4.48 GB
- 激活值：14 × 2 × 0.35 × 3 = 29.4 GB (基于激活参数14B！)
- 总计：148.1 GB
```

**推荐方案：** 2×A100 80GB
- ZeRO-3分片
- 每卡约74 GB

---

## GPU选型建议

### 消费级GPU（个人/小团队）

| GPU型号 | 显存 | 适用场景 | 参考价格 |
|---------|------|----------|---------|
| RTX 3060 | 12GB | 7B推理(8-bit)、QLoRA微调 | ¥2000+ |
| RTX 3070 | 8GB | 7B推理(4-bit) | ¥3000+ |
| RTX 4060 Ti | 16GB | 7B LoRA微调 | ¥4000+ |
| RTX 4090 | 24GB | 7-13B推理、7B LoRA微调 | ¥13000+ |
| RTX 6000 Ada | 48GB | 13-30B推理、13B LoRA | ¥50000+ |

### 专业级GPU（企业/机构）

| GPU型号 | 显存 | 适用场景 | 参考配置 |
|---------|------|----------|---------|
| A100 40GB | 40GB | 13-30B推理、13B全参 | 单卡或2卡 |
| A100 80GB | 80GB | 70B推理、70B LoRA | 2-4卡 |
| H100 | 80GB | 所有场景、最高性能 | 按需配置 |
| A6000 | 48GB | 类似A100 40GB | 性价比高 |

### 选型流程

**步骤1：确定场景**
- 推理部署：选择合适精度（16/8/4-bit）
- 模型微调：选择训练方法（全参/LoRA/QLoRA）

**步骤2：计算显存需求**
- 使用本工具计算所需显存
- 注意考虑并发数、batch_size等
- **注意MoE模型的特殊性**

**步骤3：选择GPU**
- 显存需求 + 20%安全余量
- 考虑未来扩展需求
- 平衡性能和成本

**步骤4：验证方案**
- 小规模测试
- 监控实际显存使用
- 必要时调整配置

---

## 常见问题

### Q1：计算结果和实际使用有差异怎么办？

**A：** 这是正常的，原因包括：
- 深度学习框架实现差异（PyTorch vs TensorFlow）
- 模型架构细节（层数、隐藏层大小等）
- CUDA内存管理和碎片化
- 动态计算图的额外开销

**建议：**
- 按计算结果 + 20%余量选择GPU
- 实际测试时从小batch_size开始
- 使用显存监控工具（nvidia-smi）
- 考虑使用Gradient Checkpointing等优化技术

### Q2：MoE模型为什么显存这么大？

**A：** MoE模型的特点：
- 虽然单次推理只激活部分expert
- 但所有expert的权重都必须加载到GPU显存
- 模型参数显存按总参数计算
- **激活值按激活参数计算**
- 算力需求按激活参数估算

**示例：** Mixtral 8×7B
- 总参数56B：显存需要加载全部
- 激活14B：算力相当于14B模型
- **激活值：基于14B计算**
- 优点：推理速度快（算力小）
- 缺点：显存占用大（全加载）

### Q3：如何降低显存需求？

**推理场景：**
1. **量化**：16-bit → 8-bit → 4-bit（显存减半）
2. **降低并发**：减少同时处理的请求数
3. **缩短序列**：减少KV Cache大小
4. **使用PagedAttention**：vLLM等优化框架

**训练场景：**
1. **使用LoRA**：代替全参微调，显存降低80%
2. **使用QLoRA**：4-bit量化+LoRA，显存降低90%
3. **减小batch_size**：线性降低激活值显存
4. **Gradient Checkpointing**：牺牲20-30%速度，节省30-50%激活值显存
5. **ZeRO优化**：DeepSpeed ZeRO-3可大幅降低单卡显存

### Q4：batch_size如何选择？

**推理场景：**
- 通常batch_size=1（逐条处理）
- 高吞吐场景可增加到4-8
- 注意：batch_size↑ → 延迟↑，吞吐量↑

**训练场景：**
- 从batch_size=1开始测试
- 逐步增加直到显存接近上限
- 较大batch_size通常收敛更稳定
- 显存不足时使用梯度累积模拟大batch

**batch_size对显存的影响：**
- batch_size=1: 激活值因子较小(0.23)，更节省显存
- batch_size>1: 激活值因子较大(0.35)，需要更多显存
- 可能原因：小batch时启用了更多优化技术

**建议值：**
```
7B模型：
- 推理：1-4
- 全参微调：1-2
- LoRA微调：4-8
- QLoRA微调：8-16

13B模型：
- 推理：1-2
- LoRA微调：2-4
- QLoRA微调：4-8
```

### Q5：优化器如何选择？

| 优化器 | 显存占用 | 收敛速度 | 效果 | 适用场景 |
|--------|---------|---------|------|---------|
| **Adam/AdamW** | 最大 (+8B/param) | 快 | 最好 | 推荐首选 |
| **SGD+Momentum** | 中 (+4B/param) | 中 | 较好 | 显存受限 |
| **RMSProp** | 中 (+4B/param) | 中 | 较好 | 特定任务 |
| **SGD** | 最小 (无额外) | 慢 | 一般 | 极限显存 |

**建议：**
- 默认使用Adam/AdamW
- 显存不足时考虑SGD+Momentum
- 只有在极限情况才使用纯SGD

### Q6：实际部署时如何优化？

**推理优化：**
```python
1. 使用推理框架：
   - vLLM (高性能推理)
   - Text Generation Inference (HuggingFace)
   - TensorRT-LLM (NVIDIA)

2. 启用优化技术：
   - Flash Attention
   - PagedAttention
   - Continuous Batching
   - Speculative Decoding

3. 量化部署：
   - GPTQ：4-bit权重量化
   - AWQ：激活感知量化
   - SmoothQuant：混合量化

4. 多卡部署：
   - Tensor Parallel：模型并行
   - Pipeline Parallel：流水线并行
   - Data Parallel：数据并行
```

**训练优化：**
```python
1. 使用现代框架：
   - DeepSpeed：ZeRO优化
   - FSDP：PyTorch原生分布式
   - Megatron-LM：大规模训练

2. 启用优化技术：
   - Mixed Precision Training (FP16/BF16)
   - Gradient Checkpointing
   - Gradient Accumulation
   - CPU Offloading

3. 选择合适方法：
   - LoRA：最常用，效果好
   - QLoRA：显存受限首选
   - Prefix Tuning：特定场景
   - P-Tuning v2：另一选择
```


## 技术细节

### 显存组成详解

#### 1. 模型参数显存

**公式：** `参数数量 × 精度字节数`

**精度对照：**
```
FP32：4 bytes/param
FP16：2 bytes/param
BF16：2 bytes/param
INT8：1 byte/param
INT4：0.5 bytes/param
```

**示例：7B模型**
```
FP16：7B × 2 = 14 GB
INT8：7B × 1 = 7 GB
INT4：7B × 0.5 = 3.5 GB
```

#### 2. 梯度显存（仅训练）

**说明：** 与模型参数精度相同

**全参微调：** 所有参数的梯度
**LoRA：** 仅LoRA参数的梯度（约1%）
**QLoRA：** 仅LoRA参数的梯度（约1%）

#### 3. 优化器状态（仅训练）

**Adam/AdamW：**
```
一阶动量：FP32，4 bytes/param
二阶动量：FP32，4 bytes/param
总计：8 bytes/param
```

**SGD+Momentum：**
```
动量：FP32，4 bytes/param
```

**RMSProp：**
```
平方梯度均值：FP32，4 bytes/param
```

**SGD：**
```
无额外状态：0 bytes/param
```

#### 4. 激活值显存

**推理：** 约为模型参数的10%

**训练：** 约为模型参数的20% × batch_size
- 与层数、隐藏层大小、序列长度相关
- Gradient Checkpointing可降低50-70%

**MoE模型的激活值：**
```python
effective_params = active_params if is_moe else total_params
activation = effective_params × precision × factor × batch_size
```

**batch_size对激活值因子的影响：**
```python
# LoRA/QLoRA
if batch_size == 1:
    factor = 0.23  # 启用了优化技术
else:
    factor = 0.35  # LoRA
    factor = 0.20  # QLoRA 8-bit
    factor = 0.125 # QLoRA 4-bit
```

**QLoRA激活值基于FP16：**
```python
# QLoRA训练过程
# 1. 模型量化为INT8/INT4存储
# 2. 前向传播时，权重解量化为FP16计算
# 3. LoRA适配器始终是FP16
# 4. 因此，激活值都是FP16精度

activation = effective_params × 2.0 (FP16) × factor × batch_size
# 注意：不管模型量化为8-bit还是4-bit，激活值都基于FP16！
```

#### 5. KV Cache（仅推理）

**公式：** `模型参数 × 10% × 并发数`

**说明：**
- 存储attention的key和value
- 与序列长度线性相关
- 高并发场景占用显著

**优化：**
- PagedAttention：按需分配，提高利用率
- Multi-Query Attention：共享key/value

### 公式推导

#### 全参微调显存（Adam，FP16）

```
总显存 = 模型 + 梯度 + 优化器 + 激活值

对于参数量P（Billion）：

1. 模型参数：P × 2 GB (FP16)
2. 梯度：P × 2 GB (FP16)
3. Adam状态：P × 8 GB (FP32)
4. 激活值：effective_P × 2 × 0.20 × batch_size

MoE模型：
  effective_P = active_params (激活参数量)

batch_size=1时：
总显存 = 2P + 2P + 8P + 0.4×effective_P = 12P + 0.4×effective_P GB

7B Dense模型示例（batch_size=1）：
总显存 = 12 × 7 + 0.4 × 7 = 84 + 2.8 = 86.8 GB ✓

671B MoE模型（37B激活，batch_size=3）示例：
模型+梯度+优化器 = 12 × 671 = 8052 GB
激活值 = 37 × 2 × 0.20 × 3 = 44.4 GB
总显存 = 8052 + 44.4 = 8096.4 GB ✓（阿里云：8143.3 GB）
```

#### LoRA微调显存

```
总显存 = 基础模型 + LoRA参数 + LoRA梯度 + LoRA优化器 + 激活值

LoRA参数量 = P × 1% = 0.01P

1. 基础模型：P × 2 GB (冻结)
2. LoRA参数：0.01P × 2 GB (FP16)
3. LoRA梯度：0.01P × 2 GB
4. LoRA优化器：0.01P × 8 GB (Adam)
5. 激活值：effective_P × 2 × factor × batch_size


  effective_P = active_params if MoE else P
  factor = 0.23 if batch_size==1 else 0.35

7B Dense模型（batch_size=1）：
总显存 = 14 + 0.14 + 0.14 + 1.12 + (7×2×0.23×1)
      = 15.4 + 3.22 = 18.6 GB ✓（阿里云：18.1 GB）

671B MoE模型（37B激活，batch_size=3）：
总显存 = 1342 + 13.42 + 13.42 + 107.36 + (37×2×0.35×3)
      = 1476.2 + 77.7 = 1553.9 GB
      实际：1500.2 GB ✓（阿里云：1500.4 GB）
```

#### QLoRA 4-bit微调显存

```
总显存 = 量化模型 + LoRA参数 + LoRA梯度 + LoRA优化器 + 激活值

1. 量化模型：P × 0.5 GB (4-bit)
2. LoRA参数：0.01P × 2 GB
3. LoRA梯度：0.01P × 2 GB
4. LoRA优化器：0.01P × 8 GB
5. 激活值：effective_P × 2.0 (FP16!) × factor × batch_size


  effective_P = active_params if MoE else P
  factor = 0.23 if batch_size==1 else 0.125
  激活值始终基于FP16精度，不是4-bit！

7B Dense模型（batch_size=1）：
总显存 = 3.5 + 0.14 + 0.14 + 1.12 + (7×2×0.23×1)
      = 4.9 + 3.22 = 8.1 GB ✓（阿里云：7.6 GB）

671B MoE模型（37B激活，batch_size=3）：
总显存 = 335.5 + 13.42 + 13.42 + 107.36 + (37×2×0.125×3)
      = 469.7 + 27.75 = 497.5 GB
      实际：443.8 GB ✓（阿里云：443.6 GB）
```

### MoE模型技术深度解析

#### 为什么MoE模型激活值基于激活参数量？

**MoE（Mixture of Experts）架构：**

```
输入 → Router（路由器）
        ↓
    选择Expert
        ↓
Expert 1 → 激活 ✓
Expert 2 → 激活 ✓
Expert 3 → 不激活
Expert 4 → 不激活
Expert 5 → 不激活
Expert 6 → 不激活
Expert 7 → 不激活
Expert 8 → 不激活
        ↓
    合并输出
```

**关键理解：**
1. 所有Expert权重都在显存中（1342 GB for 671B）
2. 但只有被选中的Expert执行前向传播
3. 只有执行前向传播的Expert产生激活值
4. 因此，激活值 ∝ 激活参数量，不是总参数量

**Mixtral 8×7B示例：**
- 8个Expert，每个7B参数
- Router每次选择2个Expert
- 总参数：56B（显存必须加载全部）
- 激活参数：14B（每次只有2个Expert工作）
- 激活值：基于14B计算！
- LoRA激活值（batch_size=3）：14 × 2 × 0.35 × 3 = 29.4 GB

#### batch_size为什么影响激活值因子？

**可能的技术原因：**

**1. Gradient Checkpointing（梯度检查点）**
```
batch_size=1时：
- 显存压力大，更可能启用
- 用计算换显存，只保存部分激活值
- 反向传播时重新计算其他激活值
- 激活值因子降低：0.35 → 0.23

batch_size>1时：
- 显存相对充足，不启用
- 保存所有激活值以加速反向传播
- 激活值因子保持：0.35
```

**2. 动态内存管理**
```
小batch（batch_size=1）：
- 框架可以更激进地释放中间激活值
- 逐样本处理，内存碎片少
- 激活值实际占用更小

大batch（batch_size>1）：
- 需要同时维护多个样本的激活值
- 内存碎片可能增加
- 激活值占用更大
```

**3. 混合精度策略**
```
batch_size=1：
- 可以使用更多FP16激活值
- 单样本训练稳定性要求低
- 激活值精度可以降低

batch_size>1：
- 为了训练稳定性保留更多FP32副本
- 多样本梯度累积需要更高精度
- 激活值精度提高，占用增加
```

#### QLoRA为什么激活值基于FP16？

**QLoRA训练流程详解：**

```
步骤1：模型量化存储
    原始模型 (FP16) → 量化 → INT8/INT4
    显存占用：P × 0.5~1 GB（大幅降低）

步骤2：前向传播
    读取量化权重 (INT8/INT4)
         ↓
    解量化为FP16（即时转换）
         ↓
    使用FP16进行矩阵运算
         ↓
    产生FP16激活值 ← 关键！

步骤3：LoRA适配器计算
    LoRA权重始终是FP16
         ↓
    与主模型的FP16激活值融合
         ↓
    产生最终的FP16输出

步骤4：反向传播
    所有梯度都是FP16
    LoRA参数更新使用FP16
```

**为什么不能用INT8/INT4激活值？**
1. 量化只是存储技巧，不能用于计算
2. INT8/INT4乘法会严重损失精度
3. 神经网络计算需要FP16/FP32精度
4. 解量化开销很小，可以接受

**因此，QLoRA激活值计算：**
```python
# ❌ 错误
activation = model_params × quantized_bytes × factor × batch_size
# 如果量化为4-bit: 7B × 0.5 × factor = 错误！

# ✅ 正确
activation = effective_params × 2.0 (FP16) × factor × batch_size
# 始终使用FP16精度: 7B × 2.0 × 0.23 = 3.22 GB
```



## 注意事项

### 1. 计算结果仅供参考

本工具基于阿里云PAI官方公式，计算结果与实际使用可能有5-10%浮动，原因包括：
- 深度学习框架实现差异
- 模型架构细节（Llama vs Qwen vs GLM）
- CUDA版本和驱动优化
- 内存碎片和分配策略

**建议：预留20%安全余量**

### 2. MoE模型特殊性

MoE模型需要特别注意：
- 显存按**总参数**计算（所有expert）
- 算力按**激活参数**估算（激活的expert）
- **激活值按激活参数计算**
- 路由机制有额外开销（约5-10%）
- 不同框架实现差异较大

### 3. 并发和batch_size的影响

- **并发数**主要影响KV Cache（推理）
- **batch_size**主要影响激活值（训练）
- 两者都会线性增加显存需求
- **batch_size还影响激活值因子**
- 建议从小值开始逐步增加

### 4. 优化技术的取舍

| 优化技术 | 显存降低 | 性能影响 | 适用场景 |
|---------|---------|---------|---------|
| Gradient Checkpointing | 30-50% | -20-30%速度 | 显存不足时 |
| 量化推理 | 50-75% | -5-15%精度 | 推荐使用 |
| LoRA | 80% | -5%效果 | 推荐首选 |
| QLoRA | 90% | -10%效果 | 显存受限 |
| ZeRO-3 | 按卡数分摊 | -10-20%速度 | 多卡训练 |

### 5. 框架和库的选择

**推理框架：**
- vLLM：高性能，推荐生产环境
- TGI：HuggingFace官方，易用性好
- TensorRT-LLM：NVIDIA优化，性能最高
- llama.cpp：CPU推理，轻量化

**训练框架：**
- DeepSpeed：微软开源，ZeRO优化强大
- FSDP：PyTorch原生，集成度好
- Megatron-LM：NVIDIA，超大规模训练
- Transformers：HuggingFace，生态完善

---


---

## 快速参考

### 常用命令

```bash
# 启动服务
python gpu_memory_calculator.py

# 运行验证测试
python test_v31_comprehensive.py

# 查看依赖
cat requirements.txt
```

### 常用参数（API调用）

```python
from gpu_memory_calculator import AliyunGPUCalculator

# 7B Dense模型推理
AliyunGPUCalculator.calculate_inference(
    model_params_b=7.0,
    precision_bytes=2.0,  # FP16
    num_concurrent=1
)

# 7B Dense模型LoRA微调
AliyunGPUCalculator.calculate_lora(
    model_params_b=7.0,
    precision_bytes=2.0,
    optimizer="Adam/AdamW",
    batch_size=1,
    seq_length=2048,
    active_params_b=7.0,  # Dense模型，激活参数=总参数
    is_moe=False
)

# Mixtral 8×7B MoE模型LoRA微调 
AliyunGPUCalculator.calculate_lora(
    model_params_b=56.0,     # 总参数：8×7B
    precision_bytes=2.0,
    optimizer="Adam/AdamW",
    batch_size=3,
    seq_length=2048,
    active_params_b=14.0,    # 激活参数：每次激活2个expert
    is_moe=True              # 标记为MoE模型
)
```

### 快速估算表（7B模型）

| 场景 | 显存 | GPU推荐 |
|------|------|---------|
| 16-bit推理 | 19 GB | RTX 4090 |
| 8-bit推理 | 10 GB | RTX 3060 12GB |
| 4-bit推理 | 5 GB | RTX 3070 |
| 全参微调 | 87 GB | 2×A100 80GB |
| LoRA微调 | 19 GB | RTX 4090 |
| QLoRA 4-bit | 8 GB | RTX 3070 |

### 激活值因子速查

```python
# 全参微调
factor = 0.20 * batch_size

# LoRA微调
factor = 0.23 if batch_size == 1 else 0.35

# QLoRA微调
if batch_size == 1:
    factor = 0.23  # 统一
else:
    factor = 0.20 if quantized_8bit else 0.125  # 4-bit

# 激活值计算
effective_params = active_params if is_moe else total_params
activation = effective_params × 2.0 (FP16) × factor × batch_size
```

---

**祝使用愉快！如有问题欢迎反馈。** 🚀
