# 子任务01: 环境搭建与模型测试

## 目标
搭建零样本分类环境，验证Erlangshen-Roberta-110M-NLI模型可在CPU上正常运行。

## 成功标准
- [ ] transformers、torch等依赖成功安装
- [ ] HF_ENDPOINT镜像站配置生效
- [ ] Erlangshen-Roberta-110M-NLI模型成功下载（约400MB）
- [ ] 单条样本分类测试通过
- [ ] CPU推理速度验证（目标：<2秒/item）

## 实施步骤

### 1. 安装依赖
```bash
pip3 install transformers torch tqdm jieba
```

### 2. 配置HuggingFace镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 或永久配置
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

### 3. 编写测试脚本
创建 `scripts/test_zero_shot_model.py`:
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import pipeline
import time

# 加载模型
print("正在加载Erlangshen-Roberta-110M-NLI模型...")
classifier = pipeline(
    "zero-shot-classification",
    model="IDEA-CCNL/Erlangshen-Roberta-110M-NLI",
    device=-1  # CPU
)
print("模型加载成功！")

# 测试数据
test_text = "这是一个关于银行贷款业务的数据集，包含客户信息和交易记录。"
categories = ["金融", "医疗健康", "政府政务", "交通运输", "教育培训",
              "能源环保", "农业", "科技", "商业零售", "文化娱乐", "社会民生"]

# 推理测试
start_time = time.time()
result = classifier(test_text, categories, multi_label=True)
elapsed = time.time() - start_time

print(f"\n推理耗时: {elapsed:.2f}秒")
print(f"\n分类结果:")
for label, score in zip(result["labels"][:5], result["scores"][:5]):
    print(f"  {label}: {score:.3f}")

# 验证性能目标
if elapsed < 2.0:
    print(f"\n✅ 性能测试通过！（{elapsed:.2f}s < 2.0s）")
else:
    print(f"\n⚠️  性能略慢（{elapsed:.2f}s），建议使用batch processing优化")
```

### 4. 运行测试
```bash
python3 scripts/test_zero_shot_model.py
```

### 5. 验证输出
预期输出示例：
```
正在加载Erlangshen-Roberta-110M-NLI模型...
模型加载成功！

推理耗时: 1.23秒

分类结果:
  金融: 0.856
  科技: 0.234
  商业零售: 0.112
  ...

✅ 性能测试通过！（1.23s < 2.0s）
```

## 验证清单
- [ ] 模型文件缓存在 `~/.cache/huggingface/` 目录
- [ ] 测试脚本输出包含正确的分类结果
- [ ] 推理耗时 < 2秒
- [ ] 无报错或警告信息

## 潜在问题与解决方案

### 问题1: 模型下载失败
**症状**: Connection timeout或HTTP 403错误
**解决**:
```bash
# 确认镜像站配置
echo $HF_ENDPOINT
# 手动清除缓存重试
rm -rf ~/.cache/huggingface/*
```

### 问题2: CPU推理太慢（>5秒/item）
**症状**: 单条推理超过5秒
**解决**: 这属于正常范围，后续使用batch processing优化

### 问题3: torch版本冲突
**症状**: ImportError或版本警告
**解决**:
```bash
pip3 install --upgrade torch transformers
```

## 下一步
完成后进入子任务02：实现tags增强脚本
