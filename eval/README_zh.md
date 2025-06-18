# 📘 Evalution

评估方式主要借鉴LLMxMapReduce_v2的方式，并在其codebase上进行了修改了调整

LLMxMapReduce_v2的具体评估方法如下：
https://github.com/thunlp/LLMxMapReduce/tree/main/LLMxMapReduce_V2

---

## 🧩 脚本介绍与使用方法

## AutoSurvey评估


### ✅ 1. `AutoSurvey_trainslate_main_file.py`

将AutoSurvey项目生成的 JSON 文件转换为可评估格式的 JSONL 文件。

#### 📌 功能说明
AutoSurvey 生成的 `.json` 文件中包含 `survey` 内容和引用论文 `reference`，本脚本将其重构为：

- **`title`**  : 论文题目
- **`outline`** : 提取并标准化的提纲
- **`content`** : 正文内容
- **`papers`**  : 引用论文的标题与摘要

#### 🚀 使用方法
评估AutoSurvey内容首先需要下载AutoSurvey数据库中'arxiv_paper_db.json'文件作为匹配数据补全引用论文内容,下载链接：
https://1drv.ms/u/c/8761b6d10f143944/EaqWZ4_YMLJIjGsEB_qtoHsBoExJ8bdppyBc1uxgijfZBw?e=2EIzti

在 `AutoSurvey_trainslate_main_file`的main 函数中修改以下路径：

```python
db_path = 'path/to/AutoSurvey数据库（arxiv_paper_db.json）.json'
autosurvey_json_folder = 'path/to/AutoSurvey生成的文件夹'
output_folder = 'path/to/转换后输出的文件夹'
```

然后运行：

```bash
python AutoSurvey_trainslate_main_file.py
```


---

### 📂 调用脚本执行评估


得到jsonl文件之后，运行 `eval_scisage.sh`，进行评估，获取评估得分。

在评估之前，先下载 punkt_tab：
```python
import nltk
nltk.download('punkt_tab')
```

#### 🚀 使用方法
我们支持本地模型评估和调用api两种评估方式，您可以在local_request.py文件中MODEL_CONFIGS添加您的本地模型参数

在./evaluation/API/model.py下修改APIModel.chat()和APIModel.__chat()下get_from_llm为您想使用的模型

```python
get_from_llm(text, model_name="your_model")
```

构建模型后仅需要运行以下命令
```bash
bash eval_shell.sh "需要评估的文件夹路径"
```

### 🔁 2. SciSage结果评估

使用脚本`eval_scisage.py`，将SciSage或其他系统生成的数据转换为符合评估格式，并进行自动评估。

#### 🚀 使用方法

```python
input_json_file =   "生成scisage论文的路径"
dest_dir = "结果保存的目录"
```

然后运行：

```bash
python eval_scisage.py
```

---



### reference表现评估

使用`reference_eval.py` 评估生成论文中的引用（references）与人工写的论文测试集中引用的重合程度，输出重合数量、重合率与 F1 分数。

#### 🚀 使用方法
在评估前你需要下载SurveyEval和的SurveyScope文件

SurveyScope文件: https://huggingface.co/datasets/BAAI/SurveyScope

在 `main` 函数中修改以下路径：

```python
jsonl_path_ourBenchMark = "path/to/surveyscope.jsonl" #
our_data_papers_file_path = "格式转换之后的文件目录" (eg: ./eval_scisage.py--output_translate_dir)
```

然后运行：

```bash
python reference_eval.py
```

---

## ⚠️ 注意事项

1. 所有路径请使用英文命名，避免使用中文或空格。
2. 输入文件格式应为 `.json` 或 `.jsonl`。
3. `db_path` 文件为AutoSurvey对应数据库。
4. 输出格式统一为 `jsonl`，便于与 LLMxMapReduce_v2 评估系统对接。

