## 项目说明：LLM Agent 生成跨函数 PBT（Property-Based Test）

本项目的目标是：给定一个待测 Python 项目目录，利用 LLM agent 自动分析代码、抽取与性质（property）相关的信息，并生成 **跨函数 / 跨模块的 property-based tests（PBT）**，并使用 `pylint` 做静态检查。

当前仓库仅提供 **项目框架**，方便你根据自己的 LLM / 向量检索基础设施进行扩展。

---

## 整体工作流

1. **输入被测项目根目录 `project_dir`**
   - 扫描整个目录，构建 **Mac 风格的文件树字符串**（类似 `tree` 命令在 macOS 下的输出）。
2. **分析项目中的所有函数（包括类方法）**
   - 使用 `ast` 解析所有 `.py` 文件。
   - 为每个函数/方法记录：
     - 模块路径（如：`package_a.module_b.function_c` 或 `package_a.module_b.ClassD.method_e`）
     - 源文件相对路径、起止行号、源码文本等。
   - 用户可以提供一组目标模块路径；若不给，则默认“全测”。
3. **基于源码 + 文件目录，让 LLM 选出与属性相关的 Top-N 文件**
   - 对于每一个待测函数：
     - 提供其源码 + 整体文件树。
     - 提示 LLM：**找出与其 property 相关的 Top-N 文件**（例如：`encode` 与 `decode.py` / `decoder` 等）。
     - 再把这些 Top-N 文件的内容提供给 LLM，让其提取 “对生成 PBT 有意义的信息”。
   - 在提示中要强调：**任何提到的文件、函数、类、方法都必须尽量携带“模块路径”元数据（metadata）**，便于后续 import。
4. **基于向量检索的 cross-function 信息补充**
   - 对于每一个待测函数：
     - 给 LLM 其源码，让 LLM 生成一个适合作为向量检索 query 的描述（重点是与 property 相关的语义，比如 encode/解码/校验/逆操作等）。
     - 使用外部或内置的向量检索，找到相关代码片段 / 文档。
     - 将检索结果再喂给 LLM，请其提取“对生成 PBT 有意义的信息”（同样要带上模块路径 metadata）。
5. **生成 PBT 并用 `pylint` 检查**
   - 基于步骤 3 和 4 得到的信息，LLM 生成 PBT 代码（例如基于 `hypothesis`）。
   - 对生成的测试文件运行 `pylint`，收集并反馈问题。

> 注意：**步骤 3 和 4 必须在输出结构中附带模块路径 metadata**，并在提示词中反复强调，以确保生成的 PBT 能正确 import 相关依赖。

---

## 代码结构概览

```text
pbt_gen/
  README.md
  requirements.txt
  pbt_gen/
    __init__.py
    cli.py                 # CLI 入口：解析参数、驱动主流程
    config.py              # 配置与常量（比如 Top-N、LLM 模型名等）
    file_tree.py           # 目录扫描 & Mac 风格文件树生成
    function_finder.py     # 用 AST 提取函数/方法及模块路径
    models.py              # 数据结构（FunctionInfo, FileInfo, RetrievalResult 等）
    llm_client.py          # 调用 LLM 的通用 Client（留接口给你接 OpenAI/其他）
    prompts.py             # 所有与 LLM 交互的提示词模板封装
    vector_store.py        # 向量检索接口占位（可接 Pinecone/Faiss/自建服务）
    pbt_generator.py       # 将信息整合并组织 PBT 生成与 pylint 检查
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 中目前列出了一些建议依赖，你可以根据自己的环境/LLM SDK 调整：

- `astroid` / `ast`（标准库）用于代码解析
- `pylint`：静态检查
- `openai` 或其他 LLM SDK
- `hypothesis`：用于 property-based tests（可选）

---

## 使用方式（示例）

命令行（假设已在项目根目录）：

```bash
python -m pbt_gen.cli --project-dir /path/to/your/project \
                      --output-dir ./generated_tests \
                      --functions module_a.foo,module_b.ClassC.method_d \
                      --top-n 10
```

- **`--project-dir`**: 被测项目的根目录。
- **`--output-dir`**: 生成 PBT 测试的输出目录。
- **`--functions`**: 逗号分隔的模块路径列表；若省略则对项目中所有函数/方法执行流程。
- **`--top-n`**: 步骤 3 中 LLM 选出的相关文件数目。

> 目前的代码会提供完整的管线骨架；你需要在 `llm_client.py` 和 `vector_store.py` 中填入真实实现，或对接你已有的 LLM/检索服务。

---

## 核心设计要点

- **模块路径 metadata 一致性**
  - 所有 `FunctionInfo`、检索结果、LLM 提取出的“相关函数/文件信息”都要带上 `module_path`（例如：`pkg.subpkg.module.Class.method`），方便 import。
- **可替换的 LLM 和向量检索**
  - LLM 调用通过 `llm_client.LLMClient` 抽象。
  - 向量检索通过 `vector_store.VectorStore` 抽象。
- **提示词集中管理**
  - 所有 prompt 模板集中在 `prompts.py`，方便调优。

---

## 下一步扩展建议

- 接入真实的 LLM（如 OpenAI / Azure OpenAI / 本地 LLM）。
- 接入真实的向量检索（如 Faiss / Milvus / Pinecone / 自建 Elastic/Weaviate 等）。
- 为生成的 PBT 增加运行器，真正执行测试并统计 property 覆盖情况。
- 在 CLI 中增加配置文件支持（例如 `pbt_gen.toml`），保存/复用工作流参数。


