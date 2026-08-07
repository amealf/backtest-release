# 项目工作规则

<!-- generated-by: manage-project-context; 合并到现有用户文件前必须审阅 -->

这是 Agent 处理 <strong>Backtest V4.41 Research</strong> 时使用的简明工作协议。V4.41 是建立在 V4.4 策略与排序谱系上的小版本。详细项目上下文与维护规则保存在 `project_management` 下。

## 必须阅读

1. 阅读本文件。
2. 阅读 `project_management\00_START_HERE.en.md`，了解证据优先级与完整任务读取矩阵。
3. 按矩阵阅读对应英文文档。英文是执行来源，`*.zh.md` 是中文审阅镜像。

## 按任务读取

- 范围或规划：阅读 `01_goal\PROJECT_GOAL.en.md`、`01_goal\PROJECT_CONSTRAINTS.en.md`、`03_active_work\CURRENT_TASKS.en.md` 与 `03_active_work\WORK_PROGRESS.en.md`。
- 设计、运行、解释参数探索或交付下一轮参数：还要阅读并遵守 `05_domains\research\PARAMETER_EXPLORATION_GUIDE.en.md`、`05_domains\research\RESEARCH_CONSTRAINTS.en.md` 与 `03_active_work\EXPERIMENT_PROGRESS.en.md`。
- 运行任何回测：还要阅读并更新 `03_active_work\BACKTEST_MANAGEMENT.en.md`。
- 跨品种迁移或对新品种开展全新搜索：还要阅读 `00_core\STRATEGY_CONTRACT_V4_4.en.md`、`10_instruments\INSTRUMENT_PROFILE_CONTRACT.en.md` 与 `20_campaigns\CAMPAIGN_WORKFLOW.en.md`。
- 交付物或当前行为修改：另外阅读 `02_current_state\SOURCE_OF_TRUTH.en.md`、`02_current_state\CURRENT_VERSION.en.md` 与 `04_decisions\DECISIONS.en.md` 中相关记录。
- 行为修复，或变化理由会影响后续工作：另外阅读 `04_decisions\CHANGE_REASONS.en.md`。
- 代码入口、执行流程、组件职责、依赖、数据流或接口边界修改：另外阅读 `02_current_state\CODE_ARCHITECTURE.en.md` 与 `03_active_work\WORK_PROGRESS.en.md`。
- 新方向：阅读 `07_future\NEXT_DIRECTIONS.en.md`，以及当前目标、约束、任务与工作进展。
- 历史或非活动工作：只有任务需要过去证据、复用或恢复时，才阅读 `06_history\PROJECT_HISTORY.en.md` 与相关 `99_archive` 说明。
- 参考材料：使用前在 `08_references\README.en.md` 中确认来源。

本节路径均相对于 `project_management`。

## 聊天驱动的项目记忆

- 开始修改项目时，阅读当前任务内可用的聊天与委派上下文。不能声称读取了不可见聊天。
- 将用户最新确认内容与本文件、`00_START_HERE.en.md` 及任务对应英文文档进行比较。较新的明确更正覆盖较旧摘要。
- 项目工作确认了长期变化后，在任务结束前更新对应英文文档与中文镜像；用户无需另外提出维护文档。
- 已完成或发生重要变化的工作追加到 `03_active_work\WORK_PROGRESS.en.md` 及其中文镜像。主交付版本或工作方法变化时，存档已经失效的详细进展，并保留简洁恢复指针。
- 解释型问题、假设讨论或未确认想法不触发文件修改。
- 只记录简明事实、决定与有用理由，不保存聊天原文、秘密信息或私人个人数据。

## 已启用领域模块

- 修改数据结构、迁移、管线、血缘或数据契约前，阅读 `project_management\05_domains\data\DATA_REQUIREMENTS.en.md`。
- 修改研究方法、运行实验或解释结果前，阅读 `project_management\05_domains\research\RESEARCH_CONSTRAINTS.en.md`。
- 每次参数探索都必须阅读并遵守 `project_management\05_domains\research\PARAMETER_EXPLORATION_GUIDE.en.md`。

## 执行编排

- 本项目按品种与评价区间组织回测。任何已就绪的品种配置都可以声明独立评价区间并运行；K200 是保留的证据谱系之一，不再承担全项目存储身份。
- 每个已完成评价保存到 `results\evaluation_packages\<instrument_id>\<start_YYYYMMDDTHHMMSS>__<end_YYYYMMDDTHHMMSS>`。目录名只包含品种和实际评价区间。训练、测试、迁移、留出或描述性用途写入 `EXPERIMENT.md`、比较方案和双语管理记录。
- 每个日期结果包包含 `evaluation_manifest.json`、`parameter_summary.csv`、按候选集生成的浏览器精简数据、不可变逐笔记录、实验记录和逐笔入口。后续计算以不可变批次追加，只生成缺少的逐坐标区块。
- `results\evaluation_comparison` 是通用多区间比较入口。比较方案可以选择任意已完成日期结果包，按精确 `combo_id` 连接，并保持每个品种的排名谱系独立。比较可以用于迁移、时间验证或其他实验，无需修改结果包名称。
- 现有稳定入口与已完成总入口／逐笔 HTML 属于兼容契约。新读取链路要在平行目录生成，完成逐行数值一致性与浏览器可见内容一致性后，只切换稳定跳转。旧入口和已完成页面继续保留，可立即恢复。
- 每个行情数据下载目录都必须包含 `README.md`。目录创建时生成，恢复下载或完成下载时刷新，并记录品种、来源、请求与更新时间、合约切割、主力合约选择规则、合并规则、血缘和主要文件。
- 每次启动回测前，将品种、实际读取的数据文件、回测开始时间与回测结束时间追加到 `project_management\03_active_work\BACKTEST_MANAGEMENT.en.md` 及其中文镜像。回测中途停止时仍保留记录。
- 回测运算与证据分析和 HTML 发布分离。中间探索轮只闭合不可变原始结果与小型汇总，不生成累计或逐笔 HTML。
- 参数探索采用 AI 主导的循环：对互不相邻的合法区域开展多轮跳跃搜索，选择互不相邻的优秀锚点并做有限单参数网格，随后回到跳跃搜索。单轮可以只属于其中一个阶段；完整循环同时保留探索与细化证据。
- 单参数网格不设固定点数上限。运算前冻结有限边界、步长安排、锚点和预计坐标数；只依据闭合后的完整网格趋势决定继续、细化、扩大或停止。后续网格必须通过 completed／active／pending 精确反连接。
- 用户在循环之间查看摘要，并可修正目标、锚点、范围或方向。用户暂时不在线时，AI 只能在已经授权的时间、品种、方法、数据和资源边界内连续完成多个轮次；方法变化或授权扩展仍需用户确认。
- 整组探索结束后，或用户明确要求中途发布时，才生成一次共享累计总入口 `results\all_completed_union_analysis\index.html` 与共享逐笔入口 `results\all_completed_union_analysis\trade_review\index.html`。最终发布包含全部兼容已完成轮次。

## 工作规则

- 用户只要求回答问题时不修改文件；用户要求修改后再变更项目内容。
- 采用满足已确认契约的最小实现。只有存在具体、严重且现实的故障风险时才增加防御分支，不添加推测性的多层兜底。
- 核验保持最小且与风险相称。小改动不运行大范围回归、重复哈希审计或浏览器矩阵。工作存在可视结果且环境允许时，应向用户提供一张有代表性的截图，并判断是否存在重叠、乱码、不对称或明显布局错误。常规浏览器自动化无法打开本地 `file:///` 页面时，使用 Computer Use 控制 Chrome。
- 用户指定现有工具、模板或行为时，必须检查并使用该指定项目。若要改用新写实现或扩大需求范围，必须取得用户确认。
- 项目当前真实文件、行为与产物是执行证据；管理文档用于解释意图、状态与有效边界。
- 未知事实保持「待确认」。不能把提案、后续方向或存档材料描述成当前事实。
- 保留项目已有文件。任何 `*.project-management.proposed*` 文件都需要审阅后才能合并。
- 任一管理文档变化时，保持英文与中文内容一致。
- 新增或实质修改面向人的中文项目文档或中文界面文案时，在事实与技术含义稳定后调用 `qu-ai-wei` 技能打磨。代码标识符、公式、参数、数值、路径、命令、哈希及中英文含义必须保持不变。
- 管理 Markdown 的加粗只使用 HTML `<strong>...</strong>`，不使用 Markdown `**...**`。inline code、围栏代码块，以及 `batches/**/trades.csv` 这类 glob 路径中的星号保持原样。

## 验证分级

- 只有版本升级，以及回测引擎、开平仓或成交逻辑、收益／成本计算、数据准备、schema、执行契约或结果语义发生变化时，才运行完整回归测试。
- 前端筛选、排序、选择器、按钮、导航或状态同步等行为变化，采用小范围针对性测试或浏览器交互检查。只覆盖受影响交互及其最近依赖状态；风险边界扩大时再提高验证级别。
- 纯展示变化，例如文案、颜色、间距、字体、对齐、响应式布局或可见性，只需重新生成对应 HTML 并进行一次简单功能检查；环境允许时附一张有代表性的截图。
- 变更跨越多个级别或语义影响不明确时，采用更高级别。交接时记录所选级别与证据，不能因为习惯而运行更大范围测试。

## 修改后

- 使用 `00_START_HERE.en.md` 的维护触发条件归类聊天中已确认的变化。
- 按 `00_START_HERE.en.md` 的变更类型规则更新对应文档。
- 工作完成或发生重要变化时更新 `WORK_PROGRESS`；行为变化需要长期理由时更新 `CHANGE_REASONS`。
- 管理 Markdown 或 manifest 变化后运行 `node "project_management\tools\build_dashboard.mjs"`。
- 变更影响 Dashboard 布局或交互时进行页面检查。
