# 项目管理入口

## 用途

这个目录是 <strong>Backtest V4.41 Research</strong> 的共享工作记录。V4.41 是建立在 V4.4 策略与排序谱系上的小版本。它保存当前意图、约束、有效文件、版本状态、当前工作、决策、历史、后续方向、参考材料与存档边界，不复制大型项目文件或原始来源材料。

该目录于 2026-08-01 为一个<strong>现有项目</strong>建立。

## 当前概览

- <strong>长期目标：</strong>修复成交真实性，并让 V4.4 使用仓库本地运行输入实现可复现。
- <strong>可选领域模块：</strong>data、research
- <strong>执行语言：</strong>英文。
- <strong>审阅镜像：</strong>中文。
- <strong>未知事实：</strong>在获得用户确认或项目证据前保持「待确认」。

管理 Dashboard 是离线文档查看器。它使用 HTML，不能据此认定项目包含或需要前端。

## 证据优先级

使用适用范围内优先级最高的来源：

1. 用户当前指令与已确认决定。
2. 项目当前真实材料与行为；配置、测试或命令只在实际存在时适用。
3. 当前英文项目管理文档。
4. 中文审阅镜像。
5. 历史、提案、后续方向与存档材料。

来源发生冲突时记录差异，不能默默选择更方便的来源。

## 聊天内容摄取与持续维护

初始化或修改项目前，阅读当前任务内可用的聊天与委派上下文，提取已确认目标、约束、决定、任务、当前状态、领域、更正与未知项。不能声称读取了不可用聊天。

新项目使用这些已确认事实填充中立管理框架。只有用户要求建立项目实现，并且可用聊天已明确项目形态时，才建立项目实现框架。已有项目需要把聊天内容与真实项目证据进行核对，并保护每个用户已有文件。

后续工作中，聊天确认了长期变化后，在任务结束前更新对应文档。解释型问题、假设讨论或未确认想法不修改文件。只保存结论与有用理由，不保存聊天原文、秘密信息或私人个人数据。

需求指定现有工具、模板或行为时，该指定项目就是实现边界。工作前必须检查它；若要改用新写实现或增加需求之外的行为，必须取得用户确认。

任何文件、行为、输出、工作流、字段或材料的改动，只要超出用户明确提出或已经同意的范围，都必须停止并取得用户许可。技术便利、来源追溯、相邻清理与 agent 自己的理解都不会扩大授权范围。

## 角色中立执行治理

- 项目允许任何已就绪品种配置使用任意精确评价区间。已完成结果按日期区间保存为结果包，实验角色写在方案和记录中，不进入目录名。
- `results\evaluation_comparison` 读取各结果包自己的浏览器摘要，并按精确 `combo_id` 连接。框架调整期间，现有稳定总入口和逐笔页面继续作为兼容契约。
- 中间探索轮只闭合不可变原始证据与小型汇总，默认不生成累计或逐笔 HTML。
- 参数探索采用 AI 主导的循环：多轮跳跃搜索、围绕互不相邻优秀锚点的有限单参数网格、再次跳跃搜索。单轮可以只属于其中一个阶段；报告标明阶段，并在完整循环中分开保留探索与细化证据。
- 单轴网格不设固定点数上限。运算前冻结有限边界、步长、锚点和预计坐标数；完整闭合结果得到解释后，才决定继续、细化、扩大、停止或更换方向。
- 用户在循环之间或指定检查点查看结果并修正。用户暂时不在线时，自主轮次不得超出已经授权的时间、品种、方法、数据和资源边界。
- 整组探索结束后生成一次共享累计总入口与共享逐笔入口。中途发布需要用户明确要求；最终发布包含全部兼容已完成轮次，并只允许一个累计 writer。

## 验证级别路由

- <strong>完整回归：</strong>版本升级，以及引擎行为、开平仓或成交、收益／成本计算、数据准备、schema、执行契约或结果语义发生变化。
- <strong>针对性交互：</strong>前端筛选、排序、选择器、按钮、导航和状态同步。检查受影响交互及其最近依赖状态。
- <strong>仅视觉检查：</strong>文案、颜色、间距、字体、对齐、响应式布局和可见性。重新生成对应 HTML，完成简单功能检查，并保留桌面／手机截图；只有同时影响行为或数据时才运行完整测试。
- 变更跨越多个级别或语义影响不明确时，采用更高级别，并在交接中记录所选级别。

## 默认读取顺序

修改项目前阅读：

1. `..\AGENTS.md`
2. `00_START_HERE.en.md`
3. `01_goal\PROJECT_GOAL.en.md`
4. `01_goal\PROJECT_CONSTRAINTS.en.md`
5. `02_current_state\SOURCE_OF_TRUTH.en.md`
6. `02_current_state\CURRENT_VERSION.en.md`
7. `03_active_work\CURRENT_TASKS.en.md`
8. `03_active_work\WORK_PROGRESS.en.md`
9. 任务涉及运行、解释或闭合实验时，阅读 `03_active_work\EXPERIMENT_PROGRESS.en.md`。
10. 每次运行回测前，阅读 `03_active_work\BACKTEST_MANAGEMENT.en.md`。
11. 每次设计、运行、解释参数探索或交付下一轮 CSV 时，阅读 `05_domains\research\PARAMETER_EXPLORATION_GUIDE.en.md`。
12. 跨品种迁移或对新品种开展全新搜索时，阅读 `00_core\STRATEGY_CONTRACT_V4_4.en.md`、`10_instruments\INSTRUMENT_PROFILE_CONTRACT.en.md` 和 `20_campaigns\CAMPAIGN_WORKFLOW.en.md`。

其他文档只在任务需要时读取。

## 任务选择矩阵

| 任务 | 追加读取文档 |
| --- | --- |
| 解释或回答 | 只读取回答所需文件；用户未要求修改时不改变项目状态。 |
| 修改范围或规划 | 目标、约束、当前任务与工作进展。 |
| 修改交付物或当前行为 | 目标、约束、当前有效文件、当前版本、当前任务、工作进展与相关决策。 |
| 修复行为或对行为作重要修改 | 当前版本、工作进展、变更理由与相关决策。 |
| 修改代码结构、执行流程、组件、依赖、数据流或接口边界 | 当前有效文件、当前版本、代码架构、当前任务、工作进展与相关决策。 |
| 修改已确认领域 | 下方对应模块，以及当前状态、工作进展与相关决策。 |
| 设计、运行、解释参数探索或交付下一轮参数 | 参数探索指南、研究约束、实验进度、当前约束、当前任务与当前源码身份。 |
| 运行任何回测 | 回测管理、当前源码身份、品种配置、实验活动流程，以及该运行模式要求的其他文档。 |
| 迁移到其他品种或对新品种开展全新搜索 | 策略协议、品种配置协议、实验活动流程、参数探索指南、数据要求和当前源码身份。 |
| 提议方向 | 当前目标、约束、任务、工作进展、决策与后续方向。 |
| 使用参考材料 | 参考索引及其源文件。 |
| 检查、复用或恢复历史工作 | 项目历史、存档说明与对应项目说明。 |

### 已启用领域读取规则

- 修改数据结构、迁移、管线、血缘或数据契约前，阅读 `project_management\05_domains\data\DATA_REQUIREMENTS.en.md`。
- 修改研究方法、运行实验或解释结果前，阅读 `project_management\05_domains\research\RESEARCH_CONSTRAINTS.en.md`。
- 每次参数探索都必须阅读并遵守 `project_management\05_domains\research\PARAMETER_EXPLORATION_GUIDE.en.md`。

## 文档维护触发条件

| 文档 | 更新条件 |
| --- | --- |
| `PROJECT_GOAL` | 用户改变长期目标或已确认成功标准。 |
| `PROJECT_CONSTRAINTS` | 长期通用约束被确认、修改或退役。 |
| `SOURCE_OF_TRUTH` | 权威路径、入口、分支、配置或产物变化。 |
| `CURRENT_VERSION` | 交付物、当前行为、项目结构、接口、依赖或运行流程变化。 |
| `CODE_ARCHITECTURE` | 代码入口、组件职责、依赖或调用关系、数据流、生成产物或接口边界变化。项目没有代码时标为「不适用」。 |
| `CURRENT_TASKS` | 任务被声明、完成、阻塞、恢复或改变范围。 |
| `WORK_PROGRESS` | 工作完成，或变化程度足以影响交接、审阅或当前项目理解。 |
| `EXPERIMENT_PROGRESS` | 已授权实验轮次闭合，或实验有效边界发生变化。 |
| `BACKTEST_MANAGEMENT` | 每次启动回测前更新；回测中途停止时仍保留该行。 |
| `PARAMETER_EXPLORATION_GUIDE` | 已确认的交易诊断、多指标改善判断、实验块或下一轮 CSV 交付规则发生变化。 |
| `STRATEGY_CONTRACT_V4_4` | 已接受的品种无关开仓、平仓、时序或状态机语义发生变化。 |
| `INSTRUMENT_PROFILE_CONTRACT` | 数据、成本、gap、低活跃、情景或排名谱系接口发生变化。 |
| `CAMPAIGN_WORKFLOW` | 完整迁移、目标局部细化、同谱系续探、全新搜索、跨品种排名或指令交接规则发生变化。 |
| `CHANGE_REASONS` | 重要行为变化需要长期保留修改原因、原有表现、新表现、证据影响与有效边界。 |
| `DECISIONS` | 长期选择被确认、退役或反转。 |
| `PROJECT_HISTORY` | 事件改变历史或当前工作的解释。 |
| `NEXT_DIRECTIONS` | 可检验的研究问题、策略假设或验证路径出现、变化、转为当前工作或退役。执行、交付和资源门槛写入约束、任务或工作规则，不归入后续研究方向。 |
| `08_references\README` | 保留材料或其来源、相关性、审阅状态变化。 |
| `99_archive\README` | 材料转为非活动、存档规则变化或项目恢复。 |
| 已启用领域文档 | 对应领域的长期规则变化。 |
| 根目录 Agent 文件与本入口 | 读取顺序或维护协议变化。 |

英文来源与中文镜像需要同步更新。使用下方命令重新生成 `index.html`：

```powershell
node "project_management\tools\build_dashboard.mjs"
```

## 历史与存档边界

Dashboard 将当前规则与工作、项目历史及非活动材料分开。日常工作不加载历史或存档文档；任务需要过去证据、复用或恢复时再读取。

主交付版本或工作方法变化，并且旧进展细节会误导当前工作时，将被替代的正文移入 `99_archive` 下带日期的目录，在 `WORK_PROGRESS` 中保留简洁恢复指针，并完整保留存档材料。

## 初始化文件概览

初始化器检测到以下已有文件；该列表只保留少量定位信息：

- `.gitignore`
- `.python-version`
- `package-lock.json`
- `package.json`
- `PRODUCT.md`
- `README.md`
- `RELEASE.json`
- `requirements-v4_4.txt`
- `research_variants\short_momentum_net_drop_rebound_v4_4\__init__.py`
- `research_variants\short_momentum_net_drop_rebound_v4_4\README.md`
- `research_variants\short_momentum_net_drop_rebound_v4_4\SOURCE_MANIFEST.json`
- `runtime_inputs\data_preparation\baseline_filter_atoms.csv`
- `runtime_inputs\data_preparation\baseline_filter_events.json`
- `runtime_inputs\data_preparation\data_preparation_manifest.json`
- `runtime_inputs\market_data\data_preparation_audit.json`
- `runtime_inputs\market_data\k200_clean_15s_session_filled.csv`
- `runtime_inputs\provenance\V4_2_SOURCE_MANIFEST.json`
- `runtime_inputs\templates\historical_v4_main.html`
- `runtime_inputs\templates\historical_v4_trade.html`
- `runtime_inputs\templates\market-intuition-selector.html`
- `runtime_inputs\scenarios\market_catalog.json`
- `runtime_inputs\scenarios\scenario_catalog.json`
- `tools\build_v4_41_scenario_manager.py`
- `tools\apply_v4_41_scenario.py`
- `runtime_inputs\templates\plotly.min.js`
- `RUNTIME.md`

检测结果不是权威结论。请在 `02_current_state\SOURCE_OF_TRUTH.en.md` 中确认当前入口与产物。

## 审阅方式

- 文档保持简洁并使用事实描述。
- 区分当前、已退役、提案与未知状态。
- 链接真实路径，不复制大型输出。
- 记录事实依据与失效边界。
- 加粗只使用 HTML `<strong>...</strong>`，不写 Markdown `**...**`，避免中文标点附近的 CommonMark 分隔符歧义。inline code、围栏代码块和 glob 路径中的星号保持原样。
