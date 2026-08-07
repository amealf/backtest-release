# 当前有效文件

## 有效性边界

本文件记录 **Backtest V4.3 Research** 当前权威路径与产物。初始化检测只提供候选；只有当前行为、配置、测试、生成产物或用户确认支持时，路径才成为权威来源。

## 已确认管理入口

| 作用 | 当前路径 | 状态 |
| --- | --- | --- |
| Agent 协议 | `..\..\AGENTS.md` | 审阅或批准合并后生效 |
| 详细读取地图 | `..\00_START_HERE.en.md` | 当前 |
| 管理 manifest | `..\project-management.json` | 生成控制文件 |
| 离线管理 Dashboard | `..\index.html` | 生成视图，不代表项目领域 |

表格路径均相对于本文件目录。

## 项目权威来源登记

| 来源类型 | 当前路径或标识 | 证据 | 生效时间 | 状态 |
| --- | --- | --- | --- | --- |
| 引擎 | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\code\v4_3_engine.py` | 双策略行为与回归测试 | 2026-08-02 | 当前 |
| 可恢复 runner | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\code\run_v4_3_resumable_campaign.py` | 策略绑定 runner 测试与 manifest | 2026-08-02 | 当前 |
| 阶段分析器 | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\code\analyze_v4_3_scenario_3_stage.py` | 当前交付调用路径 | 2026-08-01 | 当前 |
| 逐笔 HTML 生成器 | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\code\build_v4_3_review_delivery.py` | 固定模板哈希门禁 | 2026-08-01 | 当前 |
| 累计总入口生成器 | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\code\build_v4_3_combined_union_analysis.py` | 当前累计交付调用路径 | 2026-08-01 | 当前 |
| 身份 manifest | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\SOURCE_MANIFEST.json` | 哈希绑定双策略实现与身份映射 | 2026-08-02 | 当前 |
| 运行输入 manifest | `..\..\runtime_inputs\RUNTIME_INPUTS.json` | 相对路径、大小与哈希 | 2026-08-02 | 当前 |
| 准备 manifest | `..\..\runtime_inputs\data_preparation\data_preparation_manifest.json` | Schema v4、中性标记与受支持／默认策略 | 2026-08-02 | 当前 |
| 情景定义 | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_scenario_groups_single_select_combined_exit_20260801.json` | 当前情景 schema | 2026-08-01 | 当前 |
| Round 1 可执行计划 | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_max_w_multiround_20260802_round_01_all_window.json` | 最终冻结的 12 坐标计划；SHA-256 为 `36f9f47fd1c49ac1be68abf7a0f146b81d667c115123b385dde31239daccdc80`；绑定来源 manifest SHA-256 `78fcea5452769fe14f531c4e3036fe7ba3b2513092a0ad1e43a5b7d495c949ec` | 2026-08-02 | 已冻结并执行的 smoke 计划 |
| Round 2 可执行计划 | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_max_w_multiround_20260802_round_02_all_window.json` | 已冻结的 504 坐标笛卡尔计划；SHA-256 为 `5735c12e439c8686f1f8e2e5e5d4ffcdb956902aa21825619f5c6d9ab4eebe85`；绑定来源 manifest SHA-256 `306bf83ca50dab913fc7d8f081ec249dc05e03529ced7ce1c6dc58addab3f946`；已审计 fingerprint `307d2d8adbec3d7f0915b19ce51db65edd58d86111d491b21a91a7e060ed8095` | 2026-08-02 | 已冻结并执行的计划；原始证据完成不可变闭合 |
| Round 3 可执行计划 | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_max_w_multiround_20260802_round_03_all_window.json` | 已冻结的 72 坐标交互细化计划；SHA-256 为 `78f695c76e508e0c7210f60ea01717b03c677581f589c3b69bb277f187e7c47b`；三个彼此独立的 24 点块；fingerprint `eb61c27632d5b0efad439e5b7580c6d813374d354fead34acc695b058e9111f2` | 2026-08-02 | 已冻结并执行的终止轮计划 |
| Round 4 可执行计划 | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_max_w_multiround_20260802_round_04_all_window.json` | 已冻结的 72 坐标边界计划；SHA-256 `f991e8c46b42ccc38677eef3be7f2e67d9e822592bb53d3578a4fbc1cdb77670`；fingerprint `ebd5f4213920c77afc52b03287b396a4002ea86750f31a5fc8fc18752b16deb5` | 2026-08-02 | 已冻结并执行；不可变原始与交付已闭合 |
| Round 5 可执行计划 | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\plans\v4_3_max_w_multiround_20260802_round_05_all_window.json` | 已冻结的终局 24 坐标计划；SHA-256 `bb7f323e077ebb919d122764d7fac20b97b0790959d71761b77bcf69e6a23ed8`；两个彼此独立的 12 点块；无笔均收益块；fingerprint `e639128e241012085aa799a3b2c22087ba5ca884acb88571ece4ff89aa084621` | 2026-08-02 | 已冻结并执行；终局不可变原始与交付已闭合 |
| 跨版本场景管理器 | `D:\Code\backtest-release\shared_tools\scenario_manager\index.html` | 基于用户指定市场选定器建立的共享入口 | 2026-08-02 | 当前 |
| 测试 | `..\..\research_variants\short_momentum_net_drop_rebound_v4_3\code\test_v4_3_engine.py` 与同包测试 | 当前验证套件 | 2026-08-01 | 当前 |
| 终止轮阶段结果 | `..\..\results\campaigns\v4_3_max_w_multiround_20260802\round_05_all_window` | 原始 `IMMUTABLE_CLOSED`，24 个坐标、46,732 笔交易；completion SHA-256 `5804d8940a1ab5c582eee96872b7cef133886bfe53110a6d5846cd460f78ff65`；固定模板交付已闭合 | 2026-08-02 | 当前终局样本内阶段证据 |
| 终局解释 | `..\..\.omo\teams\team-a31b9876\artifacts\A_v4_3_max_w_round_05_20260802.md` | 四视图终局 why／run／observed／next 记录；无 Round 6 或参数接受 | 2026-08-02 | 当前有效，直到复制到干净 package staging |
| 终局只读审计 | `..\..\.omo\teams\team-a31b9876\artifacts\B_terminal_read_only_audit_20260802_v1.json` | 五份计划／completion／fingerprint、684 个唯一 ID、232,225 笔交易、57 个 batch manifest、285 个原始制品、775 个交付制品，以及零活动锁／进程 | 2026-08-02 | 已通过；SHA-256 `fb06842668c478bda18ac65552d3a19c97a416458cee804bdba01aa0f26f73bb` |
| 稳定累计总入口 | `..\..\results\all_completed_union_analysis\index.html` | 快照 `4bed7f828a7068ee0ee70001a245133c481e785358cfb3f2b043d08713ca7259` 的稳定 wrapper；684 个坐标／232,225 笔交易／五个阶段；SHA-256 `bf97c850bb3f27c29c44525b96e90b6095f2c7b4958fe6dfe6cd376d2fd0ac56` | 2026-08-02 | 当前稳定累计入口 |
| 稳定累计逐笔入口 | `..\..\results\all_completed_union_analysis\trade_review\index.html` | 同一快照的稳定固定模板逐笔 wrapper；SHA-256 `6ff458f359b319db97cd137b1080a26922df69f23a61e361027bfa7681b991d9` | 2026-08-02 | 当前稳定累计逐笔入口 |

用户要求使用、查看、新建或删除场景时，到上述跨版本场景管理 HTML 查询。历史结果快照保持不可变证据，不承担可编辑场景权威职责。

项目不使用的类别不需要填写。

## 初始化文件概览

- `.gitignore`
- `.python-version`
- `package-lock.json`
- `package.json`
- `PRODUCT.md`
- `README.md`
- `requirements-v4_3.txt`
- `research_variants\short_momentum_net_drop_rebound_v4_3\__init__.py`
- `research_variants\short_momentum_net_drop_rebound_v4_3\README.md`
- `research_variants\short_momentum_net_drop_rebound_v4_3\SOURCE_MANIFEST.json`
- `runtime_inputs\data_preparation\baseline_filter_atoms.csv`
- `runtime_inputs\data_preparation\baseline_filter_events.json`
- `runtime_inputs\data_preparation\data_preparation_manifest.json`
- `runtime_inputs\market_data\data_preparation_audit.json`
- `runtime_inputs\market_data\k200_clean_15s_session_filled.csv`
- `runtime_inputs\provenance\V4_2_SOURCE_MANIFEST.json`
- `runtime_inputs\templates\historical_v4_main.html`
- `runtime_inputs\templates\historical_v4_trade.html`
- `runtime_inputs\templates\market-intuition-selector.html`
- `runtime_inputs\templates\plotly.min.js`
- `RUNTIME.md`

该有限列表只帮助定位，不能据此声明权威来源。

## 已退役或无效材料

V4.2 源码、计划和结果属于 V4.3 运行边界之外的历史证据。`runtime_inputs\provenance` 内的副本只作记录，不能作为活跃运行依赖。

## 维护规则

权威入口、分支、配置、命令、结果、报告、发布或产物变化后更新。链接当前路径，不复制大型内容。
