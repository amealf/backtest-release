# 数据要求

## 品种配置绑定

- 每个新品种都在品种配置中绑定行情数据和准备清单的路径、大小、SHA-256 与 `bar_seconds`。campaign 计划读取这些绑定；代码不选择特定品种文件名。
- 配置记录时区、交易所时段、连续合约规则、预热边界、gap policy、low-activity policy 和可选 scenario set。
- 数据、gap 或低活跃决策缺失时，配置不可执行。其他品种配置明确完成以前，K200 是唯一已就绪配置。
- 主回测品种 K200 使用 15 秒 bar，因此策略权威颗粒度为 15 秒。所有迁移目标也必须使用 15 秒 bar。品种配置、数据准备清单与实际时间戳网格必须一致；任一处不一致都在计算以前停止并报告用户。
- gap 与低活跃配置／实现文件需要绑定哈希。它们的 policy ID 和哈希必须与数据准备清单一致；只有 `status=ready` 不足以执行。

## 跨品种评价数据

- SImain 对比数据使用 America/Chicago 时区下明确 SIH6 的 15 秒补齐交易时段 OHLC。当前测试区间为 2026-01-29 至 2026-02-23；数据从 2026-01-28 开始，为滚动指标提供预热。
- 预热 bar 可以参与滚动指标，但逐笔总体只包含开仓时间位于声明测试区间内的持仓。
- 必须记录交易所时段、源时区、连续合约日程、换月次数与调整政策、真实 gap、synthetic bar、零成交暴露，以及是否存在专用品种低活动政策。
- 跨品种使用百分比和 bps 比较；同时保留原始价格点数的 MFE、MAE 与已实现点数合计，供单品种检查。

## 当前来源

- `runtime_inputs/market_data` 下的 K200 补齐交易时段 15 秒 OHLC：233,368 行，范围为 `2026-05-23T00:00:00+09:00` 至 `2026-08-07T03:21:45+09:00`，SHA-256 为 `9760d367a109777c4789ce45d982a6c0708bacddad8f549450ed94f81ad5c405`。
- `runtime_inputs/data_preparation` 下的 baseline 原子、事件与 schema 5 清单。
- 当前已就绪绑定为 `research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\k200m.json`；SImain 与 NQ 模板只用于收集输入，不是活动来源。

## 回测结果包契约

- 逻辑根目录为 `results\evaluation_packages`，实体字节继续通过现有 results 目录联接写入 F 盘。结果包身份为 `<instrument_id>\<start_YYYYMMDDTHHMMSS>__<end_YYYYMMDDTHHMMSS>`，时间只取实际评价区间，不包含预热或文件内尚未使用的行情。
- `evaluation_manifest.json` 记录品种、显示名称、精确区间、时区、bar 周期、状态、产物路径、存储方式、来源和实验记录路径。清单不把结果包固定为训练、测试或迁移角色。
- `parameter_summary.csv` 使用统一指标名称保存该结果包的完整已完成参数总体。`browser_summaries\<candidate_set_id>.js` 是比较页面使用的小型候选集数据，不能替代权威参数汇总或逐笔记录。
- `trade_records\trades.csv` 保存不可变逐笔证据。历史适配包可以使用同卷硬链接，清单沿用来源文件已经声明的哈希。以后生成的原生结果以不可变批次追加，闭合后发布一份结果包汇总。
- `trade_review\index.html` 是结果包自己的逐笔入口。兼容结果包可以在保留 query 和 hash 的情况下跳转到历史页面；原生结果包在本目录保存 process payload、目录和确定性逐坐标区块。
- `EXPERIMENT.md`、比较方案和双语项目记录说明本次回测完成了什么，以及该结果包在实验中的用途。后续实验改变用途时，日期目录保持不变。
- 候选集合的 `combo_id` 必须精确、唯一，而且每个已选结果包都覆盖声明总体，比较才能标记为完成。比较页面只加载指标摘要，不加载完整逐笔记录。
- 新完成的评价使用 `runtime_inputs\templates\EVALUATION_PACKAGE_SPEC.template.json` 声明来源文件、统一参数字段、统一指标映射、区间身份和当前实验说明。`tools\register_v4_4_evaluation_package.py` 会拒绝替换已经存在清单的结果包。

## 每次下载的 README 契约

- 每个品种的每个下载目录都创建 `README.md`。恢复下载或完成下载时刷新同一文件；说明不能只保存在项目记忆或独立报告中。
- 记录品种、上游来源和数据类型、来源／请求时区、右端不包含的请求区间、可用时的实际首末观察值、创建／恢复／完成时间、状态，以及主要原始／衍生／审计文件。
- 写明候选合约的比较方式、主力合约判定、每段有效合约区间及其边界是否包含，并说明是否复权。补充批次沿用较早审计结论、没有重新比较候选合约时，需要明确注明。
- 写明血缘和合并契约：补充批次从父批次的右端不包含时间开始；原始下载目录相互独立；衍生行在字段一致的前提下按时间追加；重叠、重复与倒序均禁止；真实 gap 保留；未复权序列保留换月价差。
- README 保留按时间排列的更新表，使后续数据管理可以从所有保留下载目录还原完整获取链路。

## Baseline 可用时间契约

- 正常原子：该 bar 完结时可用。
- 待确认低成交量原子：连续段尚未确认时立即可用，也不影响开仓。
- 待确认区间后来恢复：不产生排除记录。
- 已确认区间：从连续段第一根低成交量原子起，每根原子的 `baseline_excluded_from` 都记录确认时刻；确认前可用，确认时及以后不可用。
- `confirmed_low_activity_active` 从确认原子持续到最后一根低成交量原子；第一根正常成交量原子已经离开门禁。
- `all_window`：同一连续段内所有有限 TR15 原子仍可用。
- `exclude_marked`：计算时间 t 只能使用满足 `baseline_available_from <= t` 的原子。
- `confirmed_low_activity_gate`：计算时间 t 只能使用 `baseline_excluded_from` 为空或晚于 t 的原子；门禁生效时停止新开仓，并取消未成交开仓委托。

重复时间戳默认报错。历史绝对路径只用于来源记录，不是当前运行依赖。

## ZIP 交接记录

- 用户要求 ZIP 时，纳入哈希绑定的 15 秒 OHLC 输入，以及每个已完成 V4.4 阶段的逐笔交易记录。
- 每份不可变原始 `batches/**/trades.csv` 与每个衍生 `analysis/stage_trades.csv` 都复制到 `trade_records/`，并保持文件字节不变。
- `trade_records/TRADE_RECORDS_MANIFEST.json` 必须记录来源 campaign／阶段、记录角色、相对来源路径、行数、大小、SHA-256，以及来源 completion／stage manifest 哈希。
- 其他结果载荷继续留在 ZIP 外：campaign 文件、batch manifest、summary、grid、chunks、snapshot、HTML 输出、日志、锁，以及未完成或失败阶段。
