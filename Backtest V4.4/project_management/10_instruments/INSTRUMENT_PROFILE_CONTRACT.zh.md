# 品种配置协议

## 必填配置

每个可执行品种配置都要绑定：

- `instrument_id`、显示名称、时区和共享策略协议 ID；
- 行情数据与准备清单的路径、大小和 SHA-256，以及明确的 `bar_seconds`；
- 冻结的成本模型参考；
- 受支持的 gap policy 模式，以及绑定哈希的配置与实现身份；
- 受支持的 low-activity policy，以及绑定哈希的配置、实现身份和因果 baseline availability 规则；
- 可选 scenario set；
- 与样本对应的 `ranking_lineage_id`。

成本或 gap 决策缺失时，配置保持 `requires_user_input`，执行必须停止并询问用户。SImain 与 NQ 模板不会虚构手续费、滑点、汇率或 gap 规则。品种配置、数据准备清单与实际行情网格都必须声明策略使用的 15 秒颗粒度；任一值不同都要停止执行，并报告期望值与实际值。

## 成本模型

名义价值按 `参考价格 × point value` 计算。手续费币种与报价币种不同时，使用冻结的汇率参考转换。往返成本为：

`slippage bps + 10,000 × 报价币种往返手续费 / 合约名义价值`。

通用输出字段为 `point_value`、`quote_currency`、`commission`、`slippage_bps`、`contract_notional_quote` 和 `round_trip_total_cost_bps`。K200 旧字段继续保留，供历史测试与页面使用。

现有 K200 结果保留原成本参考。后续 K200 阶段可以绑定按新参考价计算的成本。每个坐标保留来源阶段的成本模型，并披露实际 `round_trip_cost_bps`；它仍可进入同一个 K200 排名谱系。

## 情景与低活跃规则

`scenario_set` 可以为空。schema-v5+ 的 `profile_optional` 遇到没有 scenario set 的配置时，情景总体为空，不能回退到 K200 事件。只有明确声明的 legacy policy 可以加载旧 K200 事件。K200 事件区间保留在 K200 配置中，只作为独立过滤或诊断视图。迁移或全新搜索的主结果不得依赖 K200 情景资格。

gap 与低活跃 policy 标签属于可执行身份，不能使用自由文本冒充规则。计算以前必须确认 profile policy ID、配置哈希、实现哈希与数据准备清单声明完全一致；这些身份进入计划指纹、阶段协议、完成证据和结果语义边界。共享低活跃生命周期属于通用层；具体阈值与启用／停用决定只来自绑定 policy。

## 当前文件

- 已就绪 K200 配置：`research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\k200m.json`。
- SImain 输入模板：`research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\simain.template.json`。
- NQ 输入模板：`research_variants\short_momentum_net_drop_rebound_v4_4\instrument_profiles\nq.template.json`。
