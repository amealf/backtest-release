from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg', force=True)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import long_momentum_ARCH_shock_multi as strategy_mod


WORKDIR = Path(r"D:\Code\backtest-release\Backtest v2 ratio")
REPORT_ROOT = WORKDIR / "result" / "xagusd_30s_all long shock multi outcome" / "research"
TRAIN_RANGE = ("20250601", "20250610")
OOS_RANGE = ("20250611", "20250615")
FULL_RANGE = ("20250601", "20250615")
GRID_SHOCK_OPEN = [1.8, 2.1, 2.4, 2.7, 3.0, 3.3]
GRID_CLOSE_BAR = [3, 5, 7, 9, 12, 15]
GRID_CLOSE_SPEED = [0.6, 0.9, 1.2, 1.5]
GRID_CLOSE_WD = [1.0, 1.4, 1.8, 2.2, 2.6]

EXPERIMENTS = [
    {
        "config_key": "single_5min",
        "label": "single 5min baseline",
        "signal_mode": "single",
        "periods": ["5min"],
        "agreement_required": 1,
        "min_ready_count": 1,
        "require_base_period": True,
        "score_weights": {},
    },
    {
        "config_key": "vote_2of3_core",
        "label": "5min+30min+1H vote 2of3",
        "signal_mode": "vote",
        "periods": ["5min", "30min", "1H"],
        "agreement_required": 2,
        "min_ready_count": 3,
        "require_base_period": True,
        "score_weights": {},
    },
    {
        "config_key": "vote_3of4_full",
        "label": "5min+15min+30min+1H vote 3of4",
        "signal_mode": "vote",
        "periods": ["5min", "15min", "30min", "1H"],
        "agreement_required": 3,
        "min_ready_count": 4,
        "require_base_period": True,
        "score_weights": {},
    },
    {
        "config_key": "blend_4way_balanced",
        "label": "5min+15min+30min+1H weighted blend",
        "signal_mode": "blend",
        "periods": ["5min", "15min", "30min", "1H"],
        "agreement_required": 2,
        "min_ready_count": 4,
        "require_base_period": True,
        "score_weights": {
            "5min": 0.45,
            "15min": 0.25,
            "30min": 0.20,
            "1H": 0.10,
        },
    },
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_module_for_grid(
        exp: dict,
        start_date: str,
        end_date: str,
        run_label_suffix: str) -> None:
    strategy_mod.data_folder_path = r"D:\Code\data\20260326\\"
    strategy_mod.data_file_name = "xagusd_30s_all"
    strategy_mod.data_selection_mode = "date"
    strategy_mod.start_date = start_date
    strategy_mod.end_date = end_date
    strategy_mod.only_close = False
    strategy_mod.resample_rule = "5min"
    strategy_mod.run_mode = "grid"
    strategy_mod.volatility_method = "garch"
    strategy_mod.shock_open_multiplier_values = list(GRID_SHOCK_OPEN)
    strategy_mod.close_bar_values = list(GRID_CLOSE_BAR)
    strategy_mod.close_speed_sigma_multiplier_values = list(GRID_CLOSE_SPEED)
    strategy_mod.close_wd_sigma_multiplier_values = list(GRID_CLOSE_WD)
    strategy_mod.EXPORT_INTERACTIVE_HTML = False
    strategy_mod.EXPORT_STATS = False
    strategy_mod.MULTI_SHOCK_CONFIG = dict(exp)
    strategy_mod.RUN_LABEL_SUFFIX = run_label_suffix


def configure_module_for_manual(
        exp: dict,
        start_date: str,
        end_date: str,
        row: pd.Series,
        run_label_suffix: str) -> None:
    strategy_mod.data_folder_path = r"D:\Code\data\20260326\\"
    strategy_mod.data_file_name = "xagusd_30s_all"
    strategy_mod.data_selection_mode = "date"
    strategy_mod.start_date = start_date
    strategy_mod.end_date = end_date
    strategy_mod.only_close = False
    strategy_mod.resample_rule = "5min"
    strategy_mod.run_mode = "manual"
    strategy_mod.volatility_method = "garch"
    strategy_mod.shock_open_multiplier = float(row["shock_open_multiplier"])
    strategy_mod.close_bar = int(row["close_bar"])
    strategy_mod.close_speed_sigma_multiplier = float(row["close_speed_sigma_multiplier"])
    strategy_mod.close_wd_sigma_multiplier = float(row["close_wd_sigma_multiplier"])
    strategy_mod.EXPORT_INTERACTIVE_HTML = True
    strategy_mod.EXPORT_STATS = True
    strategy_mod.MULTI_SHOCK_CONFIG = dict(exp)
    strategy_mod.RUN_LABEL_SUFFIX = run_label_suffix


def enrich_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.copy()
    stats["return_abs"] = pd.to_numeric(stats["capital"], errors="coerce") - 100.0
    stats["trade_num"] = pd.to_numeric(stats["trade_num"], errors="coerce")
    stats["biggest_wd"] = pd.to_numeric(stats["biggest_wd"], errors="coerce")
    stats["outcome_high"] = pd.to_numeric(stats["outcome_high"], errors="coerce")
    stats["drawdown_pct_of_high"] = np.where(
        stats["outcome_high"] > 0,
        stats["biggest_wd"] / stats["outcome_high"] * 100.0,
        np.nan,
    )
    stats["quality_score"] = (
        stats["return_abs"] / stats["biggest_wd"].clip(lower=0.35)
    ) * np.log1p(stats["trade_num"].clip(lower=1))
    stats["capital_rank"] = stats["capital"].rank(ascending=False, method="min")
    return stats


def pivot_max(df: pd.DataFrame, row_col: str, col_col: str, value_col: str) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=row_col,
        columns=col_col,
        values=value_col,
        aggfunc="max",
    )
    return pivot.sort_index().sort_index(axis=1)


def draw_heatmap(ax, data: pd.DataFrame, title: str) -> None:
    if data.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "无数据", ha="center", va="center")
        ax.set_axis_off()
        return
    matrix = data.to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([str(col) for col in data.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(idx) for idx in data.index])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if pd.isna(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < np.nanmax(matrix) * 0.7 else "black",
                fontsize=8,
            )
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_grid_overview(df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)

    scatter = axes[0, 0].scatter(
        df["biggest_wd"],
        df["capital"],
        c=df["trade_num"],
        s=np.clip(df["quality_score"].fillna(0).to_numpy() * 8.0, 18.0, 220.0),
        cmap="plasma",
        alpha=0.85,
    )
    axes[0, 0].set_title("capital vs drawdown")
    axes[0, 0].set_xlabel("biggest_wd")
    axes[0, 0].set_ylabel("capital")
    plt.colorbar(scatter, ax=axes[0, 0], label="trade_num")

    draw_heatmap(
        axes[0, 1],
        pivot_max(df, "shock_open_multiplier", "close_bar", "capital"),
        "capital heatmap: shock_open x close_bar",
    )
    draw_heatmap(
        axes[1, 0],
        pivot_max(df, "close_speed_sigma_multiplier", "close_wd_sigma_multiplier", "quality_score"),
        "quality heatmap: close_speed x close_wd",
    )

    top_df = df.sort_values(["quality_score", "capital"], ascending=False).head(15)
    axes[1, 1].barh(
        range(len(top_df)),
        top_df["capital"].to_numpy(dtype=float),
        color="#3b82f6",
        alpha=0.88,
    )
    axes[1, 1].set_title("Top 15 capital")
    axes[1, 1].set_yticks(range(len(top_df)))
    axes[1, 1].set_yticklabels(top_df.index.astype(str).tolist(), fontsize=8)
    axes[1, 1].invert_yaxis()
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_parameter_profile(df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    specs = [
        ("shock_open_multiplier", "shock_open_multiplier"),
        ("close_bar", "close_bar"),
        ("close_speed_sigma_multiplier", "close_speed_sigma_multiplier"),
        ("close_wd_sigma_multiplier", "close_wd_sigma_multiplier"),
    ]
    for ax, (col, label) in zip(axes.ravel(), specs):
        group = df.groupby(col).agg(
            capital_mean=("capital", "mean"),
            capital_max=("capital", "max"),
            wd_mean=("biggest_wd", "mean"),
        ).reset_index()
        ax.plot(group[col], group["capital_mean"], marker="o", label="capital_mean")
        ax.plot(group[col], group["capital_max"], marker="s", label="capital_max")
        ax2 = ax.twinx()
        ax2.plot(group[col], group["wd_mean"], marker="^", color="#ef4444", label="wd_mean")
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("capital")
        ax2.set_ylabel("biggest_wd")
        ax.grid(True, linestyle="--", alpha=0.3)
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=8)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def pick_candidate(df: pd.DataFrame) -> pd.Series:
    filtered = df.copy()
    filtered = filtered[filtered["trade_num"].fillna(0) >= 5]
    if len(filtered) == 0:
        filtered = df.copy()
    filtered = filtered.sort_values(
        ["quality_score", "capital", "trade_num"],
        ascending=[False, False, False],
    )
    return filtered.iloc[0]


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(
        directory.glob(pattern),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    return candidates[0] if candidates else None


def save_experiment_report(
        rows: list[dict],
        comparison_path: Path,
        report_path: Path) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].bar(df["config_key"], df["train_capital"], color="#2563eb")
    axes[0].set_title("best train capital")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(df["config_key"], df["oos_capital"], color="#16a34a")
    axes[1].set_title("oos capital")
    axes[1].tick_params(axis="x", rotation=30)
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    df = df.sort_values("oos_capital", ascending=False).reset_index(drop=True)
    lines = [
        "# long_momentum_ARCH_shock_multi 研究报告",
        "",
        "## 实验范围",
        "",
        f"- 训练区间：{TRAIN_RANGE[0]} - {TRAIN_RANGE[1]}",
        f"- 验证区间：{OOS_RANGE[0]} - {OOS_RANGE[1]}",
        f"- 全样本回看：{FULL_RANGE[0]} - {FULL_RANGE[1]}",
        f"- 对比图：[{comparison_path.name}]({comparison_path.as_posix()})",
        "",
        "## 方案摘要",
        "",
    ]

    for row in df.to_dict("records"):
        lines.extend([
            f"### {row['label']}",
            "",
            f"- 训练最佳参数：`shock_open_multiplier={row['shock_open_multiplier']}`，`close_bar={row['close_bar']}`，`close_speed_sigma_multiplier={row['close_speed_sigma_multiplier']}`，`close_wd_sigma_multiplier={row['close_wd_sigma_multiplier']}`",
            f"- 训练结果：capital={row['train_capital']:.4f}，biggest_wd={row['train_biggest_wd']:.4f}，trade_num={int(row['train_trade_num'])}",
            f"- 验证结果：capital={row['oos_capital']:.4f}，biggest_wd={row['oos_biggest_wd']:.4f}，trade_num={int(row['oos_trade_num'])}",
            f"- 全样本结果：capital={row['full_capital']:.4f}，biggest_wd={row['full_biggest_wd']:.4f}，trade_num={int(row['full_trade_num'])}",
            f"- 训练统计图：[{row['overview_plot_name']}]({row['overview_plot_path']})，[{row['profile_plot_name']}]({row['profile_plot_path']})",
            f"- 验证 HTML：[{row['oos_html_name']}]({row['oos_html_path']})",
            f"- 全样本 HTML：[{row['full_html_name']}]({row['full_html_path']})",
            "",
        ])

    recommendation = df.iloc[0]
    lines.extend([
        "## 推荐",
        "",
        f"- 首选方案：`{recommendation['config_key']}`",
        f"- 推荐参数：`shock_open_multiplier={recommendation['shock_open_multiplier']}`，`close_bar={recommendation['close_bar']}`，`close_speed_sigma_multiplier={recommendation['close_speed_sigma_multiplier']}`，`close_wd_sigma_multiplier={recommendation['close_wd_sigma_multiplier']}`",
        f"- 推荐依据：验证区间 capital={recommendation['oos_capital']:.4f}，全样本 capital={recommendation['full_capital']:.4f}，训练期 quality_score={recommendation['train_quality_score']:.4f}",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8-sig")


def run() -> None:
    research_dir = ensure_dir(REPORT_ROOT)
    rows: list[dict] = []

    for exp in EXPERIMENTS:
        exp_dir = ensure_dir(research_dir / exp["config_key"])
        train_label = f"train_{TRAIN_RANGE[0]}_{TRAIN_RANGE[1]}"
        configure_module_for_grid(exp, TRAIN_RANGE[0], TRAIN_RANGE[1], train_label)
        train_meta = strategy_mod.run_shock_backtest()
        train_stats = enrich_stats(train_meta["outcome_stats"])
        train_stats.to_excel(exp_dir / f"{exp['config_key']} train_enriched.xlsx")

        overview_plot = exp_dir / f"{exp['config_key']} train_overview.png"
        profile_plot = exp_dir / f"{exp['config_key']} train_profile.png"
        save_grid_overview(
            train_stats,
            overview_plot,
            f"{exp['label']} | train {TRAIN_RANGE[0]}-{TRAIN_RANGE[1]}",
        )
        save_parameter_profile(
            train_stats,
            profile_plot,
            f"{exp['label']} | parameter profile",
        )

        candidate = pick_candidate(train_stats)

        oos_label = f"oos_{OOS_RANGE[0]}_{OOS_RANGE[1]}"
        configure_module_for_manual(exp, OOS_RANGE[0], OOS_RANGE[1], candidate, oos_label)
        oos_meta = strategy_mod.run_shock_backtest()
        oos_stats = enrich_stats(oos_meta["outcome_stats"])
        oos_row = oos_stats.iloc[0]

        full_label = f"full_{FULL_RANGE[0]}_{FULL_RANGE[1]}"
        configure_module_for_manual(exp, FULL_RANGE[0], FULL_RANGE[1], candidate, full_label)
        full_meta = strategy_mod.run_shock_backtest()
        full_stats = enrich_stats(full_meta["outcome_stats"])
        full_row = full_stats.iloc[0]

        result_root = Path(train_meta["result_root"])
        oos_html = find_latest_file(result_root / "html", f"*{oos_label}*interactive.html")
        full_html = find_latest_file(result_root / "html", f"*{full_label}*interactive.html")

        rows.append({
            "config_key": exp["config_key"],
            "label": exp["label"],
            "shock_open_multiplier": float(candidate["shock_open_multiplier"]),
            "close_bar": int(candidate["close_bar"]),
            "close_speed_sigma_multiplier": float(candidate["close_speed_sigma_multiplier"]),
            "close_wd_sigma_multiplier": float(candidate["close_wd_sigma_multiplier"]),
            "train_capital": float(candidate["capital"]),
            "train_biggest_wd": float(candidate["biggest_wd"]),
            "train_trade_num": float(candidate["trade_num"]),
            "train_quality_score": float(candidate["quality_score"]),
            "oos_capital": float(oos_row["capital"]),
            "oos_biggest_wd": float(oos_row["biggest_wd"]),
            "oos_trade_num": float(oos_row["trade_num"]),
            "full_capital": float(full_row["capital"]),
            "full_biggest_wd": float(full_row["biggest_wd"]),
            "full_trade_num": float(full_row["trade_num"]),
            "overview_plot_name": overview_plot.name,
            "overview_plot_path": str(overview_plot.resolve()),
            "profile_plot_name": profile_plot.name,
            "profile_plot_path": str(profile_plot.resolve()),
            "oos_html_name": oos_html.name if oos_html is not None else "",
            "oos_html_path": str(oos_html.resolve()) if oos_html is not None else "",
            "full_html_name": full_html.name if full_html is not None else "",
            "full_html_path": str(full_html.resolve()) if full_html is not None else "",
        })

    summary_df = pd.DataFrame(rows).sort_values("oos_capital", ascending=False)
    summary_df.to_excel(research_dir / "research_summary.xlsx", index=False)
    comparison_path = research_dir / "experiment_comparison.png"
    report_path = research_dir / "research_report.md"
    save_experiment_report(rows, comparison_path, report_path)
    print("[Research] summary saved:", research_dir / "research_summary.xlsx")
    print("[Research] report saved:", report_path)


if __name__ == "__main__":
    run()
