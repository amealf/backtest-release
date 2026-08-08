# GPT Pro review scope

## Review only package-contained evidence

Review the source code, package-contained tests, user interface, documentation consistency, package manifest and audit, security issues that can be established from the included files, and other findings that can be evaluated without external data.

## Out of scope

Do not review, fail, or report findings that depend on any item below:

- the local `F:` drive or any other machine-specific drive;
- local K200, SI, Chinext, Nasdaq, or other market-data files that are absent from this ZIP;
- omitted raw tick data, 15-second market data, full trade ledgers, bulk result CSV files, or excluded per-coordinate trade chunks;
- local browser profiles, `file:///` navigation, Chrome state, or machine-specific absolute paths;
- reproducing data-dependent backtests or tests whose declared input artifacts are intentionally absent from this compact release.

Treat such items as `out of scope / not assessable from the package`, not as release defects. The two test skips recorded by this release are expected because they require closed historical local artifacts that the compact ZIP intentionally excludes.

Do not infer defects from missing local data. Continue reviewing every other package-contained concern.

---

# GPT Pro 审阅范围

请审阅压缩包内可以独立判断的源码、包内测试、界面、文档一致性、打包清单与审计，以及能够从包内文件确认的安全问题和其他问题。

以下内容不属于审阅范围：本机 `F:` 盘或其他机器专用磁盘、压缩包没有携带的 K200／SI／Chinext／Nasdaq 等行情文件、原始 tick、15 秒行情、完整逐笔账本、大型结果 CSV、未纳入的参数逐笔区块、本地浏览器配置、`file:///` 导航、Chrome 当前状态、机器专用绝对路径，以及依赖上述本地文件的回测或测试复现。

遇到这些内容时，请标记为「超出范围／无法仅凭压缩包判断」，不要作为发布缺陷。发布记录中的两项跳过依赖压缩包有意排除的封闭历史本地文件，属于预期结果。请继续审阅其余所有能够从压缩包独立判断的内容。
