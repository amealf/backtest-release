import fs from "node:fs";
import fsPromises from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import os from "node:os";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);

function loadPlaywright() {
  try { return require("playwright"); } catch {}
  const runtimeRoot = path.join(
    os.homedir(), ".cache", "codex-runtimes", "codex-primary-runtime",
    "dependencies", "node", "node_modules",
  );
  return require(path.join(runtimeRoot, "playwright"));
}

const runRoot = path.resolve(process.argv[2] || "");
const stableMainRoot = path.resolve(process.argv[3] || "");
if (!runRoot || !stableMainRoot) {
  throw new Error("usage: node qa_v4_4_cross_instrument_comparison.mjs <run-root> <stable-main-root>");
}
const pagePath = path.join(runRoot, "index.html");
const dataPath = path.join(runRoot, "comparison_data.js");
const reportPath = path.join(runRoot, "migration_report.json");
const tradePath = path.join(runRoot, "trade_review", "index.html");
const tradeManifestPath = path.join(runRoot, "trade_review", "trade_review_manifest.json");
const stableMainPath = path.join(stableMainRoot, "index.html");
for (const target of [pagePath, dataPath, reportPath, tradePath, tradeManifestPath, stableMainPath]) {
  if (!fs.existsSync(target)) throw new Error(`required resource is missing: ${target}`);
}
const migrationReport = JSON.parse(fs.readFileSync(reportPath, "utf8"));
const expectedCandidateCount = Number(migrationReport.evaluation?.candidate_count || 0);
const expectedPositiveCount = Number(migrationReport.evaluation?.target_positive_candidate_count || 0);
if (expectedCandidateCount < 1 || expectedPositiveCount < 0) {
  throw new Error("migration report lacks the expected candidate counts");
}

const sha256 = (target) => crypto.createHash("sha256").update(fs.readFileSync(target)).digest("hex");
const qaRoot = path.join(runRoot, "qa");
await fsPromises.mkdir(qaRoot, { recursive: true });
const runtimeErrors = [];
const externalRequests = [];

function bindDiagnostics(page) {
  page.on("pageerror", (error) => runtimeErrors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") runtimeErrors.push(message.text());
  });
  page.on("request", (request) => {
    const url = request.url();
    if (!url.startsWith("file:") && !url.startsWith("data:") && !url.startsWith("blob:")) {
      externalRequests.push(url);
    }
  });
}

async function visualState(page) {
  return page.evaluate(() => ({
    title: document.title,
    replacements: (document.body.innerText.match(/�/g) || []).length,
    viewportWidth: window.innerWidth,
    documentWidth: document.documentElement.scrollWidth,
    bodyWidth: document.body.scrollWidth,
    bodyTextLength: document.body.innerText.length,
    tableCount: document.querySelectorAll("table").length,
  }));
}

function assertVisual(state, label) {
  if (
    state.replacements
    || state.documentWidth > state.viewportWidth + 1
    || state.bodyWidth > state.viewportWidth + 1
    || state.bodyTextLength < 500
    || state.tableCount < 2
  ) {
    throw new Error(`${label} visual state failed: ${JSON.stringify(state)}`);
  }
}

const { chromium } = loadPlaywright();
const browser = await chromium.launch({ headless: true, args: ["--allow-file-access-from-files"] });
const states = {};
let firstTradeHref = "";
try {
  for (const [label, viewport] of Object.entries({
    desktop: { width: 1440, height: 950 },
    mobile: { width: 390, height: 844 },
  })) {
    const page = await browser.newPage({ viewport });
    bindDiagnostics(page);
    await page.goto(pathToFileURL(pagePath).href, { waitUntil: "load", timeout: 180000 });
    await page.waitForFunction(
      (count) => window.V4_4_CROSS_INSTRUMENT_DATA?.rows?.length === count,
      expectedCandidateCount,
      { timeout: 180000 },
    );
    const contract = await page.evaluate(() => ({
      rowCount: window.V4_4_CROSS_INSTRUMENT_DATA.rows.length,
      selectorCount: document.querySelectorAll("select").length,
      returnViewCount: document.querySelectorAll("[data-return-view]").length,
      activeReturnView: document.querySelector("[data-return-view].active")?.textContent.trim() || "",
      headerCount: document.querySelectorAll("#comparison-table thead tr:first-child th").length,
      sortCount: document.querySelectorAll("#comparison-table [data-sort]").length,
      rankLinkCount: document.querySelectorAll("#comparison-table a.rank-link").length,
      firstRankText: document.querySelector("#comparison-table a.rank-link")?.textContent.trim() || "",
      firstTradeHref: document.querySelector("#comparison-table a.rank-link")?.href || "",
      firstTradeTarget: document.querySelector("#comparison-table a.rank-link")?.target || "",
      firstTradeRel: document.querySelector("#comparison-table a.rank-link")?.rel || "",
      rankHeaderText: document.querySelector("#comparison-table thead th:first-child")?.textContent.trim() || "",
      globalFilterPresent: Boolean(document.querySelector("#global-filter")),
      selectorPanelPresent: Boolean(document.querySelector(".comparison-selector")),
      sourceInstrument: document.querySelector("#source-instrument")?.value,
      sourceInterval: document.querySelector("#source-interval")?.value,
      targetInstrument: document.querySelector("#target-instrument")?.value,
      targetInterval: document.querySelector("#target-interval")?.value,
      filterFieldCount: document.querySelector("#filter-field")?.options?.length || 0,
      candidateSourcePresent: document.body.innerText.includes("候选来源"),
      oldMainLinkPresent: document.body.innerText.includes("参数总入口"),
      simainHeaderText: document.querySelector('[data-sort="target_cost_total_return"]')?.textContent.trim() || "",
      k200HeaderText: document.querySelector('[data-sort="source_cost_total_return"]')?.textContent.trim() || "",
      k200HeaderHelp: document.querySelector('[data-sort="source_cost_total_return"]')?.title || "",
      firstTargetTotal: Number(window.V4_4_CROSS_INSTRUMENT_DATA.rows
        .slice()
        .sort((a, b) => (
          Number(b.target_cost_total_return) - Number(a.target_cost_total_return)
          || Number(a.target_cost_max_drawdown_abs) - Number(b.target_cost_max_drawdown_abs)
          || Number(b.target_cost_median_trade) - Number(a.target_cost_median_trade)
          || String(a.combo_id).localeCompare(String(b.combo_id))
        ))[0].target_cost_total_return),
      firstSourceTotal: Number(window.V4_4_CROSS_INSTRUMENT_DATA.rows
        .slice()
        .sort((a, b) => (
          Number(b.target_cost_total_return) - Number(a.target_cost_total_return)
          || Number(a.target_cost_max_drawdown_abs) - Number(b.target_cost_max_drawdown_abs)
          || Number(b.target_cost_median_trade) - Number(a.target_cost_median_trade)
          || String(a.combo_id).localeCompare(String(b.combo_id))
        ))[0].source_cost_total_return),
      renderedFirstTargetTotal: document.querySelectorAll("#comparison-table tbody tr td")[
        Array.from(document.querySelectorAll("#comparison-table [data-sort]")).findIndex(
          (node) => node.dataset.sort === "target_cost_total_return",
        )
      ]?.textContent.trim(),
      renderedFirstSourceTotal: document.querySelectorAll("#comparison-table tbody tr td")[
        Array.from(document.querySelectorAll("#comparison-table [data-sort]")).findIndex(
          (node) => node.dataset.sort === "source_cost_total_return",
        )
      ]?.textContent.trim(),
      noScore: document.body.innerText.includes("综合 score") && !window.V4_4_CROSS_INSTRUMENT_DATA.report.evaluation.combined_score,
    }));
    if (!firstTradeHref) firstTradeHref = contract.firstTradeHref;
    if (
      contract.rowCount !== expectedCandidateCount
      || contract.selectorCount !== 6
      || contract.returnViewCount !== 2
      || contract.activeReturnView !== "手续费／滑点后"
      || contract.headerCount !== contract.sortCount
      || contract.rankLinkCount !== expectedCandidateCount
      || contract.firstRankText !== "查看 #1"
      || !contract.firstTradeHref.includes("trade_review/index.html?combo_id=")
      || contract.firstTradeTarget !== "_blank"
      || !contract.firstTradeRel.split(/\s+/).includes("noopener")
      || contract.rankHeaderText !== "成本后排名 ▲"
      || !contract.globalFilterPresent
      || !contract.selectorPanelPresent
      || contract.sourceInstrument !== "K200"
      || contract.sourceInterval !== "2026-05-26—2026-07-08"
      || contract.targetInstrument !== "SImain"
      || contract.targetInterval !== "2026-01-29—2026-02-23"
      || contract.filterFieldCount !== contract.headerCount
      || contract.candidateSourcePresent
      || contract.oldMainLinkPresent
      || !contract.k200HeaderText.startsWith("K200 总收益")
      || contract.simainHeaderText !== "SImain 总收益"
      || !contract.k200HeaderHelp.includes("长于 SImain")
      || !contract.noScore
      || contract.renderedFirstSourceTotal !== `${(contract.firstSourceTotal * 100).toFixed(3)}%`
      || contract.renderedFirstTargetTotal !== `${(contract.firstTargetTotal * 100).toFixed(3)}%`
    ) {
      throw new Error(`page contract failed: ${JSON.stringify(contract)}`);
    }
    await page.locator("#open-run").click();
    const selectionStatus = await page.locator("#selection-status").innerText();
    if (!selectionStatus.includes("当前页面已经是所选比较范围")) {
      throw new Error(`run selector contract failed: ${selectionStatus}`);
    }
    await page.locator("#global-filter").fill("SIH6");
    const globalFilteredCount = await page.locator("#comparison-table tbody tr").count();
    if (globalFilteredCount !== 0) throw new Error(`global filter expected empty result, got ${globalFilteredCount}`);
    await page.locator("#global-filter").fill("");
    await page.locator("#filter-field").selectOption("target_cost_total_return");
    await page.locator("#filter-operator").selectOption("gt");
    await page.locator("#filter-value").fill("0");
    await page.locator("#add-filter").click();
    const positiveCount = await page.locator("#comparison-table tbody tr").count();
    if (positiveCount !== expectedPositiveCount || await page.locator("#active-filters .filter-chip").count() !== 1) {
      throw new Error(`field filter contract failed: positive=${positiveCount}`);
    }
    await page.locator("#clear-filters").click();
    if (await page.locator("#comparison-table tbody tr").count() !== expectedCandidateCount) {
      throw new Error("clear filters did not restore all candidates");
    }
    await page.locator('[data-return-view="gross"]').click();
    const grossContract = await page.evaluate(() => {
      const rows = window.V4_4_CROSS_INSTRUMENT_DATA.rows.slice().sort((a, b) => (
        Number(b.target_gross_total_return) - Number(a.target_gross_total_return)
        || Number(a.target_gross_max_drawdown_abs) - Number(b.target_gross_max_drawdown_abs)
        || Number(b.target_gross_median_trade) - Number(a.target_gross_median_trade)
        || String(a.combo_id).localeCompare(String(b.combo_id))
      ));
      return {
        rankHeaderText: document.querySelector("#comparison-table thead th:first-child")?.textContent.trim() || "",
        activeReturnView: document.querySelector("[data-return-view].active")?.textContent.trim() || "",
        expectedSourceTotal: Number(rows[0].source_gross_total_return),
        expectedTargetTotal: Number(rows[0].target_gross_total_return),
        renderedTargetTotal: document.querySelectorAll("#comparison-table tbody tr td")[
          Array.from(document.querySelectorAll("#comparison-table [data-sort]")).findIndex(
            (node) => node.dataset.sort === "target_gross_total_return",
          )
        ]?.textContent.trim(),
        renderedSourceTotal: document.querySelectorAll("#comparison-table tbody tr td")[
          Array.from(document.querySelectorAll("#comparison-table [data-sort]")).findIndex(
            (node) => node.dataset.sort === "source_gross_total_return",
          )
        ]?.textContent.trim(),
      };
    });
    if (
      grossContract.rankHeaderText !== "成本前排名 ▲"
      || grossContract.activeReturnView !== "无手续费／滑点"
      || grossContract.renderedSourceTotal !== `${(grossContract.expectedSourceTotal * 100).toFixed(3)}%`
      || grossContract.renderedTargetTotal !== `${(grossContract.expectedTargetTotal * 100).toFixed(3)}%`
    ) {
      throw new Error(`gross return view failed: ${JSON.stringify(grossContract)}`);
    }
    await page.locator('[data-return-view="cost_adjusted"]').click();
    if (label === "desktop") {
      const popupPromise = page.waitForEvent("popup");
      await page.locator("#comparison-table a.rank-link").first().click();
      const popup = await popupPromise;
      await popup.waitForLoadState("load");
      if (!popup.url().includes("trade_review/index.html?combo_id=")) {
        throw new Error(`rank popup route failed: ${popup.url()}`);
      }
      await popup.close();
    }
    await page.locator('[data-sort="target_cost_max_drawdown_abs"]').click();
    await page.locator('[data-sort="view_rank"]').click();
    await page.locator("#theme").click();
    await page.evaluate(() => {
      const tableWrap = document.querySelector("#comparison-table");
      if (tableWrap) tableWrap.scrollLeft = 0;
      window.scrollTo(0, 0);
    });
    const state = await visualState(page);
    assertVisual(state, label);
    const screenshotPath = path.join(qaRoot, `cross_instrument_${label}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    states[label] = { ...state, screenshot: screenshotPath, screenshot_sha256: sha256(screenshotPath) };
    await page.close();
  }

  const trade = await browser.newPage({ viewport: { width: 1440, height: 950 } });
  bindDiagnostics(trade);
  const tradeUrl = new URL(firstTradeHref);
  tradeUrl.searchParams.set("reason", "entry");
  await trade.goto(tradeUrl.href, { waitUntil: "load", timeout: 180000 });
  await trade.waitForFunction((count) => (
    window.Plotly
    && window.PROCESS_PAYLOAD?.features?.datetime?.length
    && window.ALL_RESULTS_TRADE_EXPLAIN_CATALOG?.rows?.length === count
    && document.querySelector("#chart")?._fullLayout
    && document.querySelector("#tradeSelect")?.options?.length
  ), expectedCandidateCount, { timeout: 180000 });
  const tradeState = await trade.evaluate(() => ({
    catalogCount: window.ALL_RESULTS_TRADE_EXPLAIN_CATALOG.rows.length,
    researchLabel: window.ALL_RESULTS_TRADE_EXPLAIN_CATALOG.research_samples?.[0]?.label,
    sourceBarCount: window.PROCESS_PAYLOAD.features.datetime.length,
    selectedComboId: new URL(location.href).searchParams.get("combo_id"),
    selectedTradeCount: document.querySelector("#tradeSelect")?.options?.length || 0,
    traceTypes: Array.from(document.querySelector("#chart")?.data || []).map((trace) => trace.type || "scatter"),
    reasonHeading: document.querySelector("#reasonHeading")?.innerText || "",
    replacements: (document.body.innerText.match(/�/g) || []).length,
    documentWidth: document.documentElement.scrollWidth,
    viewportWidth: innerWidth,
  }));
  if (
    tradeState.catalogCount !== expectedCandidateCount
    || tradeState.researchLabel !== "SImain 冻结候选迁移验证"
    || tradeState.sourceBarCount < 90000
    || !tradeState.selectedComboId
    || tradeState.selectedTradeCount < 1
    || !tradeState.traceTypes.includes("candlestick")
    || tradeState.reasonHeading !== "开仓理由"
    || tradeState.replacements
    || tradeState.documentWidth > tradeState.viewportWidth + 1
  ) {
    throw new Error(`SImain trade-review contract failed: ${JSON.stringify(tradeState)}`);
  }
  const tradeDesktopPath = path.join(qaRoot, "simain_trade_desktop.png");
  await trade.screenshot({ path: tradeDesktopPath, fullPage: true });
  await trade.setViewportSize({ width: 390, height: 844 });
  await trade.waitForTimeout(200);
  const tradeMobileState = await trade.evaluate(() => ({
    replacements: (document.body.innerText.match(/�/g) || []).length,
    documentWidth: document.documentElement.scrollWidth,
    viewportWidth: innerWidth,
    chartWidth: document.querySelector("#chart")?.getBoundingClientRect().width || 0,
  }));
  if (tradeMobileState.replacements || tradeMobileState.documentWidth > tradeMobileState.viewportWidth + 1 || tradeMobileState.chartWidth < 250) {
    throw new Error(`SImain trade-review mobile state failed: ${JSON.stringify(tradeMobileState)}`);
  }
  const tradeMobilePath = path.join(qaRoot, "simain_trade_mobile.png");
  await trade.screenshot({ path: tradeMobilePath, fullPage: true });
  states.trade = {
    ...tradeState,
    mobile: tradeMobileState,
    desktop_screenshot: tradeDesktopPath,
    desktop_screenshot_sha256: sha256(tradeDesktopPath),
    mobile_screenshot: tradeMobilePath,
    mobile_screenshot_sha256: sha256(tradeMobilePath),
  };
  await trade.close();

  const hub = await browser.newPage({ viewport: { width: 1440, height: 950 } });
  bindDiagnostics(hub);
  const stableSource = fs.readFileSync(stableMainPath, "utf8");
  if (!stableSource.includes("entry-nav") || !stableSource.includes("cross_instrument_comparison") || stableSource.includes("location.replace")) {
    throw new Error("stable cumulative navigation shell is missing");
  }
  await hub.goto(pathToFileURL(stableMainPath).href, { waitUntil: "domcontentloaded", timeout: 180000 });
  await hub.waitForSelector(".entry-nav", { timeout: 180000 });
  await hub.waitForFunction(() => document.querySelector("#current-result")?.contentDocument?.querySelector("#table"), { timeout: 180000 });
  const hubContract = await hub.evaluate(() => ({
    switchNavPresent: Boolean(document.querySelector(".entry-nav")),
    crossHref: document.querySelector('.entry-nav a[href*="cross_instrument"]')?.getAttribute("href") || "",
    snapshotFrameUrl: document.querySelector("#current-result")?.contentWindow?.location?.href || "",
    replacements: (document.body.innerText.match(/�/g) || []).length,
  }));
  if (!hubContract.switchNavPresent || !hubContract.crossHref.includes("cross_instrument_comparison") || !hubContract.snapshotFrameUrl.includes("/snapshots/") || hubContract.replacements) {
    throw new Error(`stable hub contract failed: ${JSON.stringify(hubContract)}`);
  }
  await hub.close();
} finally {
  await browser.close();
}

if (runtimeErrors.length || externalRequests.length) {
  throw new Error(JSON.stringify({ runtimeErrors, externalRequests }, null, 2));
}

const evidence = {
  schema_version: 1,
  status: "pass",
  generated_at: new Date().toISOString(),
  page: { path: pagePath, size_bytes: fs.statSync(pagePath).size, sha256: sha256(pagePath) },
  data: { path: dataPath, size_bytes: fs.statSync(dataPath).size, sha256: sha256(dataPath) },
  report: { path: reportPath, size_bytes: fs.statSync(reportPath).size, sha256: sha256(reportPath) },
  trade: { path: tradePath, size_bytes: fs.statSync(tradePath).size, sha256: sha256(tradePath) },
  trade_manifest: { path: tradeManifestPath, size_bytes: fs.statSync(tradeManifestPath).size, sha256: sha256(tradeManifestPath) },
  stable_main: { path: stableMainPath, size_bytes: fs.statSync(stableMainPath).size, sha256: sha256(stableMainPath) },
  states,
  runtime_errors: runtimeErrors,
  external_requests: externalRequests,
};
const evidencePath = path.join(qaRoot, "cross_instrument_qa.json");
await fsPromises.writeFile(evidencePath, `${JSON.stringify(evidence, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ evidencePath, status: evidence.status, states: Object.keys(states) }));
