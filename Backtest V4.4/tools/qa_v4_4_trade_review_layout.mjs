import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";


const projectRoot = path.resolve(import.meta.dirname, "..");
const runRoot = path.join(
  projectRoot,
  "results",
  "cross_instrument_comparison",
  "runs",
  "k200_train_test_si__combined_350_v56_20260807",
);
const testEntry = path.join(runRoot, "trade_review_k200_test", "index.html");
const comboId =
  "v4_4_rolling_tr_sum_bpall_window_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s332_rx1_e320_bh240_trw20_k1p4_w6_m4p5_7d214509a0";
const outputRoot = path.join(
  projectRoot,
  "project_management",
  "screenshots",
  "trade_review_layout_20260808",
);
fs.mkdirSync(outputRoot, { recursive: true });

function entryUrl(filePath, researchContractId) {
  const url = new URL(pathToFileURL(filePath).href);
  url.searchParams.set("combo_id", comboId);
  url.searchParams.set("research_contract_id", researchContractId);
  return url.href;
}

async function waitForReview(page) {
  await page.waitForSelector("#chart .main-svg", { timeout: 45_000 });
  await page.waitForFunction(
    () =>
      document.querySelector("#tradeSelect")?.options.length > 0 &&
      !document.querySelector("#peerReviewLink")?.hidden,
    null,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(250);
}

async function state(page) {
  return page.evaluate(() => {
    const chart = document.getElementById("chart");
    const peer = document.getElementById("peerReviewLink");
    const annotations = chart?.layout?.annotations || [];
    return {
      title: document.title,
      favicon: document.querySelector('link[rel="icon"]')?.href || "",
      peerLabel: peer?.textContent?.trim() || "",
      peerHref: peer?.href || "",
      tradePickerHeaderExists: Boolean(document.querySelector(".trade-picker-head")),
      tradePickerMetaExists: Boolean(document.querySelector(".trade-picker-meta")),
      metrics: document.getElementById("tradeMetrics")?.innerText || "",
      legendY: chart?.layout?.legend?.y,
      chartAuditAnnotationCount: annotations.filter((item) =>
        /drop=|ratio=|基准=|阈值=|W基准=|^L=/.test(String(item.text || "")),
      ).length,
      url: location.href,
      errors: [],
    };
  });
}

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 2048, height: 1152 } });
const page = await context.newPage();
const errors = [];
page.on("console", (message) => {
  if (message.type() === "error") errors.push(message.text());
});
page.on("pageerror", (error) => errors.push(error.message));

await page.goto(entryUrl(testEntry, "v4_4_cross_instrument_comparison"), {
  waitUntil: "load",
});
await waitForReview(page);
await page.click("#controlsCollapse");
await page.evaluate(() => {
  const reboundIndex = currentTrades.findIndex(
    (trade) => String(trade.exit_reason || "").toLowerCase() === "rebound_threshold",
  );
  if (reboundIndex < 0) return;
  const select = document.getElementById("tradeSelect");
  select.value = String(reboundIndex);
  select.dispatchEvent(new Event("change", { bubbles: true }));
});
await page.waitForTimeout(300);
const test = await state(page);
await page.screenshot({ path: path.join(outputRoot, "test-dark.png"), fullPage: false });

await page.click("#lightThemeBtn");
await page.waitForTimeout(250);
await page.screenshot({ path: path.join(outputRoot, "test-light.png"), fullPage: false });

await page.click("#peerReviewLink");
await waitForReview(page);
await page.click("#controlsCollapse");
const training = await state(page);
await page.screenshot({ path: path.join(outputRoot, "training-light.png"), fullPage: false });

const mobile = await browser.newPage({ viewport: { width: 390, height: 844 } });
const mobileErrors = [];
mobile.on("console", (message) => {
  if (message.type() === "error") mobileErrors.push(message.text());
});
mobile.on("pageerror", (error) => mobileErrors.push(error.message));
await mobile.goto(entryUrl(testEntry, "v4_4_cross_instrument_comparison"), {
  waitUntil: "load",
});
await waitForReview(mobile);
await mobile.click("#controlsCollapse");
const mobileOverflow = await mobile.evaluate(
  () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
);
await mobile.screenshot({ path: path.join(outputRoot, "test-mobile.png"), fullPage: false });
await browser.close();

const assertions = {
  title_has_no_version: !/V\d/i.test(test.title) && test.title === "组合平仓逐笔查看",
  favicon_is_blue_z: test.favicon.includes("%231f77d0") && test.favicon.includes("%3EZ%3C"),
  test_links_to_same_training_combo:
    test.peerLabel === "显示训练集" &&
    new URL(test.peerHref).searchParams.get("combo_id") === comboId,
  training_links_to_same_test_combo:
    training.peerLabel === "显示测试集" &&
    new URL(training.peerHref).searchParams.get("combo_id") === comboId,
  picker_extras_removed:
    !test.tradePickerHeaderExists && !test.tradePickerMetaExists,
  metrics_moved_below_reason:
    /drop=/.test(test.metrics) &&
    /ratio=/.test(test.metrics) &&
    /基准=/.test(test.metrics) &&
    /阈值=/.test(test.metrics) &&
    test.chartAuditAnnotationCount === 0,
  legend_below_chart: Number(test.legendY) < 0,
  no_desktop_errors: errors.length === 0,
  no_mobile_errors: mobileErrors.length === 0,
  no_mobile_page_overflow: mobileOverflow === false,
};
const report = {
  schema_version: 1,
  status: Object.values(assertions).every(Boolean) ? "passed" : "failed",
  assertions,
  test,
  training,
  errors,
  mobileErrors,
  mobileOverflow,
};
fs.writeFileSync(
  path.join(outputRoot, "qa-report.json"),
  `${JSON.stringify(report, null, 2)}\n`,
  "utf8",
);
process.stdout.write(`${JSON.stringify(assertions, null, 2)}\n`);
if (report.status !== "passed") process.exitCode = 1;
