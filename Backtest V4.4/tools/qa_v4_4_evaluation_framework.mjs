import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";


const projectRoot = path.resolve(import.meta.dirname, "..");
const currentRun = path.join(
  projectRoot,
  "results",
  "cross_instrument_comparison",
  "runs",
  "k200_train_test_si__combined_350_v56_20260807",
  "index.html",
);
const newEntry = path.join(
  projectRoot,
  "results",
  "cross_instrument_comparison",
  "index.html",
);
const screenshotRoot = path.join(
  projectRoot,
  "project_management",
  "screenshots",
  "evaluation_framework_20260808",
);
fs.mkdirSync(screenshotRoot, { recursive: true });

async function inspect(page, filePath, screenshotName) {
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(pathToFileURL(filePath).href, { waitUntil: "load" });
  await page.waitForFunction(
    () => window.V4_4_CROSS_INSTRUMENT_DATA?.rows?.length === 350,
    null,
    { timeout: 30_000 },
  );
  await page.waitForTimeout(250);
  const state = await page.evaluate(() => ({
    title: document.title,
    bodyText: document.body.innerText,
    rowCount: window.V4_4_CROSS_INSTRUMENT_DATA.rows.length,
    firstRow: window.V4_4_CROSS_INSTRUMENT_DATA.rows[0],
    sourceTradeReview: window.V4_4_CROSS_INSTRUMENT_DATA.artifacts.sourceTradeReview,
    sourceTestTradeReview:
      window.V4_4_CROSS_INSTRUMENT_DATA.artifacts.sourceTestTradeReview,
    targetTradeReview:
      window.V4_4_CROSS_INSTRUMENT_DATA.artifacts.targetTradeReview,
    visibleRows: document.querySelectorAll("tbody tr").length,
  }));
  await page.screenshot({
    path: path.join(screenshotRoot, screenshotName),
    fullPage: false,
  });
  return { ...state, errors, finalUrl: page.url() };
}

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 2048, height: 1152 } });
const oldPage = await context.newPage();
const newPage = await context.newPage();
const oldState = await inspect(oldPage, currentRun, "current-run.png");
const newState = await inspect(newPage, newEntry, "date-package-entry.png");
await browser.close();

const report = {
  schema_version: 1,
  status:
    oldState.errors.length === 0 &&
    newState.errors.length === 0 &&
    oldState.title === newState.title &&
    oldState.bodyText === newState.bodyText &&
    JSON.stringify(oldState.firstRow) === JSON.stringify(newState.firstRow) &&
    oldState.visibleRows === newState.visibleRows
      ? "passed"
      : "failed",
  current: oldState,
  framework: newState,
  assertions: {
    title_equal: oldState.title === newState.title,
    body_text_equal: oldState.bodyText === newState.bodyText,
    first_row_equal:
      JSON.stringify(oldState.firstRow) === JSON.stringify(newState.firstRow),
    visible_row_count_equal: oldState.visibleRows === newState.visibleRows,
    current_console_errors: oldState.errors.length,
    framework_console_errors: newState.errors.length,
  },
};
fs.writeFileSync(
  path.join(screenshotRoot, "qa-report.json"),
  `${JSON.stringify(report, null, 2)}\n`,
  "utf8",
);
process.stdout.write(`${JSON.stringify(report.assertions, null, 2)}\n`);
if (report.status !== "passed") process.exitCode = 1;
