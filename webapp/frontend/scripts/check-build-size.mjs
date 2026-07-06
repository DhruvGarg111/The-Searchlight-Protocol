import { gzipSync } from "node:zlib";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const DIST_DIR = fileURLToPath(new URL("../dist", import.meta.url));
const ASSETS_DIR = join(DIST_DIR, "assets");
const TOLERANCE = Number.parseFloat(process.env.SEARCHLIGHT_SIZE_BUDGET_TOLERANCE ?? "0.05");

const budgets = {
  js: { raw: 287_150, gzip: 92_910 },
  css: { raw: 26_060, gzip: 6_000 },
};

function collectAssetBytes(extension) {
  const files = readdirSync(ASSETS_DIR)
    .filter((file) => file.endsWith(`.${extension}`))
    .map((file) => join(ASSETS_DIR, file));

  return files.reduce(
    (total, file) => {
      const source = readFileSync(file);
      return {
        raw: total.raw + statSync(file).size,
        gzip: total.gzip + gzipSync(source).length,
      };
    },
    { raw: 0, gzip: 0 },
  );
}

function assertWithinBudget(kind, measured) {
  const budget = budgets[kind];
  const rawLimit = budget.raw * (1 + TOLERANCE);
  const gzipLimit = budget.gzip * (1 + TOLERANCE);
  const rawOk = measured.raw <= rawLimit;
  const gzipOk = measured.gzip <= gzipLimit;

  console.log(
    `${kind.toUpperCase()}: raw ${measured.raw} / ${Math.round(rawLimit)} bytes, gzip ${measured.gzip} / ${Math.round(gzipLimit)} bytes`,
  );

  if (!rawOk || !gzipOk) {
    throw new Error(`${kind.toUpperCase()} bundle size exceeds budget`);
  }
}

assertWithinBudget("js", collectAssetBytes("js"));
assertWithinBudget("css", collectAssetBytes("css"));
