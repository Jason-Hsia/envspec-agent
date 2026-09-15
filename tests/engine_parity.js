/**
 * 一致性校验 · Node 侧
 *
 * 把 web/index.html 里的内置引擎拉进 VM 跑一遍黄金集，导出每题的
 * 规划序列 / 引用列表 / 拒答标记 / 答案全文，供 parity_check.py 与
 * Python 后端的真实输出做逐项比对。
 *
 * 用法：node tests/engine_parity.js <输出路径>
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.resolve(__dirname, "..");
const HTML = path.join(ROOT, "web", "index.html");
const OUT = process.argv[2] || path.join(ROOT, "out", "_js_parity.json");

if (!fs.existsSync(HTML)) {
  console.error("[error] 未找到 web/index.html，请先运行 python web/build_web.py");
  process.exit(1);
}
const html = fs.readFileSync(HTML, "utf8");
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (blocks.length < 2) {
  console.error("[error] index.html 中未找到预期的两个脚本块，模板可能被改动");
  process.exit(1);
}

/* ---- 极简 DOM 桩：引擎只用到 getElementById / querySelectorAll ---- */
function makeEl() {
  const target = { innerHTML: "", textContent: "", value: "", title: "", className: "" };
  return new Proxy(target, {
    get(o, k) {
      if (k in o) return o[k];
      if (k === "classList") return { toggle() {}, add() {}, remove() {}, contains() { return false; } };
      if (k === "style") return {};
      if (k === "querySelectorAll" || k === "querySelector") return () => [];
      if (k === "getAttribute") return () => "";
      return () => {};
    },
    set(o, k, v) { o[k] = v; return true; }
  });
}
const sandbox = {
  console,
  setTimeout,
  clearTimeout,
  location: { protocol: "file:" },
  document: {
    getElementById: () => makeEl(),
    querySelectorAll: () => [],
    querySelector: () => null,
    title: ""
  }
};
vm.createContext(sandbox);
vm.runInContext("var window = globalThis; var self = globalThis;", sandbox);

vm.runInContext(blocks[0], sandbox, { filename: "data.js" });
vm.runInContext(
  blocks[1] + "\n;globalThis.__runAgent = runAgent; globalThis.__D = window.__ENVSPEC_DATA__;",
  sandbox, { filename: "engine.js" }
);

const DATA = sandbox.__D;
const items = DATA.golden.items;
const results = {};
for (const it of items) {
  const res = sandbox.__runAgent(it.query);
  results[it.id] = {
    query: it.query,
    plan: res.plan.map(p => p.tool),
    citations: res.citations.map(c => ({ idx: c.idx, std_id: c.std_id, table: c.table,
      clause_no: c.clause_no, value: c.value, unit: c.unit })),
    refused: res.refused,
    stale_warnings: res.stale_warnings,
    n_limits: res.limits.length,
    answer_sha: require("crypto").createHash("sha256").update(res.answer, "utf8").digest("hex").slice(0, 16),
    answer_len: res.answer.length
  };
}
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(results, null, 1), "utf8");
console.log(`[ok] JS 引擎已跑完 ${items.length} 题 -> ${OUT}`);
