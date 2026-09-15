/**
 * 前端 DOM 冒烟测试
 *
 * 一致性测试证明的是「算得对」，这个脚本管的是「画得出」：
 * 逐个调用渲染函数，断言没有抛异常、没有把 undefined / NaN / [object Object]
 * 泄漏进界面。构建产物是注入式单文件，脚本拼接出错、数据字段改名、
 * 渲染用到未定义变量这类问题，Node 语法检查是发现不了的 —— 必须真的跑一遍渲染。
 *
 * 用法：node tests/ui_smoke.js
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.resolve(__dirname, "..");
const HTML_PATH = path.join(ROOT, "web", "index.html");
const html = fs.readFileSync(HTML_PATH, "utf8");

let failed = 0;
const notes = [];
function check(name, cond, detail) {
  if (cond) { console.log("  ✓ " + name); return true; }
  failed++;
  console.log("  ✗ " + name + (detail ? "\n      " + detail : ""));
  return false;
}

/* ---- 构建产物完整性 ---- */
console.log("\n[1] 构建产物");
check("未残留数据占位符", html.indexOf("/*__ENVSPEC_DATA__*/") < 0 && html.indexOf("__ENVSPEC_DATA__ = null") < 0);
check("包含完整 HTML 结构", html.indexOf("</html>") > 0 && html.indexOf("<!DOCTYPE html>") === 0);
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
check("脚本块数量为 2", blocks.length === 2, "实际 " + blocks.length);
check("页面体积合理 (>100KB)", html.length > 100 * 1024, (html.length / 1024).toFixed(1) + " KB");

/* ---- 带持久注册表的 DOM 桩 ---- */
const els = {};
function makeEl(id) {
  const t = {
    _id: id, innerHTML: "", textContent: "", value: "", title: "", className: "",
    scrollIntoView() {}, addEventListener() {}, getAttribute() { return ""; }
  };
  return new Proxy(t, {
    get(o, k) {
      if (k in o) return o[k];
      if (k === "classList") return { toggle() {}, add() {}, remove() {}, contains() { return false; } };
      if (k === "style") return {};
      if (k === "querySelectorAll" || k === "querySelector") return () => [];
      return () => {};
    },
    set(o, k, v) { o[k] = v; return true; }
  });
}
const sandbox = {
  console, setTimeout, clearTimeout,
  location: { protocol: "file:" },
  document: {
    getElementById: id => (els[id] = els[id] || makeEl(id)),
    querySelectorAll: () => [],
    querySelector: () => null,
    title: ""
  }
};
vm.createContext(sandbox);
vm.runInContext("var window = globalThis; var self = globalThis;", sandbox);
vm.runInContext(blocks[0], sandbox, { filename: "data.js" });

let loadErr = null;
try {
  vm.runInContext(blocks[1] + `
;globalThis.__api = { runAgent, renderResult, renderEval, renderKB, renderArch,
                      understand, DATA: window.__ENVSPEC_DATA__ };`, sandbox, { filename: "engine.js" });
} catch (e) { loadErr = e; }

console.log("\n[2] 脚本装载与初始化");
if (!check("引擎脚本无异常装载", !loadErr, loadErr && (loadErr.message + "\n" + String(loadErr.stack).split("\n").slice(1, 4).join("\n")))) {
  console.log("\n结果：装载即失败，后续断言跳过。");
  process.exit(1);
}
const api = sandbox.__api;
const DATA = api.DATA;

const BAD = [/undefined/, /\bNaN\b/, /\[object Object\]/, /&lt;/, /\$\{/];
function badMark(text) {
  const hit = BAD.find(re => re.test(text));
  if (!hit) return null;
  const m = text.match(hit);
  const i = text.indexOf(m[0]);
  return "命中 " + hit + " @ ..." + text.slice(Math.max(0, i - 60), i + 60).replace(/\s+/g, " ") + "...";
}

check("题目列表已渲染", (els.qlist && els.qlist.innerHTML || "").length > 200);
check("题目列表含 24 题", (els.qlist && els.qlist.innerHTML.match(/class="qitem"/g) || []).length === DATA.golden.items.length);
check("评测面板已渲染", (els.evalBody && els.evalBody.innerHTML || "").length > 800);
check("知识库面板已渲染", (els.kbBody && els.kbBody.innerHTML || "").length > 800);
check("架构面板已渲染", (els.archBody && els.archBody.innerHTML || "").length > 800);
check("门禁徽章已填值", (els.gateBadge && els.gateBadge.textContent || "").indexOf("门禁") === 0);
check("引擎徽章已填值", (els.engineBadge && els.engineBadge.innerHTML || "").length > 0);

console.log("\n[3] 全题目渲染");
let rendered = 0, staleCases = 0, refusedCases = 0;
DATA.golden.items.forEach(item => {
  let res, err = null;
  try { res = api.runAgent(item.query); } catch (e) { err = e; }
  if (!check(item.id + " Agent 执行", !err, err && err.message)) return;
  try { api.renderResult(res, item); } catch (e) { err = e; }
  if (!check(item.id + " 结果渲染", !err, err && err.message)) return;
  const out = els.result.innerHTML;
  if (!check(item.id + " 输出非空", out.length > 400, "长度 " + out.length)) return;
  const bad = badMark(out);
  check(item.id + " 无异常文本泄漏", !bad, bad);
  rendered++;
  if (res.stale_warnings.length) staleCases++;
  if (res.refused) refusedCases++;
});

console.log("\n[4] 关键面板出现率");
check("时效性拦截面板有样例", staleCases > 0, staleCases + " 题触发");
check("拒答路径有样例", refusedCases > 0, refusedCases + " 题触发");
check("被渲染题目数 = 24", rendered === DATA.golden.items.length, rendered + "/" + DATA.golden.items.length);

console.log("\n[5] 面板重渲染幂等");
["renderEval", "renderKB", "renderArch"].forEach(fn => {
  let err = null;
  try { api[fn](); api[fn](); } catch (e) { err = e; }
  check(fn + " 可重复调用", !err, err && err.message);
});

console.log("\n" + "=".repeat(60));
if (failed === 0) {
  console.log("✅ 全部通过：渲染层无异常、无占位符残留、无 undefined/NaN 泄漏");
  console.log("   覆盖 " + rendered + " 题完整渲染路径，其中时效性提示 " + staleCases + " 题、拒答 " + refusedCases + " 题");
} else {
  console.log("❌ " + failed + " 项未通过");
}
console.log("=".repeat(60));
process.exit(failed === 0 ? 0 : 1);
