// Generic in-browser predictor for tabular models exported to ONNX.
// Everything about the page (fields, metrics, texts) comes from meta.json.

const $ = (id) => document.getElementById(id);
let meta, session;

function fmtValue(v, fmt) {
  if (fmt === "percent") return (v * 100).toFixed(1) + "%";
  if (fmt === "currency") return "$" + Math.round(v).toLocaleString("en-US");
  if (typeof v === "number") return Number.isInteger(v) ? v.toString() : v.toFixed(3);
  return v;
}

function renderHeader() {
  document.title = meta.title;
  $("title").textContent = meta.title;
  $("lead").textContent = meta.subtitle;
  if (meta.eyebrow) $("eyebrow").textContent = meta.eyebrow;
  $("chips").innerHTML = (meta.tags || []).map((t) => `<span class="chip">${t}</span>`).join("");
  const links = [];
  if (meta.repo_url) links.push(`<a class="btn" href="${meta.repo_url}" target="_blank" rel="noopener">GitHub repo</a>`);
  if (meta.notebook_url) links.push(`<a class="btn" href="${meta.notebook_url}" target="_blank" rel="noopener">Training notebook</a>`);
  if (meta.dataset && meta.dataset.url) links.push(`<a class="btn" href="${meta.dataset.url}" target="_blank" rel="noopener">Dataset</a>`);
  $("links").innerHTML = links.join("");
  if (meta.result_title) $("result-title").textContent = meta.result_title;
  $("footer").innerHTML = meta.footer || "";
}

function renderForm() {
  $("form").innerHTML = meta.features.map((f) => {
    const hint = f.help ? ` <span class="hint">· ${f.help}</span>` : "";
    let input;
    if (f.type === "category") {
      input = `<select id="f_${f.name}">${f.options.map((o) => {
        const label = (f.labels && f.labels[o]) || o;
        return `<option value="${o}"${o === f.default ? " selected" : ""}>${label}</option>`;
      }).join("")}</select>`;
    } else {
      input = `<input id="f_${f.name}" type="number" min="${f.min}" max="${f.max}" step="${f.step || "any"}" value="${f.default}">`;
    }
    return `<div class="field"><label for="f_${f.name}">${f.label}${hint}</label>${input}</div>`;
  }).join("");
}

function readForm() {
  const row = {};
  for (const f of meta.features) {
    const el = $("f_" + f.name);
    row[f.name] = f.type === "category" ? el.value : Number(el.value);
  }
  return row;
}

function fillForm(row) {
  for (const f of meta.features) if (row[f.name] !== undefined) $("f_" + f.name).value = row[f.name];
}

async function predict() {
  if (!session) return;
  const row = readForm();
  const feeds = {};
  for (const f of meta.features) {
    feeds[f.name] = f.type === "category"
      ? new ort.Tensor("string", [String(row[f.name])], [1, 1])
      : new ort.Tensor("float32", Float32Array.from([row[f.name]]), [1, 1]);
  }
  const out = await session.run(feeds);
  if (meta.task === "regression") {
    let y = out[session.outputNames[0]].data[0];
    if (meta.target_transform === "log1p") y = Math.expm1(y);
    showRegression(y);
  } else {
    const probs = out[meta.proba_output || "probabilities"].data;
    const idx = meta.classes.indexOf(meta.positive_class);
    showClassification(probs[idx]);
  }
}

function showClassification(p) {
  const t = meta.thresholds || [0.35, 0.65];
  const level = p < t[0] ? 0 : p < t[1] ? 1 : 2;
  const colors = ["var(--good)", "var(--warn)", "var(--bad)"];
  const color = meta.positive_is_good ? colors[2 - level] : colors[level];
  const verdict = meta.verdicts[level];
  $("result").innerHTML = `
    <div class="muted">${meta.output_label}</div>
    <div class="score" style="color:${color}">${(p * 100).toFixed(1)}%</div>
    <div class="verdict" style="color:${color}">${verdict}</div>
    <div class="bar"><span style="width:${(p * 100).toFixed(1)}%;background:${color}"></span></div>
    <p class="muted">${meta.result_note || ""}</p>`;
}

function showRegression(y) {
  $("result").innerHTML = `
    <div class="muted">${meta.output_label}</div>
    <div class="score" style="color:var(--accent)">${fmtValue(y, meta.output_format)}</div>
    <p class="muted">${meta.result_note || ""}</p>`;
}

function renderMetrics() {
  $("metrics").innerHTML = Object.entries(meta.test_metrics).map(([k, v]) =>
    `<div class="metric"><b>${typeof v === "number" ? fmtValue(v, meta.metric_formats?.[k]) : v}</b><span>${k}</span></div>`
  ).join("");
  if (meta.metrics_note) $("metrics-sub").textContent = meta.metrics_note;

  if (!meta.comparison || !meta.comparison.length) { $("compare-card").remove(); }
  else {
    const cols = Object.keys(meta.comparison[0]).filter((c) => c !== "model");
    $("compare").innerHTML =
      `<thead><tr><th>Model</th>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
      meta.comparison.map((r) =>
        `<tr class="${r.model === meta.model_name ? "best" : ""}"><td>${r.model}</td>${cols.map((c) =>
          `<td>${typeof r[c] === "number" ? r[c].toFixed(3) : r[c]}</td>`).join("")}</tr>`).join("") + "</tbody>";
  }

  if (!meta.figures || !meta.figures.length) { $("figs-card").remove(); }
  else $("figs").innerHTML = meta.figures.map((f) => `<figure style="margin:0"><img src="${f.src}" alt="${f.caption}" loading="lazy"><figcaption class="muted">${f.caption}</figcaption></figure>`).join("");

  $("steps").innerHTML = (meta.steps || []).map((s) => `<li>${s}</li>`).join("");
}

async function main() {
  meta = await (await fetch("meta.json")).json();
  renderHeader();
  renderForm();
  renderMetrics();
  $("predict").onclick = predict;
  $("reset").onclick = () => { renderForm(); predict(); };
  $("example").onclick = () => {
    const ex = meta.examples[Math.floor(Math.random() * meta.examples.length)];
    fillForm(ex); predict();
  };
  $("form").addEventListener("change", predict);
  try {
    session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
    await predict();
  } catch (e) {
    $("result").innerHTML = `<div class="placeholder">Could not load the model: ${e.message}</div>`;
  }
}
main();
