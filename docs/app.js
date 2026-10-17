"use strict";

const byId = id => document.getElementById(id);
const number = value => new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
const integer = value => new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);

async function readJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Unable to load ${path} (${response.status})`);
  return response.json();
}

function validCase(item) {
  return item && [item.m, item.n, item.tickets, item.direct_us, item.packed_us, item.grouped_us, item.mul_saved_pct].every(Number.isFinite)
    && [item.m, item.n, item.tickets, item.direct_us, item.packed_us, item.grouped_us].every(value => value > 0);
}

function showCase(item) {
  const maximum = Math.max(item.direct_us, item.packed_us, item.grouped_us);
  const timings = [["direct", item.direct_us], ["packed", item.packed_us], ["grouped", item.grouped_us]];
  for (const [name, duration] of timings) {
    byId(`${name}-value`).textContent = `${number(duration)} µs`;
    byId(`${name}-bar`).style.width = `${duration / maximum * 100}%`;
  }
  byId("axis-max").textContent = `${number(maximum)} µs / batch`;
  const best = Math.min(item.direct_us, item.packed_us);
  const faster = best > item.grouped_us;
  const bestName = item.packed_us < item.direct_us ? "packed direct" : "direct";
  const factor = faster ? best / item.grouped_us : item.grouped_us / best;
  byId("speed-ratio").textContent = `${number(factor)}×`;
  byId("speed-description").textContent = factor < 1.005 ? "about the same time" : faster ? `faster than ${bestName}` : `slower than ${bestName}`;
  byId("multiplications-saved").textContent = `${number(item.mul_saved_pct)}%`;
  byId("batch-tickets").textContent = integer(item.tickets);
  byId("benchmark-chart").setAttribute("aria-label", `${item.m} by ${item.n} batch. Direct: ${number(item.direct_us)} microseconds; packed direct: ${number(item.packed_us)} microseconds; grouped: ${number(item.grouped_us)} microseconds. Lower is better.`);
}

async function loadBenchmarks() {
  try {
    const data = await readJson("data/benchmarks.json");
    const cases = Array.isArray(data.cases) ? data.cases.filter(validCase) : [];
    if (!cases.length) throw new Error("No complete benchmark cases found.");
    const select = byId("batch-select");
    select.replaceChildren();
    cases.forEach((item, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${item.m} × ${item.n} · ${integer(item.tickets)} ${item.tickets === 1 ? "ticket" : "tickets"}`;
      select.append(option);
    });
    select.value = String(cases.length - 1);
    select.disabled = false;
    select.addEventListener("change", () => showCase(cases[Number(select.value)]));
    const total = Number.isFinite(data.correctness?.tickets) ? data.correctness.tickets : cases.reduce((sum, item) => sum + item.tickets, 0);
    byId("local-check-count").textContent = integer(total);
    const environment = data.environment || {};
    byId("environment-label").textContent = [environment.cpu, environment.rust, environment.profile].filter(Boolean).join(" · ");
    byId("timing-scope").textContent = data.scope || "Kernel timing only. See the raw data for included costs.";
    byId("benchmark-message").hidden = true;
    byId("benchmark-content").hidden = false;
    showCase(cases[cases.length - 1]);
  } catch (error) {
    byId("benchmark-message").textContent = "Could not load CPU timings.";
    const link = document.createElement("a");
    link.href = "data/benchmarks.json";
    link.textContent = " Open raw data";
    byId("benchmark-message").append(link);
    byId("batch-select").replaceChildren(new Option("Measurements unavailable", ""));
  }
}

async function loadConformance() {
  try {
    const data = await readJson("data/conformance.json");
    if (!Array.isArray(data.cases) || !data.cases.length) throw new Error("No conformance cases found.");
    const cases = data.cases;
    const passed = cases.filter(item => item.official_state_equal === true && item.official_hash_equal === true && item.official_plain_verifier_accepted === true).length;
    const controls = cases.filter(item => item.corrupted_raw_byte_rejected === true && item.zero_target_rejected === true).length;
    byId("conformance-count").textContent = `${passed} / ${cases.length}`;
    byId("official-check-copy").textContent = `${passed} of ${cases.length} cases match the official state and hash, and pass its plain-proof verifier.`;
    byId("negative-check-copy").textContent = `${controls} of ${cases.length} cases reject both corrupted raw data and a zero target.`;
    if (passed !== cases.length || controls !== cases.length) {
      byId("conformance-message").hidden = false;
      byId("conformance-message").textContent = "Some verifier checks failed or are missing. See the raw report.";
    }
  } catch (error) {
    byId("conformance-message").hidden = false;
    byId("conformance-message").textContent = "Could not load the verifier report. The text above shows the published result.";
  }
}

byId("copy-commands").addEventListener("click", async () => {
  const button = byId("copy-commands");
  try {
    await navigator.clipboard.writeText(byId("commands").textContent);
    button.textContent = "Copied";
    byId("copy-status").textContent = "Commands copied to clipboard.";
    window.setTimeout(() => { button.textContent = "Copy commands"; }, 2200);
  } catch (error) {
    byId("copy-status").textContent = "Select the terminal text and copy it manually.";
    button.textContent = "Select text to copy";
  }
});

loadBenchmarks();
loadConformance();
