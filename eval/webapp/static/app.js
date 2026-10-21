// Alpine.js root component for the Eval Control Panel.
// Keeps state, talks to the FastAPI backend, renders Chart.js charts on demand.

function app() {
  return {
    // ── state ─────────────────────────────────────────────────────────
    tab: "settings",
    tabs: [
      { id: "settings",   label: "⚙ Settings" },
      { id: "run",        label: "▶ Run" },
      { id: "dashboard",  label: "📊 Dashboard" },
      { id: "history",    label: "🗂 History" },
    ],

    cfg: {},
    newKeys: { openrouter_api_key: "", openai_api_key: "" },
    openrouterStatus: null,
    orModels: [
      "anthropic/claude-sonnet-4-6",
      "anthropic/claude-opus-4-5",
      "openai/gpt-4o",
      "openai/gpt-4o-mini-2024-07-18",
      "google/gemini-2.0-flash-lite-001",
      "mistralai/mistral-large",
      "qwen/qwen-2.5-72b-instruct",
      "google/gemini-2.0-flash-001",
    ],

    // IMPORTANT: mock defaults to false. Mock mode fakes perfect answers
    // (returns ground_truth verbatim) and should ONLY be used to smoke-test
    // the evaluator pipeline itself. It is NOT a real evaluation.
    run: { target: "naive", tier: "reference", workers: 4, limit: null,
           mock: false, ragas: false, no_bertscore: true, no_sbert: true },
    currentRun: null,
    consoleBuffer: "",
    eventSource: null,

    runs: [],
    selectedRunId: "",
    selectedRun: null,
    selectedReport: null,
    thresholds: {},

    charts: {},   // Chart.js instances

    // ── lifecycle ─────────────────────────────────────────────────────
    async init() {
      await this.loadConfig();
      await this.loadThresholds();
      await this.refreshRuns();
    },

    // ── API helpers ───────────────────────────────────────────────────
    async api(method, path, body) {
      const res = await fetch(path, {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const ct = res.headers.get("content-type") || "";
      return ct.includes("json") ? res.json() : res.text();
    },

    // ── Settings ──────────────────────────────────────────────────────
    async loadConfig() {
      this.cfg = await this.api("GET", "/api/config");
    },
    async reload() { await this.loadConfig(); },

    async saveConfig() {
      const body = { ...this.cfg };
      // only send raw keys if the user typed new ones
      if (this.newKeys.openrouter_api_key) body.openrouter_api_key = this.newKeys.openrouter_api_key;
      else delete body.openrouter_api_key;
      if (this.newKeys.openai_api_key) body.openai_api_key = this.newKeys.openai_api_key;
      else delete body.openai_api_key;

      await this.api("PUT", "/api/config", body);
      this.newKeys = { openrouter_api_key: "", openai_api_key: "" };
      await this.loadConfig();
      this.toast("Configuration saved.");
    },

    async testOpenRouter() {
      // first persist the typed key if present, so the server can use it
      if (this.newKeys.openrouter_api_key) {
        await this.api("PUT", "/api/config", { openrouter_api_key: this.newKeys.openrouter_api_key });
      }
      try {
        const r = await this.api("POST", "/api/config/test-openrouter");
        this.openrouterStatus = r.ok
          ? { ok: true,  msg: `✓ OK — ${r.n_models} models available.` }
          : { ok: false, msg: `✗ failed (HTTP ${r.status})` };
      } catch (e) {
        this.openrouterStatus = { ok: false, msg: `✗ ${e.message}` };
      }
    },

    async testOllama() {
      try {
        const r = await this.api("POST", "/api/config/test-ollama");
        alert(r.ok ? `✓ Ollama OK — ${r.n_models} models:\n\n${r.models.join("\n")}`
                   : `✗ Ollama unreachable: ${r.error || r.status}`);
      } catch (e) { alert(`Error: ${e.message}`); }
    },

    async loadThresholds() {
      try { this.thresholds = await this.api("GET", "/api/thresholds"); }
      catch (_) { this.thresholds = {}; }
    },

    // ── Runs ──────────────────────────────────────────────────────────
    async refreshRuns() {
      this.runs = await this.api("GET", "/api/runs");
    },

    async startRun() {
      // Preflight check for real (non-mock) runs — catch missing deps/keys early.
      if (!this.run.mock) {
        try {
          const pf = await this.api("GET", "/api/preflight");
          if (!pf.ok) {
            alert("Cannot start a real run — fix these first:\n\n• " +
                  pf.issues.join("\n• "));
            return;
          }
        } catch (e) { /* preflight best-effort */ }
      }
      this.consoleBuffer = "";
      this.currentRun = await this.api("POST", "/api/runs", this.run);
      await this.refreshRuns();
      this.streamRun(this.currentRun.id);
    },

    streamRun(runId) {
      if (this.eventSource) this.eventSource.close();
      const es = new EventSource(`/api/runs/${runId}/stream`);
      this.eventSource = es;

      es.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.line !== undefined) {
            this.consoleBuffer += data.line + "\n";
            this.$nextTick(() => {
              const c = this.$refs.console;
              if (c) c.scrollTop = c.scrollHeight;
            });
          }
        } catch (_) {}
      };
      es.addEventListener("done", (ev) => {
        try { this.currentRun = JSON.parse(ev.data); } catch (_) {}
        es.close();
        this.refreshRuns();
      });
      es.onerror = () => { es.close(); };
    },

    async loadRun(runId) {
      if (!runId) return;
      this.selectedRunId = runId;
      const data = await this.api("GET", `/api/runs/${runId}`);
      this.selectedRun = data.meta;
      this.selectedReport = data.report || null;
      this.$nextTick(() => this.renderCharts());
    },

    async cancelRun(runId) {
      if (!confirm("Kill the running subprocess?")) return;
      await this.api("POST", `/api/runs/${runId}/cancel`);
      await this.refreshRuns();
    },

    async deleteRun(runId) {
      if (!confirm("Delete this run?")) return;
      await this.api("DELETE", `/api/runs/${runId}`);
      if (this.selectedRunId === runId) {
        this.selectedRunId = ""; this.selectedReport = null; this.selectedRun = null;
      }
      await this.refreshRuns();
    },

    // ── KPIs / rendering ──────────────────────────────────────────────
    statusColor(s) {
      return {
        pending:   "bg-slate-500 text-white",
        running:   "bg-amber-500 text-white",
        done:      "bg-emerald-600 text-white",
        failed:    "bg-rose-600 text-white",
        cancelled: "bg-slate-400 text-white",
      }[s] || "bg-slate-200";
    },

    kpi(name, value, threshold, higherIsBetter = true, fmt = 3) {
      if (value === null || value === undefined || isNaN(value)) {
        return { name, value: "—", threshold: threshold?.toFixed?.(2) ?? "—", pass: null, badge: "" };
      }
      const pass = threshold === undefined ? null
                 : higherIsBetter ? value >= threshold : value <= threshold;
      return {
        name,
        value: typeof value === "number" ? value.toFixed(fmt) : value,
        threshold: threshold !== undefined ? threshold.toFixed(2) : "—",
        pass,
        badge: pass === null ? "" : pass ? "PASS" : "FAIL",
      };
    },

    kpisForTier(tier, rep) {
      const th = this.thresholds[tier] || {};
      const k = [];
      k.push(this.kpi("Accuracy",         rep.routing?.accuracy,        th.accuracy));
      k.push(this.kpi("Macro-F1",         rep.routing?.macro_f1,        th.macro_f1));
      k.push(this.kpi("Macro-Precision",  rep.routing?.macro_precision));
      k.push(this.kpi("Macro-Recall",     rep.routing?.macro_recall));
      k.push(this.kpi("Refusal F1",       rep.refusal?.f1));
      k.push(this.kpi("Refusal Precision",rep.refusal?.precision,       th.refusal_precision));
      k.push(this.kpi("Refusal Recall",   rep.refusal?.recall,          th.refusal_recall));
      k.push(this.kpi("Citation Recall",  rep.retrieval?.citation_recall, th.citation_recall));
      k.push(this.kpi("MRR",              rep.retrieval?.mrr));
      k.push(this.kpi("nDCG@k",           rep.retrieval?.ndcg_at_k));
      if (rep.similarity) {
        k.push(this.kpi("Token-F1",       rep.similarity.token_f1,      th.token_f1));
        k.push(this.kpi("ROUGE-L F1",     rep.similarity.rouge_l_f1,    th.rouge_l_f1));
        if (rep.similarity.sbert_cosine)   k.push(this.kpi("SBERT cos",   rep.similarity.sbert_cosine, th.sbert_cosine));
        if (rep.similarity.bertscore_f1 !== null && rep.similarity.bertscore_f1 !== undefined)
          k.push(this.kpi("BERTScore",     rep.similarity.bertscore_f1, th.bertscore_f1));
      }
      if (rep.ragas) {
        k.push(this.kpi("Faithfulness",    rep.ragas.faithfulness,       th.faithfulness));
        k.push(this.kpi("Answer Rel.",     rep.ragas.answer_relevancy,   th.answer_relevancy));
        k.push(this.kpi("Ctx Precision",   rep.ragas.context_precision,  th.context_precision));
        k.push(this.kpi("Ctx Recall",      rep.ragas.context_recall,     th.context_recall));
      }
      k.push(this.kpi("Latency p95 (s)",   rep.system?.latency_p95_s,   th.p95_latency_s, false, 2));
      k.push(this.kpi("Cost (USD)",        rep.system?.cost_usd, undefined, true, 4));
      return k;
    },

    renderCharts() {
      // destroy old charts first
      Object.values(this.charts).forEach(c => c.destroy());
      this.charts = {};
      if (!this.selectedReport) return;

      for (const [tier, rep] of Object.entries(this.selectedReport)) {
        // per-class F1 bar chart
        const f1Ctx = document.getElementById(`chart-f1-${tier}`);
        if (f1Ctx && rep.routing?.per_class) {
          const entries = Object.entries(rep.routing.per_class)
            .filter(([, v]) => v.support > 0);
          this.charts[`f1-${tier}`] = new Chart(f1Ctx, {
            type: "bar",
            data: {
              labels: entries.map(([k]) => k),
              datasets: [
                { label: "Precision", data: entries.map(([, v]) => v.precision), backgroundColor: "#60a5fa" },
                { label: "Recall",    data: entries.map(([, v]) => v.recall),    backgroundColor: "#34d399" },
                { label: "F1",        data: entries.map(([, v]) => v.f1),        backgroundColor: "#f59e0b" },
              ],
            },
            options: {
              responsive: true, maintainAspectRatio: false,
              scales: { y: { beginAtZero: true, max: 1 } },
              plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
            },
          });
        }

        // metric radar
        const radarCtx = document.getElementById(`chart-radar-${tier}`);
        if (radarCtx) {
          const labels = [], values = [];
          const push = (k, v) => { if (v != null && !isNaN(v)) { labels.push(k); values.push(v); } };
          push("Accuracy",        rep.routing?.accuracy);
          push("Macro-F1",        rep.routing?.macro_f1);
          push("Refusal F1",      rep.refusal?.f1);
          push("Citation R",      rep.retrieval?.citation_recall);
          push("MRR",             rep.retrieval?.mrr);
          push("nDCG@k",          rep.retrieval?.ndcg_at_k);
          if (rep.similarity) { push("Token-F1", rep.similarity.token_f1); push("ROUGE-L", rep.similarity.rouge_l_f1); }
          if (rep.ragas) { push("Faithful.", rep.ragas.faithfulness); push("Answer Rel.", rep.ragas.answer_relevancy); }

          this.charts[`radar-${tier}`] = new Chart(radarCtx, {
            type: "radar",
            data: { labels, datasets: [{ label: tier, data: values, backgroundColor: "rgba(59,130,246,0.25)", borderColor: "#2563eb", pointBackgroundColor: "#2563eb" }] },
            options: {
              responsive: true, maintainAspectRatio: false,
              scales: { r: { min: 0, max: 1, ticks: { stepSize: 0.2 } } },
              plugins: { legend: { display: false } },
            },
          });
        }
      }
    },

    toast(msg) {
      const el = document.createElement("div");
      el.textContent = msg;
      el.className = "fixed bottom-4 right-4 bg-slate-900 text-white px-4 py-2 rounded-lg shadow-lg text-sm z-50";
      document.body.appendChild(el);
      setTimeout(() => el.remove(), 2400);
    },
  };
}
