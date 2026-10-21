// =========================================================================
// Chat page logic — multi-model comparison with RAGAS metrics
// Shared by both Agentic & Naive UIs.
// =========================================================================

const AVAILABLE_MODELS = [
  { value: "mistralai/ministral-14b-2512",     label: "ministral 14b" },
  { value: "qwen/qwen3-vl-8b-instruct",         label: "qwen3 vl 8b instruct" },
  { value: "google/gemma-3-12b-it",       label: "gemma 3 12b it" },
  
  
];

const MODEL_COLORS = [
  "#6366f1",  // indigo
  "#8b5cf6",  // violet
  "#06b6d4",  // cyan
  "#f59e0b",  // amber
];

function chatPage(defaultTarget) {
  const isAgentic = defaultTarget === "agentic";
  return {
    target: defaultTarget,
    isAgentic,

    // Multi-model selection (up to 3)
    availableModels: AVAILABLE_MODELS,
    selectedModels: [
      "",                          // empty = not selected
      "",
      "",
    ],

    // Agentic-specific
    synth_model: "qwen/qwen3-14b",
    use_reranker: false,
    agenticRouterOptions: [
       
      { value: "google/gemma-4-31b-it", label: "Gemma 4 31B" },
      { value: "google/gemma-4-26b-a4b-it", label: "Gemma 4 26B" },
         ],
    agenticSynthOptions: AVAILABLE_MODELS,
    routerModel: "google/gemma-4-31b-it",

    // Pairs
    pairs: [],
    selectedPair: "",
    question: "",
    ground_truth: "",

    // Results per model slot: [{model, answer, loading, error, tools_called, iterations, ragas, latency}]
    results: [null, null, null],

    // Global
    error: "",
    loading: false,
    modelsSaved: false,

    // Config
    cfg: {},
    newKeys: { openrouter_api_key: "" },
    sessionKey: "",
    openrouterStatus: null,

    // ── Computed ──────────────────────────────────────────────────

    get activeModelCount() {
      return this.selectedModels.filter(m => m !== "").length;
    },

    get activeModels() {
      return this.selectedModels
        .map((value, idx) => ({ value, idx }))
        .filter(m => m.value !== "");
    },

    get gridCols() {
      const n = this.activeModelCount;
      if (n <= 1) return "cols-1";
      if (n === 2) return "cols-2";
      return "cols-3";
    },

    get hasAnyResult() {
      return this.results.some(r => r !== null);
    },

    get hasSessionKey() {
      return !!this.sessionKey;
    },

    get sessionKeyMasked() {
      if (!this.sessionKey) return "non défini";
      const k = this.sessionKey;
      if (k.length < 8) return "***";
      return `${k.slice(0, 4)}...${k.slice(-4)}`;
    },

    modelLabel(value) {
      const m = AVAILABLE_MODELS.find(m => m.value === value);
      return m ? m.label : value;
    },

    modelColor(idx) {
      return MODEL_COLORS[idx % MODEL_COLORS.length];
    },

    // ── Init ─────────────────────────────────────────────────────

    async init() {
      // Load existing session key (don't remove on refresh - user should stay logged in)
      this.sessionKey = sessionStorage.getItem("openrouter_api_key") || "";
      await this.loadConfig();
      await this.loadChatPairs();
    },

    // ── API helper ───────────────────────────────────────────────

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

    // ── Config ───────────────────────────────────────────────────

    async loadConfig() {
      this.cfg = await this.api("GET", "/api/config");
    },

    async saveOpenRouterKey() {
      if (!this.newKeys.openrouter_api_key) return;
      this.sessionKey = this.newKeys.openrouter_api_key;
      sessionStorage.setItem("openrouter_api_key", this.sessionKey);
      this.newKeys.openrouter_api_key = "";
      this.toast("Clé OpenRouter enregistrée pour cette session");
    },

    async testOpenRouter() {
      const key = this.newKeys.openrouter_api_key || this.sessionKey;
      if (!key) {
        this.openrouterStatus = { ok: false, msg: "✗ Aucune clé OpenRouter" };
        return;
      }
      if (this.newKeys.openrouter_api_key) {
        this.sessionKey = this.newKeys.openrouter_api_key;
        sessionStorage.setItem("openrouter_api_key", this.sessionKey);
      }
      try {
        const r = await this.api("POST", "/api/config/test-openrouter", {
          openrouter_api_key: key,
        });
        this.openrouterStatus = r.ok
          ? { ok: true, msg: `✓ Connecté — ${r.n_models} modèles disponibles` }
          : { ok: false, msg: `✗ Échec (HTTP ${r.status})` };
      } catch (e) {
        this.openrouterStatus = { ok: false, msg: `✗ Erreur : ${e.message}` };
      }
    },

    // ── Pairs ────────────────────────────────────────────────────

    async loadChatPairs() {
      this.pairs = await this.api("GET", "/api/chat/pairs");
      this.selectedPair = "";
      this.toast(`${this.pairs.length} paires de référence chargées`);
    },

    selectChatPair() {
      const p = this.pairs.find(x => x.id === this.selectedPair);
      if (p) {
        this.question = p.question;
        this.ground_truth = p.ground_truth;
        this.results = [null, null, null];
        this.error = "";
      }
    },

    // ── Model save ───────────────────────────────────────────────

    saveModels() {
      if (this.activeModelCount === 0) {
        this.toast("Sélectionnez au moins un modèle");
        return;
      }
      this.modelsSaved = true;
      this.toast(`${this.activeModelCount} modèle(s) enregistré(s)`);
    },

    resetModels() {
      this.modelsSaved = false;
      this.results = [null, null, null];
    },

    // ── Ask (multi-model parallel, two-phase) ──────────────────────

    async askChat() {
      if (!this.question) return;
      if (this.activeModelCount === 0) {
        this.error = "Veuillez selectionner au moins un modele";
        return;
      }

      this.loading = true;
      this.error = "";
      this.results = [null, null, null];

      const questionText = this.question;
      const groundTruth = this.ground_truth || null;

      // Agentic: single retrieval, multi-synth (shared context)
      if (this.isAgentic && this.activeModelCount > 1) {
        this.activeModels.forEach(({ value: modelValue, idx }) => {
          this.results[idx] = {
            model: modelValue,
            modelLabel: this.modelLabel(modelValue),
            answer: "",
            loading: true,
            error: null,
            tools_called: [],
            units_called: [],
            iterations: 0,
            contexts: [],
            ragas: null,
            ragasLoading: false,
            ragasError: null,
            latency: null,
          };
        });

        const t0 = performance.now();
        try {
          const res = await this.api("POST", "/api/chat/ask_multi", {
            question: questionText,
            target: this.target,
            use_reranker: this.use_reranker,
            model: this.routerModel,
            synth_models: this.activeModels.map(m => m.value),
            openrouter_api_key: this.sessionKey || null,
          });

          const latency = ((performance.now() - t0) / 1000).toFixed(1);
          if (res.error) {
            this.error = res.error;
            this.activeModels.forEach(({ idx }) => {
              this.results[idx] = {
                ...this.results[idx],
                loading: false,
                error: res.error,
                latency,
              };
            });
          } else {
            const items = res.items || [];
            const byModel = new Map(items.map(i => [i.model, i]));
            this.activeModels.forEach(({ value: modelValue, idx }) => {
              const item = byModel.get(modelValue);
              if (!item) {
                this.results[idx] = {
                  ...this.results[idx],
                  loading: false,
                  error: "Reponse manquante",
                  latency,
                };
                return;
              }
              if (item.error) {
                this.results[idx] = {
                  ...this.results[idx],
                  loading: false,
                  error: item.error,
                  latency,
                };
                return;
              }
              const updated = {
                ...this.results[idx],
                loading: false,
                answer: item.answer,
                tools_called: res.tools_called || [],
                units_called: res.units_called || [],
                iterations: res.iterations || 0,
                contexts: item.contexts || [],
                ragas: null,
                ragasLoading: true,
                latency,
              };
              this.results.splice(idx, 1, updated);
              this._fireRagasForIndex(idx, questionText, groundTruth);
            });
          }
        } catch (e) {
          const latency = ((performance.now() - t0) / 1000).toFixed(1);
          this.error = e.message;
          this.activeModels.forEach(({ idx }) => {
            this.results[idx] = {
              ...this.results[idx],
              loading: false,
              error: e.message,
              latency,
            };
          });
        }

        this.loading = false;
        return;
      }

      // ─── Phase 1: Get answers (NO RAGAS) ───────────────────────
      const promises = this.activeModels.map(({ value: modelValue, idx }) => {
        // Initialize result slot immediately
        this.results[idx] = {
          model: modelValue,
          modelLabel: this.modelLabel(modelValue),
          answer: "",
          loading: true,
          error: null,
          tools_called: [],
          units_called: [],
          iterations: 0,
          contexts: [],
          ragas: null,
          ragasLoading: false,
          ragasError: null,
          latency: null,
        };

        const body = {
          question: questionText,
          target: this.target,
          use_reranker: this.use_reranker,
          model: this.isAgentic ? this.routerModel : modelValue,
          synth_model: this.isAgentic ? modelValue : null,
          compute_ragas: false,          // ← don't block for RAGAS
          ground_truth: groundTruth,
          openrouter_api_key: this.sessionKey || null,
        };

        const t0 = performance.now();

        return this.api("POST", "/api/chat/ask", body)
          .then(res => {
            const latency = ((performance.now() - t0) / 1000).toFixed(1);
            if (res.error) {
              this.results[idx] = {
                ...this.results[idx],
                loading: false,
                error: res.error,
                latency,
              };
            } else {
              const updated = {
                ...this.results[idx],
                loading: false,
                answer: res.answer,
                tools_called: res.tools_called || [],
                units_called: res.units_called || [],
                iterations: res.iterations || 0,
                contexts: res.contexts || [],
                ragas: null,
                ragasLoading: true,   // ← start RAGAS per model
                latency,
              };
              this.results.splice(idx, 1, updated);
              this._fireRagasForIndex(idx, questionText, groundTruth);
            }
          })
          .catch(e => {
            const latency = ((performance.now() - t0) / 1000).toFixed(1);
            this.results[idx] = {
              ...this.results[idx],
              loading: false,
              error: e.message,
              latency,
            };
          });
      });

      await Promise.allSettled(promises);
      this.loading = false;

      // No global wait; each model triggers its own RAGAS.
    },

    // ── Phase 2: RAGAS per model as soon as its answer arrives ────

    async _fireRagasForIndex(idx, questionText, groundTruth) {
      const r = this.results[idx];
      if (!r || r.error || !r.answer) {
        if (r) this.results.splice(idx, 1, { ...r, ragasLoading: false });
        return;
      }

      const body = {
        question: questionText,
        answer: r.answer,
        contexts: r.contexts || [],
        ground_truth: groundTruth,
        openrouter_api_key: this.sessionKey || null,
      };

      try {
        const res = await this.api("POST", "/api/chat/ragas", body);
        if (res.error) {
          this.results.splice(idx, 1, {
            ...this.results[idx],
            ragasLoading: false,
            ragasError: res.error,
          });
        } else {
          this.results.splice(idx, 1, {
            ...this.results[idx],
            ragasLoading: false,
            ragas: res.ragas || null,
          });
        }
      } catch (e) {
        this.results.splice(idx, 1, {
          ...this.results[idx],
          ragasLoading: false,
          ragasError: e.message,
        });
      }
    },

    // ── RAGAS formatting helpers ─────────────────────────────────

    formatMetric(val) {
      if (val === null || val === undefined) return "—";
      return (val * 100).toFixed(1) + "%";
    },

    metricClass(val) {
      if (val === null || val === undefined) return "metric-na";
      if (val >= 0.7) return "metric-good";
      if (val >= 0.4) return "metric-mid";
      return "metric-bad";
    },

    // ── Toast ────────────────────────────────────────────────────

    toast(msg) {
      const el = document.createElement("div");
      el.textContent = msg;
      el.className = "toast";
      document.body.appendChild(el);
      setTimeout(() => el.remove(), 2500);
    },
  };
}