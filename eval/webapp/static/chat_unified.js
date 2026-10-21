// =========================================================================
// Activiity Chat — unified UI (naive + agentic)
// Monochrome interface: model panel, tier sidebar, architecture toggle.
// =========================================================================

const SYNTH_MODELS = [
  { value: "mistralai/ministral-14b-2512", label: "ministral 14b", custom: false },
  { value: "qwen/qwen3-vl-8b-instruct", label: "qwen3 vl 8b instruct", custom: false },
  { value: "google/gemma-3-12b-it", label: "gemma 3 12b it", custom: false },
];

const ROUTER_MODELS = [
  { value: "google/gemma-4-31b-it", label: "gemma 4 31b" },
  { value: "google/gemma-4-26b-a4b-it", label: "gemma 4 26b" },
];

const TIER_LABELS = {
  reference: "Facile",
  medium: "Moyen",
  advanced: "Difficile",
};

function unifiedChat() {
  return {
    // ── Architecture ─────────────────────────────────────────────
    target: "agentic", // 'naive' | 'agentic'

    // ── Models panel ─────────────────────────────────────────────
    panelOpen: false,
    maxSynth: 3,
    synthOptions: SYNTH_MODELS.map(m => ({ ...m })),
    selectedSynth: [],
    customSynthInput: "",
    routerOptions: ROUTER_MODELS,
    routerModel: ROUTER_MODELS[0].value,
    customRouterInput: "",
    apiKeyInput: "",
    sessionKey: "",

    // ── Pairs / tiers ────────────────────────────────────────────
    tiers: [
      { key: "reference", label: TIER_LABELS.reference, open: false, pairs: [] },
      { key: "medium", label: TIER_LABELS.medium, open: false, pairs: [] },
      { key: "advanced", label: TIER_LABELS.advanced, open: false, pairs: [] },
    ],
    pairIndex: new Map(),
    selectedPairId: "",
    question: "",
    groundTruth: "",

    // ── Conversation ─────────────────────────────────────────────
    turns: [],

    // ── History ──────────────────────────────────────────────────
    historyOpen: false,
    history: [],

    error: "",
    loading: false,

    // ── Init ─────────────────────────────────────────────────────
    async init() {
      this.sessionKey = sessionStorage.getItem("openrouter_api_key") || "";
      this._loadHistory();
      await this.loadPairs();
    },

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

    // ── Models panel ─────────────────────────────────────────────

    isSynthChecked(value) {
      return this.selectedSynth.includes(value);
    },

    toggleSynth(value) {
      if (this.isSynthChecked(value)) {
        this.selectedSynth = this.selectedSynth.filter(v => v !== value);
        return;
      }
      if (this.selectedSynth.length >= this.maxSynth) {
        this.toast(`Maximum ${this.maxSynth} modeles de synthese`);
        return;
      }
      this.selectedSynth = [...this.selectedSynth, value];
    },

    addCustomSynth() {
      const v = this.customSynthInput.trim();
      if (!v) return;
      if (!v.includes("/")) {
        this.toast("Format attendu : organisation/modele");
        return;
      }
      if (this.synthOptions.some(m => m.value === v)) {
        this.toast("Ce modele existe deja dans la liste");
        this.customSynthInput = "";
        return;
      }
      this.synthOptions.push({ value: v, label: v, custom: true });
      this.customSynthInput = "";
      if (this.selectedSynth.length < this.maxSynth) {
        this.toggleSynth(v);
      } else {
        this.toast(`Modele ajoute (maximum ${this.maxSynth} atteint)`);
      }
    },

    removeCustomSynth(value) {
      this.synthOptions = this.synthOptions.filter(m => m.value !== value);
      this.selectedSynth = this.selectedSynth.filter(v => v !== value);
    },

    isKnownRouter(value) {
      return this.routerOptions.some(m => m.value === value);
    },

    applyCustomRouter() {
      const v = this.customRouterInput.trim();
      if (!v) return;
      if (!v.includes("/")) {
        this.toast("Format attendu : organisation/modele");
        return;
      }
      this.routerModel = v;
      this.customRouterInput = "";
      this.toast("Modele de routage mis a jour");
    },

    routerShort() {
      const known = this.routerOptions.find(m => m.value === this.routerModel);
      return known ? known.label : this.routerModel;
    },

    saveKey() {
      const k = this.apiKeyInput.trim();
      if (!k) return;
      this.sessionKey = k;
      sessionStorage.setItem("openrouter_api_key", k);
      this.apiKeyInput = "";
      this.toast("Cle OpenRouter enregistree pour cette session");
    },

    maskedKey() {
      const k = this.sessionKey;
      if (k.length < 8) return "***";
      return `${k.slice(0, 4)}...${k.slice(-4)}`;
    },

    // ── Pairs ────────────────────────────────────────────────────

    async loadPairs() {
      try {
        const pairs = await this.api("GET", "/api/chat/pairs");
        this.pairIndex = new Map();
        for (const t of this.tiers) t.pairs = [];
        for (const p of pairs) {
          const group = this.tiers.find(t => t.key === p.tier);
          if (!group) continue;
          group.pairs.push(p);
          this.pairIndex.set(`${group.key}/${p.id}`, p);
        }
      } catch (e) {
        this.toast("Chargement des paires impossible : " + e.message);
      }
    },

    selectPair(tierKey, pairId) {
      const key = `${tierKey}/${pairId}`;
      const p = this.pairIndex.get(key);
      if (!p) return;
      this.selectedPairId = key;
      this.question = p.question;
      this.groundTruth = p.ground_truth || "";
      this.error = "";
    },

    // ── Architecture toggle ──────────────────────────────────────

    setTarget(t) {
      if (this.target === t) return;
      this.target = t;
      this.toast(t === "naive" ? "Architecture : RAG Naif" : "Architecture : RAG Agentique");
    },

    targetLabel() {
      return this.target === "naive" ? "RAG Naif" : "RAG Agentique";
    },

    // ── Ask ──────────────────────────────────────────────────────

    _slot(modelValue) {
      return {
        model: modelValue,
        modelLabel: modelValue,
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
    },

    async ask() {
      const q = this.question.trim();
      if (!q || this.loading) return;

      if (this.selectedSynth.length === 0) {
        this.error = "Aucun modele de synthese selectionne. Ouvrez le panneau Modeles.";
        return;
      }

      this.error = "";

      const turn = {
        question: q,
        pairId: this.selectedPairId || "",
        ground_truth: this.groundTruth || "",
        results: this.selectedSynth.map(m => this._slot(m)),
      };
      this.turns.push(turn);
      const ti = this.turns.length - 1;

      this.history.unshift({
        ts: Date.now(),
        target: this.target,
        routerModel: this.routerModel,
        synthModels: [...this.selectedSynth],
        pairId: turn.pairId,
        question: q,
        results: turn.results,
      });
      if (this.history.length > 50) this.history.pop();

      const questionText = q;

      this.question = "";
      this.groundTruth = "";
      this.selectedPairId = "";

      this.loading = true;
      this._scrollDown();

      try {
        if (this.target === "naive") {
          await Promise.allSettled(
            this.selectedSynth.map((modelValue, ri) =>
              this._askNaive(ti, ri, questionText, modelValue)
            )
          );
        } else if (this.selectedSynth.length === 1) {
          await this._askAgenticSingle(ti, 0, questionText, this.selectedSynth[0]);
        } else {
          await this._askAgenticMulti(ti, questionText);
        }
      } catch (e) {
        this.error = e.message;
        for (let ri = 0; ri < turn.results.length; ri++) {
          const r = turn.results[ri];
          if (r.loading) {
            r.loading = false;
            r.error = r.error || e.message;
          }
        }
      }

      this.loading = false;
    },

    async _askNaive(ti, ri, question, modelValue) {
      const t0 = performance.now();
      try {
        const res = await this.api("POST", "/api/chat/ask", {
          question,
          target: "naive",
          model: modelValue,
          use_reranker: false,
          compute_ragas: false,
          openrouter_api_key: this.sessionKey || null,
        });
        const latency = ((performance.now() - t0) / 1000).toFixed(1);
        const r = this.turns[ti].results[ri];
        if (res.error) {
          r.loading = false;
          r.error = res.error;
          r.latency = latency;
          return;
        }
        r.loading = false;
        r.answer = res.answer;
        r.contexts = res.contexts || [];
        r.latency = latency;
        this._scrollDown();
        this._persistHistory();
        this._fireRagas(ti, ri, question);
      } catch (e) {
        const r = this.turns[ti].results[ri];
        r.loading = false;
        r.error = e.message;
        r.latency = ((performance.now() - t0) / 1000).toFixed(1);
      }
    },

    async _askAgenticSingle(ti, ri, question, modelValue) {
      const t0 = performance.now();
      try {
        const res = await this.api("POST", "/api/chat/ask", {
          question,
          target: "agentic",
          model: this.routerModel,
          synth_model: modelValue,
          use_reranker: true,
          compute_ragas: false,
          openrouter_api_key: this.sessionKey || null,
        });
        const latency = ((performance.now() - t0) / 1000).toFixed(1);
        const r = this.turns[ti].results[ri];
        if (res.error) {
          r.loading = false;
          r.error = res.error;
          r.latency = latency;
          return;
        }
        r.loading = false;
        r.answer = res.answer;
        r.contexts = res.contexts || [];
        r.tools_called = res.tools_called || [];
        r.units_called = res.units_called || [];
        r.iterations = res.iterations || 0;
        r.latency = latency;
        this._scrollDown();
        this._persistHistory();
        this._fireRagas(ti, ri, question);
      } catch (e) {
        const r = this.turns[ti].results[ri];
        r.loading = false;
        r.error = e.message;
        r.latency = ((performance.now() - t0) / 1000).toFixed(1);
      }
    },

    async _askAgenticMulti(ti, question) {
      const n = this.turns[ti].results.length;
      try {
        const res = await this.api("POST", "/api/chat/ask_multi", {
          question,
          target: "agentic",
          model: this.routerModel,
          synth_models: [...this.selectedSynth],
          use_reranker: true,
          openrouter_api_key: this.sessionKey || null,
        });

        if (res.error) {
          for (let ri = 0; ri < n; ri++) {
            const r = this.turns[ti].results[ri];
            r.loading = false;
            r.error = res.error;
          }
          this.error = res.error;
          return;
        }

        const byModel = new Map((res.items || []).map(i => [i.model, i]));
        const ragasPromises = [];
        for (let ri = 0; ri < n; ri++) {
          const r = this.turns[ti].results[ri];
          const item = byModel.get(r.model);
          if (!item) {
            r.loading = false;
            r.error = "Reponse manquante";
            continue;
          }
          r.loading = false;
          r.tools_called = res.tools_called || [];
          r.units_called = res.units_called || [];
          r.iterations = res.iterations || 0;
          r.contexts = item.contexts || [];
          if (item.error) {
            r.error = item.error;
            continue;
          }
          r.answer = item.answer;
          this._persistHistory();
          ragasPromises.push(this._fireRagas(ti, ri, question));
        }
        this._scrollDown();
        await Promise.allSettled(ragasPromises);
      } catch (e) {
        for (let ri = 0; ri < n; ri++) {
          const r = this.turns[ti].results[ri];
          r.loading = false;
          if (!r.error) r.error = e.message;
        }
      }
    },

    // ── RAGAS ────────────────────────────────────────────────────

    async _fireRagas(ti, ri, questionText) {
      const r = this.turns[ti].results[ri];
      if (!r || !r.answer) return;
      r.ragasLoading = true;

      try {
        const res = await this.api("POST", "/api/chat/ragas", {
          question: questionText,
          answer: r.answer,
          contexts: r.contexts || [],
          ground_truth: this.turns[ti].ground_truth || null,
          openrouter_api_key: this.sessionKey || null,
        });
        if (res.error) {
          r.ragasError = res.error;
        } else {
          r.ragas = res.ragas || null;
          if (!r.ragas) r.ragasError = "Scores indisponibles";
        }
      } catch (e) {
        r.ragasError = e.message;
      }
      r.ragasLoading = false;
      this._persistHistory();
    },

    // ── History ──────────────────────────────────────────────────

    toggleHistory() {
      this.historyOpen = !this.historyOpen;
    },

    closeHistory() {
      this.historyOpen = false;
    },

    _loadHistory() {
      try {
        const raw = localStorage.getItem("activiity_chat_history");
        if (raw) {
          const data = JSON.parse(raw);
          if (Array.isArray(data)) this.history = data;
        }
      } catch (e) { /* ignore corrupted storage */ }
    },

    _persistHistory() {
      try {
        const data = this.history.slice(0, 50).map(h => ({
          ts: h.ts,
          target: h.target,
          routerModel: h.routerModel,
          synthModels: h.synthModels,
          pairId: h.pairId,
          question: h.question,
          results: (h.results || []).map(r => ({
            model: r.model,
            modelLabel: r.modelLabel,
            error: r.error || null,
            latency: r.latency ?? null,
            ragas: r.ragas || null,
            ragasError: r.ragasError || null,
          })),
        }));
        localStorage.setItem("activiity_chat_history", JSON.stringify(data));
      } catch (e) { /* quota or serialization issue: ignore */ }
    },

    fmtTime(ts) {
      const d = new Date(ts);
      return d.toLocaleDateString("fr-FR") + " " +
        d.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
    },

    reuseQuestion(h) {
      this.question = h.question;
      this.historyOpen = false;
      this.toast("Question rechargee dans le champ de saisie");
    },

    clearHistory() {
      this.history = [];
      localStorage.removeItem("activiity_chat_history");
      this.toast("Historique efface");
    },

    // ── Helpers ──────────────────────────────────────────────────

    statusText(r) {
      if (r.loading) return "Traitement...";
      if (r.error) return "Erreur";
      if (r.latency !== null) return `${r.latency} s`;
      return "";
    },

    fmt(v) {
      if (v === null || v === undefined) return "-";
      return (v * 100).toFixed(1) + "%";
    },

    _scrollDown() {
      this.$nextTick(() => {
        const el = this.$refs.messages;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    toast(msg) {
      const el = document.createElement("div");
      el.textContent = msg;
      el.className = "toast";
      this.$refs.toasts.appendChild(el);
      setTimeout(() => el.remove(), 2600);
    },
  };
}
