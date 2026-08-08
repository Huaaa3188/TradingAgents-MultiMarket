"use strict";

/* TradingAgents Web UI: Awwwards-Grade Interactive Quantum Matrix & SSE Engine. */

const state = {
  config: null,
  source: null,
  lastText: {}, // section key -> last markdown text rendered
  activeNodes: new Set(),
};

const $ = (id) => document.getElementById(id);

function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

// ---------------------------------------------------------------- Lucide Helpers

function updateLucideIcons() {
  if (window.lucide && typeof window.lucide.createIcons === "function") {
    window.lucide.createIcons();
  }
}

// ---------------------------------------------------------------- Interactive Canvas Engine

function initParticleCanvas() {
  const canvas = $("bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener("resize", () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const particles = [];
  const particleCount = Math.floor(Math.min(width, height) / 18);
  const mouse = { x: width / 2, y: height / 2, radius: 140 };

  window.addEventListener("mousemove", (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  });

  for (let i = 0; i < particleCount; i++) {
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.6,
      vy: (Math.random() - 0.5) * 0.6,
      radius: Math.random() * 1.8 + 0.6,
      alpha: Math.random() * 0.4 + 0.1,
    });
  }

  function render() {
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < particleCount; i++) {
      const p = particles[i];

      // Physics move
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0) p.x = width;
      if (p.x > width) p.x = 0;
      if (p.y < 0) p.y = height;
      if (p.y > height) p.y = 0;

      // Mouse attraction / repulsion
      const dx = mouse.x - p.x;
      const dy = mouse.y - p.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < mouse.radius) {
        const force = (mouse.radius - dist) / mouse.radius;
        p.x -= (dx / dist) * force * 1.5;
        p.y -= (dy / dist) * force * 1.5;
      }

      // Draw particle
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 242, 254, ${p.alpha})`;
      ctx.shadowBlur = 8;
      ctx.shadowColor = "#00f2fe";
      ctx.fill();

      // Draw lines between nearby particles
      for (let j = i + 1; j < particleCount; j++) {
        const p2 = particles[j];
        const ldx = p.x - p2.x;
        const ldy = p.y - p2.y;
        const ldist = Math.sqrt(ldx * ldx + ldy * ldy);
        if (ldist < 100) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = `rgba(0, 242, 254, ${0.15 * (1 - ldist / 100)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(render);
  }

  render();
}

// ---------------------------------------------------------------- Custom Cursor Engine

function initCustomCursor() {
  const dot = $("cursor-dot");
  const follower = $("cursor-follower");
  if (!dot || !follower) return;

  let fx = 0, fy = 0;
  let mx = 0, my = 0;

  window.addEventListener("mousemove", (e) => {
    mx = e.clientX;
    my = e.clientY;
    dot.style.left = `${mx}px`;
    dot.style.top = `${my}px`;
  });

  function animate() {
    fx += (mx - fx) * 0.15;
    fy += (my - fy) * 0.15;
    follower.style.left = `${fx}px`;
    follower.style.top = `${fy}px`;
    requestAnimationFrame(animate);
  }

  animate();
}

// ---------------------------------------------------------------- init & Config

async function init() {
  initParticleCanvas();
  initCustomCursor();
  
  try {
    const res = await fetch("/api/config");
    state.config = await res.json();
    fillForm(state.config);
    bindEvents();
    if ($("analysis-date") && state.config.defaults) {
      $("analysis-date").value = state.config.defaults.today;
    }
  } catch (err) {
    console.error("Failed to load initial config", err);
  }

  updateLucideIcons();
}

function fillForm(config) {
  const providerSel = $("provider");
  if (!providerSel) return;
  providerSel.innerHTML = "";
  for (const p of config.providers) {
    const opt = document.createElement("option");
    opt.value = p.key;
    opt.textContent = p.name;
    providerSel.appendChild(opt);
  }
  const def = config.defaults;
  const idx = config.providers.findIndex((p) => p.key === def.llm_provider);
  providerSel.selectedIndex = idx >= 0 ? idx : 0;
  if ($("debate-rounds")) $("debate-rounds").value = def.max_debate_rounds;
  if ($("risk-rounds")) $("risk-rounds").value = def.max_risk_discuss_rounds;
  if ($("output-language")) $("output-language").value = def.output_language;
  if ($("checkpoint")) $("checkpoint").checked = !!def.checkpoint_enabled;
  refreshModels();
  refreshKeyStatus();
}

function refreshModels() {
  const provider = $("provider").value;
  const models = state.config.models[provider] || {};
  fillModelSelect("deep-model-wrap", "deep_think_llm", models.deep, state.config.defaults.deep_think_llm);
  fillModelSelect("quick-model-wrap", "quick_think_llm", models.quick, state.config.defaults.quick_think_llm);
}

function fillModelSelect(wrapId, name, options, fallback) {
  const wrap = $(wrapId);
  if (!wrap) return;
  wrap.innerHTML = "";
  if (options && options.length) {
    const sel = document.createElement("select");
    sel.name = name;
    for (const m of options) {
      const opt = document.createElement("option");
      opt.value = m.value;
      opt.textContent = m.display;
      sel.appendChild(opt);
    }
    if (fallback && [...sel.options].some((o) => o.value === fallback)) sel.value = fallback;
    wrap.appendChild(sel);
  } else {
    const input = document.createElement("input");
    input.type = "text";
    input.name = name;
    input.placeholder = "输入模型 ID";
    input.className = "model-input";
    wrap.appendChild(input);
  }
}

function refreshKeyStatus() {
  const provider = $("provider").value;
  const p = state.config.providers.find((x) => x.key === provider);
  const el = $("provider-key-status");
  if (!el) return;
  if (!p) {
    el.textContent = "";
    el.className = "key-status";
    return;
  }
  if (p.env_var === null) {
    el.textContent = "该 provider 无需 API key（本地服务）";
    el.className = "key-status ok";
  } else if (p.has_key) {
    el.textContent = `${p.env_var} 已配置（服务端环境）`;
    el.className = "key-status ok";
  } else {
    el.textContent = `${p.env_var} 未配置——请先在服务端设置，否则调用会失败`;
    el.className = "key-status warn";
  }
}

function bindEvents() {
  $("provider").addEventListener("change", () => {
    refreshModels();
    refreshKeyStatus();
  });
  $("ticker").addEventListener("input", debounce(onTickerInput, 400));
  $("analyze-form").addEventListener("submit", onSubmit);
  document.querySelectorAll(".tab").forEach((b) =>
    b.addEventListener("click", () => switchTab(b.dataset.view))
  );

  // Bind Quick Ticker Chips click
  document.querySelectorAll(".ticker-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const sym = chip.dataset.symbol;
      if (sym) {
        $("ticker").value = sym;
        onTickerInput();
      }
    });
  });
}

// ---------------------------------------------------------------- View Switching

function switchTab(viewName, autoSelectTicker = null, autoSelectDate = null) {
  document.querySelectorAll(".tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === viewName);
  });
  document.querySelectorAll(".view").forEach((v) => {
    v.classList.toggle("active", v.id === `view-${viewName}`);
    v.classList.toggle("hidden", v.id !== `view-${viewName}`);
  });
  if (viewName === "reports") {
    loadReportsList(autoSelectTicker, autoSelectDate);
  }
}

// ---------------------------------------------------------------- Ticker Detection

async function onTickerInput() {
  const ticker = $("ticker").value.trim();
  const el = $("ticker-detect");
  const fund = document.querySelector('input[name="analysts"][value="fundamentals"]');
  const statusBadge = $("ticker-status-badge");

  if (!ticker) {
    el.innerHTML = "";
    if (fund) fund.disabled = false;
    if (statusBadge) statusBadge.style.color = "var(--text-muted)";
    return;
  }

  if (statusBadge) statusBadge.style.color = "var(--accent-cyan)";

  try {
    const res = await fetch(`/api/detect?ticker=${encodeURIComponent(ticker)}`);
    const d = await res.json();
    
    el.innerHTML = `
      <div class="detect-badge">
        <i data-lucide="shield-check"></i>
        <span>AI 解密: ${d.market_type} | ${d.instrument_type} | ${d.asset_type}</span>
      </div>
    `;
    updateLucideIcons();

    const isCrypto = d.asset_type === "crypto";
    if (fund) {
      fund.disabled = isCrypto;
      if (isCrypto) fund.checked = false;
    }
  } catch {
    el.innerHTML = "";
  }
}

// ---------------------------------------------------------------- Analysis Run & SSE Stream

async function onSubmit(ev) {
  ev.preventDefault();
  const form = ev.target;
  const deepThinkVal = form.querySelector('[name="deep_think_llm"]')?.value || "";
  const quickThinkVal = form.querySelector('[name="quick_think_llm"]')?.value || "";
  
  const body = {
    ticker: $("ticker").value.trim(),
    analysis_date: $("analysis-date").value,
    provider: $("provider").value,
    deep_think_llm: deepThinkVal,
    quick_think_llm: quickThinkVal,
    max_debate_rounds: parseInt($("debate-rounds").value, 10) || 1,
    max_risk_discuss_rounds: parseInt($("risk-rounds").value, 10) || 1,
    output_language: $("output-language").value,
    checkpoint: $("checkpoint").checked,
    data_vendors: "akshare,yfinance",
    analysts: [...form.querySelectorAll('input[name="analysts"]:checked')].map((c) => c.value),
  };

  resetResults();
  setStatus("启动量子投研引擎中…", "running");
  
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    connectSSE(data.id);
  } catch (e) {
    setStatus("启动失败：" + e.message, "error");
  }
}

function resetResults() {
  state.lastText = {};
  state.activeNodes.clear();
  $("sections").innerHTML = "";
  $("reliability-card").innerHTML = "";
  const signalCard = $("signal-card");
  signalCard.innerHTML = "";
  signalCard.classList.add("hidden");

  // Reset pipeline visualizer chips
  document.querySelectorAll(".node-chip").forEach((chip) => {
    chip.classList.remove("active", "completed");
  });
}

function setStatus(text, statusClass) {
  const container = $("run-status");
  container.className = `run-status glass-card ${statusClass}`;
  
  let iconName = "circle-dashed";
  let spin = false;

  if (statusClass === "running") {
    iconName = "loader-2";
    spin = true;
  } else if (statusClass === "done") {
    iconName = "check-circle-2";
  } else if (statusClass === "error") {
    iconName = "alert-circle";
  }

  container.innerHTML = `
    <div class="status-indicator">
      <i data-lucide="${iconName}" class="status-icon ${spin ? "spin" : ""}"></i>
      <span class="status-text">${text}</span>
    </div>
  `;
  updateLucideIcons();
}

function connectSSE(id) {
  if (state.source) state.source.close();
  const src = new EventSource(`/api/analyze/${id}/events`);
  state.source = src;

  src.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    handleEvent(msg);
  };

  src.onerror = () => {
    // EventSource auto-reconnects
  };
}

function handleEvent(msg) {
  switch (msg.type) {
    case "meta":
      setStatus(
        `正在分析标的 ${msg.ticker}（${msg.market_type} / ${msg.instrument_type}；活跃分析师：${msg.analysts.join(", ")}）`,
        "running"
      );
      break;
    case "chunk":
      updateFromChunk(msg.data);
      break;
    case "signal":
      showSignal(msg.signal);
      break;
    case "done":
      setStatus(`分析完成 — 最终决策信号：${msg.signal || "n/a"}`, "done");
      showSignal(msg.signal);
      linkToReport(msg.ticker, msg.report_path);
      break;
    case "error":
      setStatus("分析过程出错：" + msg.message, "error");
      break;
    case "end":
      if (state.source) state.source.close();
      break;
  }
}

function setPipelineNodeStatus(nodeKey, status) {
  const chip = document.querySelector(`.node-chip[data-node="${nodeKey}"]`);
  if (chip) {
    if (status === "active") {
      chip.classList.remove("completed");
      chip.classList.add("active");
    } else if (status === "completed") {
      chip.classList.remove("active");
      chip.classList.add("completed");
    }
  }
}

function updateFromChunk(data) {
  const textFields = [
    ["market_report", "Market Analyst", "trending-up"],
    ["sentiment_report", "Sentiment Analyst", "message-square"],
    ["news_report", "News Analyst", "newspaper"],
    ["fundamentals_report", "Fundamentals Analyst", "pie-chart"],
  ];

  for (const [key, title, icon] of textFields) {
    if (data[key]) {
      updateSection(key, title, data[key], icon);
      setPipelineNodeStatus(key, "completed");
    }
  }

  if (data.investment_debate_state) {
    updateDebate(data.investment_debate_state);
    setPipelineNodeStatus("debate", "completed");
  }
  if (data.risk_debate_state) {
    updateRisk(data.risk_debate_state);
    setPipelineNodeStatus("risk", "completed");
  }
  if (data.trader_investment_plan) {
    updateSection("trader", "Trader 投资交易计划", data.trader_investment_plan, "briefcase");
    setPipelineNodeStatus("trader", "completed");
  }
  if (data.final_trade_decision) {
    updateSection("decision", "Portfolio Manager 最终裁决报告", data.final_trade_decision, "award");
    setPipelineNodeStatus("decision", "completed");
  }
  if (data.data_contract_status) {
    updateReliability(data.data_contract_status);
  }
}

function updateDebate(st) {
  const parts = [];
  if (st.bull_history) parts.push(`## Bull Researcher 看多观点\n\n${st.bull_history}`);
  if (st.bear_history) parts.push(`## Bear Researcher 看空观点\n\n${st.bear_history}`);
  if (st.judge_decision) parts.push(`## Research Manager 裁决总结\n\n${st.judge_decision}`);
  if (parts.length) updateSection("debate", "多空研究辩论 (Bull / Bear Debate)", parts.join("\n\n"), "swords");
}

function updateRisk(st) {
  const parts = [];
  if (st.aggressive_history) parts.push(`## Aggressive Analyst 激进风控视角\n\n${st.aggressive_history}`);
  if (st.conservative_history) parts.push(`## Conservative Analyst 保守风控视角\n\n${st.conservative_history}`);
  if (st.neutral_history) parts.push(`## Neutral Analyst 中性风控视角\n\n${st.neutral_history}`);
  if (st.judge_decision) parts.push(`## Risk Portfolio Manager 最终风控裁决\n\n${st.judge_decision}`);
  if (parts.length) updateSection("risk", "风险评估矩阵讨论 (Risk Debate)", parts.join("\n\n"), "shield-alert");
}

// ---------------------------------------------------------------- Markdown & UI Section Rendering

async function renderMarkdown(text) {
  try {
    const res = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ markdown: text }),
    });
    const data = await res.json();
    return data.html || "";
  } catch {
    return `<p>${text}</p>`;
  }
}

async function updateSection(key, title, text, iconName = "file-text") {
  if (state.lastText[key] === text) return;
  state.lastText[key] = text;

  let card = document.querySelector(`[data-section="${key}"]`);
  if (!card) {
    card = document.createElement("details");
    card.className = "section-card";
    card.open = true;
    card.dataset.section = key;

    const summary = document.createElement("summary");
    summary.innerHTML = `
      <div class="summary-title">
        <i data-lucide="${iconName}"></i>
        <span>${title}</span>
      </div>
      <i data-lucide="chevron-down" class="chevron-icon"></i>
    `;

    const body = document.createElement("div");
    body.className = "markdown";

    card.appendChild(summary);
    card.appendChild(body);
    $("sections").appendChild(card);
    updateLucideIcons();
  }

  const markdownDiv = card.querySelector(".markdown");
  markdownDiv.innerHTML = await renderMarkdown(text);
  updateLucideIcons();
}

function showSignal(sig) {
  if (!sig) return;
  const card = $("signal-card");
  card.classList.remove("hidden");

  let signalClass = "hold";
  let iconName = "minus-circle";
  let labelText = "NEUTRAL / HOLD";

  const upperSig = sig.toUpperCase();
  if (upperSig.includes("BUY")) {
    signalClass = "buy";
    iconName = "trending-up";
    labelText = "BULLISH / BUY";
  } else if (upperSig.includes("SELL")) {
    signalClass = "sell";
    iconName = "trending-down";
    labelText = "BEARISH / SELL";
  }

  card.innerHTML = `
    <div class="signal-hero ${signalClass}">
      <div class="signal-meta">
        <span class="signal-label">QUANTUM SIGNAL DECISION</span>
        <div class="signal-value">${sig}</div>
      </div>
      <div class="signal-badge">
        <i data-lucide="${iconName}"></i>
        <span>${labelText}</span>
      </div>
    </div>
  `;
  updateLucideIcons();
}

function linkToReport(ticker, reportPath) {
  if (!reportPath) return;
  const existing = document.querySelector(".report-link-banner");
  if (existing) existing.remove();

  const div = document.createElement("div");
  div.className = "report-link-banner glass-card";
  div.style.padding = "14px 20px";
  div.style.marginTop = "10px";
  div.style.display = "flex";
  div.style.alignItems = "center";
  div.style.justifyContent = "space-between";

  div.innerHTML = `
    <span style="font-weight: 600; color: var(--accent-cyan); display: flex; align-items: center; gap: 8px;">
      <i data-lucide="check-circle-2"></i> 完整 Markdown 投研报告已成功归档保存
    </span>
    <button id="view-saved-report-btn" class="glow-button" style="width: auto; padding: 6px 14px; font-size: 12px; margin: 0;">
      <i data-lucide="external-link"></i> 打开完整归档
    </button>
  `;
  $("sections").prepend(div);
  updateLucideIcons();

  $("view-saved-report-btn").addEventListener("click", () => {
    // Extract date from reportPath if available (e.g. results/600519/2026-08-07/complete_report.md)
    let dateStr = null;
    if (reportPath) {
      const parts = reportPath.split("/");
      if (parts.length >= 2) {
        dateStr = parts[parts.length - 2];
      }
    }
    switchTab("reports", ticker, dateStr);
  });
}

function updateReliability(status) {
  const container = $("reliability-card");
  if (!container || !status) return;

  const overall = status.overall || "not_checked";
  let statusClass = "pass";
  let iconName = "check-circle-2";
  let titleText = "数据源契约校验通过";

  if (overall === "warning" || overall === "warn") {
    statusClass = "warn";
    iconName = "alert-triangle";
    titleText = "数据源契约校验存在警告风险";
  } else if (overall === "fail" || overall === "error") {
    statusClass = "fail";
    iconName = "x-circle";
    titleText = "数据源契约校验未通过";
  }

  let checksHtml = "";
  if (status.checks && status.checks.length) {
    checksHtml = status.checks
      .map((c) => {
        const failures = c.failures || [];
        const warnings = c.warnings || [];
        const hasFailures = failures.length > 0;
        const hasWarnings = warnings.length > 0;

        let itemClass = "pass";
        let itemIcon = "check";
        if (hasFailures) {
          itemClass = "fail";
          itemIcon = "x";
        } else if (hasWarnings) {
          itemClass = "warn";
          itemIcon = "alert-triangle";
        }

        const sourceLabel = c.source ? `${c.source.toUpperCase()}` : "VENDOR";
        const symbolLabel = c.symbol ? ` [${c.symbol}]` : "";
        const semanticLabel = c.semantic ? ` (${c.semantic})` : "";
        
        let msg = "契约校验合格";
        if (hasFailures) {
          msg = failures.join(", ");
        } else if (hasWarnings) {
          msg = warnings.join(", ");
        } else if (c.as_of) {
          msg = `数据基准日 ${c.as_of}`;
        }

        return `
          <div class="reliability-item ${itemClass}">
            <i data-lucide="${itemIcon}"></i>
            <span><strong>${sourceLabel}${symbolLabel}${semanticLabel}</strong>: ${msg}</span>
          </div>
        `;
      })
      .join("");
  } else {
    checksHtml = `
      <div class="reliability-item pass">
        <i data-lucide="check"></i>
        <span>全量基础数据源响应正常</span>
      </div>
    `;
  }

  container.innerHTML = `
    <div class="reliability-card ${statusClass}">
      <div class="reliability-header">
        <i data-lucide="${iconName}"></i>
        <span>${titleText}</span>
      </div>
      <div class="reliability-checks">${checksHtml}</div>
    </div>
  `;
  updateLucideIcons();
}

// ---------------------------------------------------------------- Archived Reports Viewer

async function loadReportsList(autoSelectTicker = null, autoSelectDate = null) {
  const container = $("reports-list");
  if (!container) return;
  container.innerHTML = `<p class="muted">正在调取历史报告归档库…</p>`;

  try {
    const res = await fetch("/api/reports");
    const data = await res.json();
    const reports = data.reports || [];

    if (!reports || !reports.length) {
      container.innerHTML = `<p class="muted">暂无历史报告归档。</p>`;
      return;
    }

    container.innerHTML = "";
    let targetBtnToClick = null;

    reports.forEach((rep) => {
      const btn = document.createElement("button");
      btn.className = "report-item";
      const isAutoTarget = autoSelectTicker && autoSelectDate && rep.ticker === autoSelectTicker && rep.date === autoSelectDate;

      btn.innerHTML = `
        <div style="font-weight: 700; display: flex; align-items: center; justify-content: space-between;">
          <span>${rep.ticker}</span>
          <i data-lucide="file-text" style="width: 14px; height: 14px; color: var(--accent-cyan);"></i>
        </div>
        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px; display: flex; align-items: center; justify-content: space-between;">
          <span>${rep.date}</span>
          ${rep.has_data_reliability ? '<span style="color: var(--accent-cyan); font-size: 10px;">契约归档</span>' : ''}
        </div>
      `;
      btn.addEventListener("click", () => {
        document.querySelectorAll(".report-item").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        viewReport(rep.ticker, rep.date);
      });

      if (isAutoTarget) {
        targetBtnToClick = btn;
      }
      container.appendChild(btn);
    });

    updateLucideIcons();

    if (targetBtnToClick) {
      targetBtnToClick.click();
    } else if (container.firstChild && container.firstChild.click) {
      // Auto view the first report in archive list
      container.firstChild.click();
    }
  } catch (err) {
    console.error("Failed to load reports list", err);
    container.innerHTML = `<p class="muted">调取历史报告失败。</p>`;
  }
}

async function viewReport(ticker, date, file = "complete_report.md") {
  const viewer = $("report-viewer");
  if (!viewer) return;
  viewer.innerHTML = `<p class="muted">正在调取并渲染归档报告 [${ticker}] (${date})…</p>`;

  try {
    const url = `/api/reports/data?ticker=${encodeURIComponent(ticker)}&date=${encodeURIComponent(date)}&file=${encodeURIComponent(file)}`;
    const res = await fetch(url);
    const data = await res.json();

    if (res.ok && (data.html || data.markdown)) {
      viewer.innerHTML = data.html || (await renderMarkdown(data.markdown));
      updateLucideIcons();
    } else {
      viewer.innerHTML = `<p class="muted">无法读取报告内容：${data.error || "未找到归档记录"}</p>`;
    }
  } catch (err) {
    console.error("Failed to load report data", err);
    viewer.innerHTML = `<p class="muted">加载报告数据失败。</p>`;
  }
}

// ---------------------------------------------------------------- Entry Point

document.addEventListener("DOMContentLoaded", init);
