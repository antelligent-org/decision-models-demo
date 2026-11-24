/* app.js: the decision models demo page. Plain script, no build step.
   It talks only to serve.py on the same origin; serve.py talks to the models. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var VIEWS = ["task", "live", "results", "oos", "retrain", "try"];
  var info = null;
  var view = null;

  // ── Formatting. Never round in a model's favour: accuracy down, time and cost up. ──
  function pctDown(x) { return x == null ? "n/a" : (Math.floor(x * 1000 + 1e-9) / 10).toFixed(1) + "%"; }
  function msUp(x) { return x == null ? "n/a" : Math.ceil(x - 1e-9).toLocaleString("en-GB"); }
  function usdUp(x) {
    if (x == null) return "n/a";
    var p = x < 1 ? 4 : 2, f = Math.pow(10, p);
    return "$" + (Math.ceil(x * f - 1e-9) / f).toLocaleString("en-GB", { minimumFractionDigits: p, maximumFractionDigits: p });
  }
  function conf(x) { return x == null ? "n/a" : (Math.floor(x * 100) / 100).toFixed(2); }
  function n(x) { return x == null ? "n/a" : x.toLocaleString("en-GB"); }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function median(xs) { if (!xs.length) return null; var s = xs.slice().sort(function (a, b) { return a - b; }); return s[Math.ceil(s.length / 2) - 1]; }

  function toast(text) {
    var t = $("toast"); t.textContent = text; t.classList.add("on");
    clearTimeout(toast.timer); toast.timer = setTimeout(function () { t.classList.remove("on"); }, 2200);
  }

  function api(path, body) {
    var opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    return fetch(path, opts).then(function (r) {
      return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.status); return j; });
    });
  }

  // ── Views ──
  function show(name) {
    if (VIEWS.indexOf(name) < 0) name = "task";
    view = name;
    VIEWS.forEach(function (v) { $("view-" + v).classList.toggle("on", v === name); });
    document.querySelectorAll(".tabs button").forEach(function (b) { b.classList.toggle("on", b.dataset.view === name); });
    if (history.replaceState) history.replaceState(null, "", "#" + name);
    if (name === "oos") setTimeout(function () { $("oosInput").focus(); }, 50);
    if (name === "results") renderResults();
  }

  function demo() { return (info && info.demo) || {}; }

  var MODELS = ["llm", "modernbert", "fern"];
  var NAMES = { llm: "LLM", modernbert: "ModernBERT", fern: "Fern" };

  // Fern by name and size, the way ModernBERT is shown: "fern-pinna · 1.5B".
  function fernName(base, name) {
    var c = (info && info.fern) || {};
    name = name || (base === c.base ? c.name : null);
    if (!name) return base || "not trained yet";
    var params = base === c.base ? c.params : null;
    return name.toLowerCase().replace(/\s+/g, "-") + (params ? " · " + params : "");
  }

  // ── Theme: dark or light, whichever the projector shows best. Remembered per browser. ──
  function setTheme(t) {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("dm-theme", t); } catch (e) {}
  }
  function toggleTheme() { setTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light"); }

  // ── Status ──
  function renderStatus() {
    var pill = $("modePill");
    var cached = info.mode === "cached";
    var down = [];
    if (!info.llm.ready) down.push("LLM");
    if (!info.fern.ready) down.push("Fern");
    pill.className = "pill" + (cached ? " is-cached" : down.length ? " is-off" : " is-live");
    pill.querySelector("span").textContent = cached ? "Cached" : down.length ? "Live · " + down.join(", ") + " offline" : "Live";
    pill.title = cached ? "The LLM and Fern answer from the benchmark run and earlier answers" :
      down.length ? "Retrying every 30 seconds; until then those answers come from the cache" : "All three models live";
    document.querySelectorAll('[data-fill="llm-model"]').forEach(function (e) { e.textContent = info.llm.model || "not connected"; });
    document.querySelectorAll('[data-fill="llm-host"]').forEach(function (e) {
      e.textContent = info.llm.provider === "anthropic" ? "Anthropic's API" : info.llm.provider === "openai" ? "OpenAI's API" : "A hosted API";
    });
    document.querySelectorAll('[data-fill="fern-model"]').forEach(function (e) { e.textContent = fernName(info.fern.base); });
    document.querySelectorAll('[data-fill="train-count"]').forEach(function (e) { e.textContent = n(info.train_count); });
    $("attribution").textContent = "Data: " + info.dataset.attribution;
  }

  // ── 1. The task ──
  function renderTask() {
    var m = demo().task || { text: "I still have not received my new card, I ordered over a week ago.", label: "card_arrival" };
    $("taskMessage").textContent = m.text;
    $("taskGold").textContent = m.label;
    // All 77 as dots, one lit: the size of the choice without 77 names on screen. Hover one for its name.
    $("dots").innerHTML = info.intents.map(function (l) {
      return '<span data-name="' + esc(l) + '" aria-label="' + esc(l) + '"' + (l === m.label ? ' class="hit"' : "") + "></span>";
    }).join("");
  }

  // ── 2. Live ──
  var live = { rows: [], running: false, seed: 2611, results: { llm: [], modernbert: [], fern: [] } };

  function loadSample(seed) {
    live.seed = seed;
    var size = demo().live_count || 20;
    return api("/api/sample?n=" + size + "&seed=" + seed).then(function (rows) {
      live.rows = rows;
      live.results = { llm: [], modernbert: [], fern: [] };
      $("liveCount").textContent = rows.length;
      var grid = $("liveGrid");
      grid.querySelectorAll(".cell").forEach(function (c) { c.remove(); });
      rows.forEach(function (r, i) {
        var msg = document.createElement("div");
        msg.className = "cell msg"; msg.textContent = r.text; msg.title = r.text + "  (" + r.label + ")";
        grid.appendChild(msg);
        MODELS.forEach(function (col) {
          var c = document.createElement("div");
          c.className = "cell col-" + col; c.id = "cell-" + col + "-" + i;
          grid.appendChild(c);
        });
      });
      MODELS.forEach(function (col) { $("score-" + col).textContent = ""; });
    });
  }

  function invented(label) { return info && label && label !== "none_of_these" && info.intents.indexOf(label) < 0; }
  function shortError(msg) {
    msg = String(msg || "");
    if (msg.indexOf("no cached answer") >= 0) return "offline, and no cached answer for this message";
    if (/no (OpenAI key|Claude API key|Fern API key)/.test(msg)) return "not connected";
    return msg.slice(0, 90);
  }
  function cellHtml(r) {
    if (r.error) return '<span class="lab off">offline</span>';
    return '<span class="tick ' + (r.correct ? "ok" : "bad") + '">' + (r.correct ? "✓" : "✗") + "</span>" +
      '<span class="lab" title="' + esc(r.label) + (invented(r.label) ? " (not one of the 77)" : "") + '"' +
      (invented(r.label) ? ' style="font-style:italic"' : "") + ">" + esc(r.label) + "</span>" +
      (r.cached ? '<span class="cachedtag" title="' + esc(r.cached_why) + '">cached</span>' : "") +
      '<span class="conf">' + (r.confidence == null ? "" : conf(r.confidence)) + "</span>" +
      '<span class="ms">' + msUp(r.ms) + " ms</span>";
  }

  function score(col) {
    var rs = live.results[col].filter(function (r) { return !r.error; });
    var right = rs.filter(function (r) { return r.correct; }).length;
    var p50 = median(rs.map(function (r) { return r.ms; }));
    $("score-" + col).innerHTML = rs.length ? right + "/" + rs.length + "<small>p50 " + msUp(p50) + " ms</small>" : "";
  }

  function runColumn(col, token) {
    var i = 0;
    function next() {
      if (token !== live.token || i >= live.rows.length) return Promise.resolve();
      var idx = i++, cell = $("cell-" + col + "-" + idx);
      cell.className = "cell col-" + col + " pending";
      return api("/api/ask", { model: col, text: live.rows[idx].text })
        .catch(function (err) { return { error: String(err.message || err) }; })
        .then(function (r) {
          if (token !== live.token) return;
          live.results[col].push(r);
          cell.className = "cell col-" + col + " fresh";
          cell.innerHTML = cellHtml(r);
          if (r.error) cell.title = r.error;
          score(col);
          return next();
        });
    }
    return next();
  }

  // ModernBERT's GPU idles down when the page sits still, and its first call after that
  // is slow. A few unscored calls first, as the benchmark makes, so the timed calls are warm.
  function warmBert() {
    var p = Promise.resolve();
    for (var i = 0; i < 5; i++) p = p.then(function () { return api("/api/warm", {}); });
    return p.catch(function () {});
  }

  function runLive() {
    if (live.running) return;
    live.running = true;
    live.token = {};
    live.results = { llm: [], modernbert: [], fern: [] };
    document.querySelectorAll("#liveGrid .cell:not(.msg)").forEach(function (c) { c.className = c.className.replace(" fresh", ""); c.innerHTML = ""; });
    var token = live.token;
    warmBert().then(function () {
      return Promise.all(MODELS.map(function (col) { return runColumn(col, token); }));
    }).then(function () { live.running = false; });
  }

  // ── 3. Results ──
  function renderResults() {
    var d = info && info.results;
    if (!d) { $("resultsTable").innerHTML = "<tr><td>No full run yet: run bench.py run</td></tr>"; $("routeCard").style.display = "none"; return; }
    var cols = MODELS.filter(function (c) { return d.models[c]; });
    var s = function (c) { return d.models[c].summary; };
    var model = {
      llm: function (m) { return m.config.model; },
      modernbert: function () { return "ModernBERT-base · 149M"; },
      fern: function (m) { return fernName(m.base, m.base_name); }
    };
    var best = function (key, hi) {
      var vals = cols.map(function (c) { return s(c)[key]; }).filter(function (v) { return v != null; });
      var target = hi ? Math.max.apply(null, vals) : Math.min.apply(null, vals);
      return function (c) { return s(c)[key] === target ? " best" : ""; };
    };
    var bAcc = best("accuracy", true), bP50 = best("p50_ms", false), bCost = best("cost_per_10k_usd", false);
    var head = "<thead><tr><th></th>" + cols.map(function (c) {
      return '<th class="col-' + c + '"><span class="dot"></span>' + NAMES[c] +
        "<small>" + esc(model[c](d.models[c].meta)) + "</small></th>";
    }).join("") + "</tr></thead>";
    var row = function (label, f) { return "<tr><td>" + label + "</td>" + cols.map(f).join("") + "</tr>"; };
    var cost = function (c) {
      var sum = s(c);
      if (sum.plan) {  // a flat monthly fee has no price per call to set beside the others
        return '<td class="col-' + c + ' big na">N/A<small>flat $' + n(sum.plan.usd_per_month) + " a month · " +
          esc(sum.plan.plan) + "</small></td>";
      }
      var how = sum.instance ? "compute s × $" + sum.instance.usd_per_hour + "/h, " + esc(sum.instance.instance.replace("AWS EC2 ", ""))
        : "API tokens × list price";
      return '<td class="col-' + c + ' big' + bCost(c) + '">' + usdUp(sum.cost_per_10k_usd) +
        "<small>" + how + "</small></td>";
    };
    var body = [
      row("Accuracy", function (c) { return '<td class="col-' + c + ' big' + bAcc(c) + '">' + pctDown(s(c).accuracy) + "<small>" + n(s(c).right) + " / " + n(s(c).n) + "</small></td>"; }),
      row("p50 per call", function (c) { return '<td class="col-' + c + ' big' + bP50(c) + '">' + msUp(s(c).p50_ms) + " ms</td>"; }),
      row("p95 per call", function (c) { return "<td>" + msUp(s(c).p95_ms) + " ms</td>"; }),
      row("Cost per 10,000 calls", cost),
      row("Training examples", function (c) { return "<td>" + (c === "llm" ? "0" : n(d.train_messages)) + "</td>"; }),
      row("Answers not one of the 77", function (c) { return "<td>" + n(s(c).not_an_intent || 0) + "</td>"; })
    ].join("");
    $("resultsTable").innerHTML = head + "<tbody>" + body + "</tbody>";
    var when = new Date(d.started_utc);
    $("resultsTitle").textContent = "All " + n(d.test_messages) + " test messages";
    $("resultsStamp").textContent = "Run " +
      when.toLocaleString("en-GB", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZoneName: "short" }) +
      " · one call at a time · rounded against each model: accuracy down, latency and cost up";
    var r = d.routing;
    if (r && r.closed) {
      $("routeCard").innerHTML =
        "<div><h4>Use both: ModernBERT answers when it is sure<small>confidence above " + r.threshold + ", the rest go to the LLM</small></h4></div>" +
        '<div><div class="num">' + (Math.floor(r.modernbert_share * 1000) / 10).toFixed(1) + '%</div><div class="lbl">handled by ModernBERT</div></div>' +
        '<div><div class="num">' + pctDown(r.modernbert_accuracy_on_its_share) + '</div><div class="lbl">ModernBERT accuracy</div></div>' +
        '<div><div class="num">' + pctDown(r.closed.blended_accuracy) + '</div><div class="lbl">blended accuracy</div></div>' +
        '<div><div class="num">' + usdUp(r.closed.blended_usd_per_10k) + '</div><div class="lbl">blended cost per 10,000</div></div>';
    } else { $("routeCard").innerHTML = ""; }
    $("routeCard").style.display = $("routeCard").innerHTML ? "" : "none";
    $("resultsNote").textContent = "Latency: Fern includes a round trip to UK South; ModernBERT runs in-process on the laptop. " +
      "ModernBERT's cost prices its measured seconds at a cloud machine with the same chip, kept busy, one call at a time.";
  }

  // ── 4. Out of set ──
  var oosPick = -1;
  function answerHtml(r, col) {
    if (!r) return "";
    if (r.error) return '<div class="err">' + esc(shortError(r.error)) + "</div>";
    var top = col === "modernbert" && r.top ? '<div class="top">' + r.top.slice(0, 3).map(function (t) { return esc(t[0]) + " " + conf(t[1]); }).join(" · ") + "</div>" : "";
    return '<div class="lab' + (invented(r.label) ? " bad" : "") + '">' + esc(r.label) + "</div>" +
      (invented(r.label) ? '<div class="err">not one of the 77: a tool it made up</div>' : "") +
      '<div class="meta">' + (r.confidence == null ? "" : "confidence <b>" + conf(r.confidence) + "</b> · ") +
      msUp(r.ms) + " ms" + (r.cached ? " · cached" : "") + "</div>" + top;
  }

  function askOos() {
    var text = $("oosInput").value.trim();
    if (!text) return;
    MODELS.forEach(function (c) { $("oos-" + c).innerHTML = '<div class="meta">…</div>'; });
    $("meterFill").style.width = "0"; $("routerVerdict").textContent = "…"; $("llmReply").innerHTML = "";
    ["llm", "fern"].forEach(function (col) {
      api("/api/ask", { model: col, text: text }).catch(function (e) { return { error: e.message }; })
        .then(function (r) { $("oos-" + col).innerHTML = answerHtml(r, col); });
    });
    warmBert().then(function () { return api("/api/route", { text: text }); }).then(function (res) {
      var guess = res.modernbert;
      $("oos-modernbert").innerHTML = answerHtml(guess, "modernbert");
      var fill = $("meterFill");
      fill.style.width = Math.max(1, guess.confidence * 100) + "%";
      fill.classList.toggle("low", res.routed);
      $("routerVerdict").innerHTML = res.routed
        ? "ModernBERT is <b>" + conf(guess.confidence) + "</b> sure, below " + res.threshold + ": <b>this one goes to the LLM</b>"
        : "ModernBERT is <b>" + conf(guess.confidence) + "</b> sure, above " + res.threshold + ": ModernBERT answers, no LLM call";
      if (res.routed && res.llm) {
        var l = res.llm;
        $("llmReply").innerHTML = l.error ? '<span class="err">LLM: ' + esc(shortError(l.error)) + "</span>" :
          '<span class="who">LLM (' + esc(info.llm.model) + ") · " + esc(l.label) + " · " + msUp(l.ms) + " ms" + (l.cached ? " · cached" : "") + "</span>" +
          esc(l.reply || "(picked an intent; no reply needed)");
      }
    }).catch(function (e) { $("oos-modernbert").innerHTML = '<div class="err">' + esc(e.message) + "</div>"; });
  }

  // ── 5. Retrain ──
  var rt = { timer: null, poll: null, t0: 0 };
  function rtMsg() { return demo().retrain || null; }

  function renderRetrain() {
    var m = rtMsg();
    $("rtMessage").textContent = m ? m.text : "Choose a message: set retrain in demo.json";
    $("rtGold").querySelector(".answer").innerHTML = m ? '<div class="lab ok">' + esc(m.label) + "</div>" : "";
    $("rtBefore").querySelector(".answer").innerHTML = "";
    $("rtAfter").querySelector(".answer").innerHTML = "";
    $("rtSummary").innerHTML = "";
    $("rtTimer").textContent = "0.0 s"; $("rtPhase").textContent = "Ready"; $("rtBar").style.width = "0";
  }

  function askBefore() {
    var m = rtMsg(); if (!m) return;
    warmBert().then(function () { return api("/api/ask", { model: "modernbert", text: m.text }); }).then(function (r) {
      $("rtBefore").querySelector(".answer").innerHTML =
        '<div class="lab ' + (r.correct ? "ok" : "bad") + '">' + (r.correct ? "✓ " : "✗ ") + esc(r.label) + "</div>" +
        '<div class="meta">confidence ' + conf(r.confidence) + " · " + msUp(r.ms) + " ms" + (r.retrained ? " · retrained model" : "") + "</div>";
    });
  }

  function startRetrain() {
    var m = rtMsg(); if (!m) return;
    api("/api/retrain", { text: m.text, label: m.label }).then(function () {
      rt.t0 = performance.now();
      clearInterval(rt.timer); clearInterval(rt.poll);
      rt.timer = setInterval(function () { $("rtTimer").textContent = ((performance.now() - rt.t0) / 1000).toFixed(1) + " s"; }, 100);
      rt.poll = setInterval(pollRetrain, 250);
    }).catch(function (e) { toast(e.message); });
  }

  function pollRetrain() {
    api("/api/retrain/status").then(function (s) {
      var phase = s.phase || s.state;
      if (s.state === "running" && phase === "training") phase = "training: step " + s.step + " of " + s.steps + (s.loss != null ? " · loss " + s.loss.toFixed(3) : "");
      $("rtPhase").textContent = phase.charAt(0).toUpperCase() + phase.slice(1);
      var frac = s.state === "done" ? 1 : s.phase === "training" ? 0.15 + 0.6 * s.step / s.steps : s.phase && s.phase.indexOf("rerunning") === 0 ? 0.85 : 0.08;
      $("rtBar").style.width = (frac * 100) + "%";
      if (s.after && !$("rtAfter").dataset.shown) {
        var m = rtMsg(), ok = s.after.label === m.label;
        $("rtAfter").querySelector(".answer").innerHTML = '<div class="lab ' + (ok ? "ok" : "bad") + '">' + (ok ? "✓ " : "✗ ") + esc(s.after.label) + "</div>" +
          '<div class="meta">confidence ' + conf(s.after.confidence) + " · trained " + s.train_seconds.toFixed(1) + " s</div>";
        $("rtAfter").dataset.shown = "1";
      }
      if (s.state === "done" || s.state === "failed") {
        clearInterval(rt.timer); clearInterval(rt.poll);
        delete $("rtAfter").dataset.shown;
        if (s.state === "failed") { $("rtPhase").textContent = "Failed: " + s.error; return; }
        $("rtTimer").textContent = s.seconds.toFixed(1) + " s";
        var broken = s.broken || [], fixed = s.fixed || [];
        $("rtSummary").innerHTML =
          '<div><div class="num">' + pctDown(s.accuracy_before) + '</div><div class="lbl">before · ' + n(s.right_before) + " / " + n(info.test_count) + "</div></div>" +
          '<div><div class="num">' + pctDown(s.accuracy_after) + '</div><div class="lbl">after · ' + n(s.right_after) + " / " + n(info.test_count) + "</div></div>" +
          '<div><div class="num" style="color:var(--ok)">' + fixed.length + '</div><div class="lbl">test messages it now gets right</div></div>' +
          '<div><div class="num" style="color:' + (broken.length ? "var(--bad)" : "var(--ok)") + '">' + broken.length + '</div><div class="lbl">it used to get right, and now gets wrong</div></div>' +
          '<div class="list">' + (broken.length ? "Borderline messages that moved, for example: " + broken.slice(0, 3).map(function (b) { return "“" + esc(b.text) + "” → " + esc(b.now); }).join(" · ") + ". " : "Nothing that was right broke. ") +
          "The corrected message is one of the 3,080, so the after count includes one message the model has now seen.</div>";
      }
    });
  }

  function resetBert() {
    api("/api/reset", {}).then(function () { renderRetrain(); toast("ModernBERT is back to the original checkpoint"); });
  }

  // ── 6. Try it ──
  function renderTry() {
    $("repoUrl").textContent = demo().repo_url || "(repository link: set repo_url in demo.json)";
    $("tryCredit").textContent = info.dataset.attribution + ": " + info.dataset.url +
      ". ModernBERT by Answer.AI and LightOn, Apache 2.0. Fern models by antelligent.";
  }

  // ── Keys ──
  document.addEventListener("keydown", function (e) {
    var typing = document.activeElement === $("oosInput");
    if (typing) {
      // Sending releases the box, so the number keys switch views again; Enter or / brings it back.
      if (e.key === "Enter") { e.preventDefault(); askOos(); $("oosInput").blur(); }
      else if (e.key === "Tab") {
        e.preventDefault();
        var list = demo().out_of_set || [];
        if (list.length) { oosPick = (oosPick + 1) % list.length; $("oosInput").value = list[oosPick]; }
      } else if (e.key === "Escape") { $("oosInput").blur(); }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var k = e.key.toLowerCase();
    if (k >= "1" && k <= "6") { show(VIEWS[+k - 1]); return; }
    if (k === "arrowright") { show(VIEWS[Math.min(VIEWS.length - 1, VIEWS.indexOf(view) + 1)]); return; }
    if (k === "arrowleft") { show(VIEWS[Math.max(0, VIEWS.indexOf(view) - 1)]); return; }
    if (k === "l") { toggleTheme(); return; }
    if (k === "c") {
      api("/api/mode", { mode: info.mode === "cached" ? "live" : "cached" }).then(function (r) {
        info.mode = r.mode; renderStatus(); toast(r.mode === "cached" ? "Cached: the LLM and Fern replay recorded answers" : "Live");
      });
      return;
    }
    if (view === "live" && k === " ") { e.preventDefault(); runLive(); }
    if (view === "live" && k === "s" && !live.running) { loadSample(live.seed + 1); }
    if (view === "oos" && (k === "enter" || k === "/")) { e.preventDefault(); $("oosInput").focus(); }
    if (view === "retrain" && k === "enter") askBefore();
    if (view === "retrain" && k === "r") startRetrain();
    if (view === "retrain" && k === "x") resetBert();
  });
  document.querySelectorAll(".tabs button").forEach(function (b) { b.addEventListener("click", function () { show(b.dataset.view); b.blur(); }); });
  $("themeBtn").addEventListener("click", function () { toggleTheme(); this.blur(); });
  window.addEventListener("hashchange", function () { if (info) show(location.hash.slice(1)); });

  // ── Boot ──
  if (location.protocol === "file:") {
    document.body.insertAdjacentHTML("afterbegin", '<p class="offline-note">This page needs its service: run <code>.venv/bin/python serve.py</code> in this folder and open http://127.0.0.1:8765/</p>');
    return;
  }
  api("/api/info").then(function (d) {
    info = d;
    renderStatus(); renderTask(); renderRetrain(); renderTry();
    loadSample(demo().live_seed || 2611);
    show((location.hash || "#task").slice(1));
  }).catch(function (e) { toast("Service not reachable: " + e.message); });
  setInterval(function () { api("/api/info").then(function (d) { var keep = info.mode; info = d; if (d.mode !== keep) info.mode = d.mode; renderStatus(); }).catch(function () {}); }, 15000);
})();
