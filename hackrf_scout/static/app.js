'use strict';
(function () {
  // ------------------------------------------------------------------ helpers
  const $ = (sel, root) => (root || document).querySelector(sel);
  const SVGNS = 'http://www.w3.org/2000/svg';

  // Build DOM without innerHTML: every string goes in as text, so log lines, signal names and
  // notes (some come from third-party data) can never inject markup.
  const norm = (kids) => kids.flat(Infinity).filter((k) => k != null && k !== false).map((k) => (k.nodeType ? k : document.createTextNode(String(k))));
  // like el.replaceChildren(), but skips null/false and flattens arrays instead of printing them
  function fill(el, ...kids) { el.replaceChildren(...norm(kids)); return el; }
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v === false || v == null) continue;
      if (k === 'class') el.className = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else if (k in el && k !== 'list') el[k] = v;
      else el.setAttribute(k, v === true ? '' : v);
    }
    el.append(...norm(kids));
    return el;
  }
  function svg(tag, attrs, ...kids) {
    const el = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
    el.append(...kids);
    return el;
  }
  const option = (value, label) => h('option', { value: String(value) }, label == null ? String(value) : label);
  const field = (label, input) => h('label', null, label, input);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };

  const trim = (n) => String(+n.toFixed(3));
  const fmtMHz = (hz) => (hz / 1e6).toFixed(3);
  const fmtKHz = (hz) => (hz / 1e3).toLocaleString(undefined, { maximumFractionDigits: 1 });
  const fmtNum = (n, d) => (n == null ? '–' : (+n).toFixed(d == null ? 1 : d));
  const fmtInt = (n) => (n == null ? '–' : Number(n).toLocaleString());
  const fmtTime = (ts) => (ts ? String(ts).replace('T', ' ').slice(0, 19) : '–');
  const fmtBytes = (n) => (n == null ? '–' : n > 1e6 ? (n / 1e6).toFixed(1) + ' MB' : (n / 1e3).toFixed(0) + ' kB');
  const rangeLabel = (b) => `${trim(b.lo_hz / 1e6)}–${trim(b.hi_hz / 1e6)} MHz`;
  const safeUrl = (u) => {
    try {
      const x = new URL(u);
      return x.protocol === 'https:' || x.protocol === 'http:' ? x.href : null;
    } catch (e) { return null; }
  };
  const extLink = (url, text) => {
    const href = safeUrl(url);
    return href ? h('a', { href, target: '_blank', rel: 'noopener noreferrer' }, text) : text;
  };

  // Timestamps from the server are local time without a zone, so "how long ago" is measured
  // against the server clock (from /api/status), not the browser's.
  const parseTs = (ts) => new Date(String(ts).slice(0, 23));
  function serverNow() {
    const s = state.status;
    return s ? parseTs(s.server_time).getTime() + (Date.now() - state.statusAt) : Date.now();
  }
  function ago(ts) {
    if (!ts) return '–';
    const sec = Math.max(0, Math.round((serverNow() - parseTs(ts).getTime()) / 1000));
    if (sec < 90) return `${sec}s ago`;
    if (sec < 5400) return `${Math.round(sec / 60)} min ago`;
    if (sec < 172800) return `${Math.round(sec / 3600)} h ago`;
    return `${Math.round(sec / 86400)} d ago`;
  }

  function toast(msg, kind) {
    const box = $('#toast');
    const el = h('div', { class: kind === 'err' ? 'err' : '' }, msg);
    box.append(el);
    setTimeout(() => el.remove(), kind === 'err' ? 8000 : 3500);
  }

  function download(name, blob) {
    const a = h('a', { href: URL.createObjectURL(blob), download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  // ------------------------------------------------------------------ state and API
  const state = { token: null, status: null, statusAt: 0, bands: [], pendingBands: null };
  try { state.token = sessionStorage.getItem('scout-token'); } catch (e) { /* ignore */ }
  (function takeTokenFromHash() {
    const m = /[#&?]token=([^&]+)/.exec(location.hash);
    if (!m) return;
    state.token = decodeURIComponent(m[1]);
    try { sessionStorage.setItem('scout-token', state.token); } catch (e) { /* ignore */ }
    history.replaceState(null, '', location.pathname + '#/live'); // keep the token out of history
  })();

  class ApiError extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }

  function qs(params) {
    const u = new URLSearchParams();
    for (const [k, v] of Object.entries(params || {})) {
      if (Array.isArray(v)) v.forEach((x) => u.append(k, x));
      else if (v !== null && v !== undefined && v !== '' && v !== false) u.append(k, v === true ? 'true' : v);
    }
    const s = u.toString();
    return s ? '?' + s : '';
  }

  let tokenPrompt = null;
  function askToken(rejected) {
    if (tokenPrompt) return tokenPrompt;
    tokenPrompt = new Promise((resolve) => {
      const input = h('input', { type: 'password', autocomplete: 'off', required: true, 'aria-label': 'Access token' });
      const dlg = h('dialog', null,
        h('form', {
          class: 'form-grid',
          onsubmit: (ev) => {
            ev.preventDefault();
            state.token = input.value.trim();
            try { sessionStorage.setItem('scout-token', state.token); } catch (e) { /* ignore */ }
            dlg.close();
            dlg.remove();
            tokenPrompt = null;
            resolve();
          },
        },
        h('h2', null, 'Access token'),
        h('p', { class: 'muted' }, rejected ? 'That token was not accepted. Try again.' : 'This server requires an access token.'),
        input,
        h('div', { class: 'row end' }, h('button', { class: 'primary', type: 'submit' }, 'Continue'))));
      dlg.addEventListener('cancel', (e) => e.preventDefault());
      document.body.append(dlg);
      dlg.showModal();
      input.focus();
    });
    return tokenPrompt;
  }

  async function api(path, opts) {
    opts = opts || {};
    const headers = { Accept: 'application/json' };
    if (state.token) headers.Authorization = 'Bearer ' + state.token;
    if (opts.body !== undefined) headers['Content-Type'] = 'application/json';
    const res = await fetch(path + qs(opts.params), {
      method: opts.method || 'GET',
      headers,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
      signal: opts.signal,
    });
    if (res.status === 401) {
      await askToken(!!state.token);
      return api(path, opts);
    }
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { /* not JSON */ }
    if (!res.ok) throw new ApiError(res.status, (data && data.detail) || res.statusText || 'request failed');
    return data;
  }

  async function downloadApi(path, params, filename) {
    const headers = state.token ? { Authorization: 'Bearer ' + state.token } : {};
    const res = await fetch(path + qs(params), { headers });
    if (!res.ok) throw new ApiError(res.status, 'download failed');
    download(filename, await res.blob());
  }

  // ------------------------------------------------------------------ status (header pill, shared by views)
  const statusListeners = new Set();
  async function refreshStatus() {
    try {
      const s = await api('/api/status');
      state.status = s;
      state.statusAt = Date.now();
      renderPill(s);
      statusListeners.forEach((fn) => fn(s));
    } catch (e) {
      renderPill(null);
    }
  }

  function renderPill(s) {
    const el = $('#scanner-pill');
    let text = 'server unreachable';
    let cls = 'pill error';
    if (s) {
      const sc = s.scanner || {};
      cls = 'pill';
      if (sc.state === 'running') {
        text = 'Scanning' + (sc.mode === 'run' ? (sc.phase === 'capturing' ? ' · recording IQ' : ' · with IQ capture') : '');
        cls += ' running';
      } else if (sc.state === 'starting' || sc.state === 'stopping') {
        text = sc.state === 'starting' ? 'Starting…' : 'Stopping…';
        cls += ' busy';
      } else if (sc.last_exit && sc.last_exit.code !== 0) {
        text = `Scanner failed (exit ${sc.last_exit.code})`;
        cls += ' error';
      } else {
        text = s.db_exists ? 'Scanner stopped' : 'No database yet';
      }
    }
    el.className = cls;
    el.textContent = text;
  }

  async function loadBands() {
    try { state.bands = (await api('/api/bands')).bands; } catch (e) { state.bands = []; }
  }

  // ------------------------------------------------------------------ small widgets
  // 503 means "no database file yet", which is normal before the first scan, not a failure
  function errorNote(e) {
    return e && e.status === 503
      ? h('div', { class: 'empty' }, 'No database yet. Start a scan on the Live tab and the data will show up here.')
      : h('div', { class: 'note err' }, e && e.message ? e.message : String(e));
  }

  function stat(value, label) { return h('div', { class: 'stat' }, h('b', null, value), h('span', null, label)); }

  function renderStats(el, s) {
    const c = s.counts || {};
    fill(el,
      stat(fmtInt(c.signals), 'signals stored'),
      stat(fmtInt(c.observations), 'observations'),
      stat(fmtInt(s.sweep_count), 'sweeps processed'),
      stat(s.last_signal_at ? ago(s.last_signal_at) : '–', 'last signal seen'),
      stat(fmtBytes(s.db_bytes), 'database size'),
    );
  }

  function sparkline(values) {
    const W = 300, H = 60, pad = 4;
    const s = svg('svg', { class: 'spark', viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', role: 'img', 'aria-label': 'SNR over time' });
    s.append(svg('line', { x1: 0, y1: H - 1, x2: W, y2: H - 1 }));
    if (values.length < 2) return s;
    const min = Math.min(...values), span = Math.max(...values) - min || 1;
    const pts = values.map((v, i) => `${(pad + (i / (values.length - 1)) * (W - 2 * pad)).toFixed(1)},${(H - pad - ((v - min) / span) * (H - 2 * pad)).toFixed(1)}`);
    s.append(svg('path', { d: 'M' + pts.join('L') }));
    return s;
  }

  function barChart(items, key, label) {
    const W = 400, H = 90, base = 80;
    const s = svg('svg', { class: 'bars', viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none', role: 'img', 'aria-label': label });
    s.append(svg('line', { x1: 0, y1: base, x2: W, y2: base }));
    const max = Math.max(1, ...items.map((i) => i[key]));
    const bw = W / Math.max(items.length, 1);
    items.forEach((it, i) => {
      const hgt = Math.max(1, (it[key] / max) * (base - 6));
      const r = svg('rect', { x: (i * bw + 1).toFixed(1), y: (base - hgt).toFixed(1), width: Math.max(1, bw - 2).toFixed(1), height: hgt.toFixed(1) });
      r.append(svg('title', {}, `${it.t}: ${it[key]}`));
      s.append(r);
    });
    return s;
  }

  const QUICK_BANDS = ['433', '868', '2.4ghz', '5.8ghz', 'pmr446', 'vhf', 'uhf'];

  // Band chooser used by the scanner form and the signal filters: quick buttons, a menu with
  // every preset, and a free START:STOP range in MHz.
  function bandPicker(initial, onChange) {
    let selected = initial.slice();
    const byKey = Object.fromEntries(state.bands.map((b) => [b.key, b]));
    const chipsEl = h('div', { class: 'chips' });
    const more = h('select', { 'aria-label': 'Add a band', onchange: () => { if (more.value) add(more.value); more.value = ''; } }, option('', 'More bands…'));
    const groups = {};
    state.bands.forEach((b) => { (groups[b.group] = groups[b.group] || []).push(b); });
    for (const [g, list] of Object.entries(groups)) more.append(h('optgroup', { label: g }, list.map((b) => option(b.key, `${b.name} (${rangeLabel(b)})`))));
    const custom = h('input', { type: 'text', placeholder: 'MHz, e.g. 430:440', size: 16, 'aria-label': 'Custom range in MHz', onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); addCustom(); } } });

    function change() { render(); onChange(selected.slice()); }
    function toggle(key) { selected = selected.includes(key) ? selected.filter((k) => k !== key) : selected.concat(key); change(); }
    function add(key) { if (!selected.includes(key)) selected = selected.concat(key); change(); }
    function addCustom() {
      const v = custom.value.trim().replace(/\s+/g, '');
      if (!/^\d+(\.\d+)?:\d+(\.\d+)?$/.test(v)) { toast('Use START:STOP in MHz, for example 430:440', 'err'); return; }
      custom.value = '';
      add(v);
    }
    function chip(key) {
      const b = byKey[key];
      const on = selected.includes(key);
      return h('button', { type: 'button', class: 'chip' + (on ? ' on' : ''), 'aria-pressed': String(on), title: b ? `${b.name} · ${rangeLabel(b)}` : `${key} MHz`, onclick: () => toggle(key) }, key, on && !QUICK_BANDS.includes(key) ? ' ×' : '');
    }
    function render() {
      const quick = QUICK_BANDS.filter((k) => byKey[k]);
      fill(chipsEl, ...quick.map(chip), ...selected.filter((k) => !quick.includes(k)).map(chip));
    }
    render();
    return {
      el: h('div', { class: 'form-grid' }, chipsEl, h('div', { class: 'row' }, more, custom, h('button', { type: 'button', class: 'small', onclick: addCustom }, 'Add range'))),
      get: () => selected.slice(),
      set: (arr) => { selected = arr.slice(); render(); },
    };
  }

  function identCell(r) {
    if (!r.ident_name) return h('span', { class: 'muted' }, 'unidentified');
    if (r.ident_source === 'artemis' && r.ident_score != null) return [r.ident_name, ' ', h('span', { class: 'badge ok' }, Math.round(r.ident_score) + '%')];
    return [r.ident_name, ' ', h('span', { class: 'badge' }, 'bandplan')];
  }

  // ------------------------------------------------------------------ LIVE view
  const FORM_DEFAULTS = {
    mode: 'scan', bands: [], lna: 24, vga: 20, amp: false, snr: 10, min_hits: 3, region_keywords: '', duration: '',
    scan_seconds: 600, capture_seconds: 5, capture_max: 3, capture_which: 'unidentified', capture_min_snr: 15, cycles: 0,
    bin_width: 100000, expire: 20, obs_interval: 30,
  };
  function loadForm() {
    let saved = {};
    try { saved = JSON.parse(store.get('scout-form') || '{}'); } catch (e) { saved = {}; }
    const f = Object.assign({}, FORM_DEFAULTS, saved);
    if (state.pendingBands) { f.bands = state.pendingBands; state.pendingBands = null; }
    return f;
  }

  const numInput = (value, min, max, step, extra) => h('input', Object.assign({ type: 'number', value: value, min, max, step: step || 1 }, extra || {}));
  function selectOf(values, current) {
    const el = h('select', null, values.map((v) => option(v)));
    el.value = String(current);
    return el;
  }

  function startForm() {
    const f = loadForm();
    const els = {};
    const picker = bandPicker(f.bands, () => {});
    const modeRadios = ['scan', 'run'].map((v) => h('label', { class: 'inline' }, h('input', { type: 'radio', name: 'mode', value: v, checked: f.mode === v, onchange: syncMode }), v === 'scan' ? 'Scan only' : 'Scan and record IQ of new signals'));
    els.lna = selectOf([0, 8, 16, 24, 32, 40], f.lna);
    els.vga = selectOf(Array.from({ length: 32 }, (_, i) => i * 2), f.vga);
    els.amp = h('input', { type: 'checkbox', checked: f.amp });
    els.snr = numInput(f.snr, 1, 60, 0.5);
    els.min_hits = numInput(f.min_hits, 1, 50);
    els.region_keywords = h('input', { type: 'text', value: f.region_keywords, placeholder: 'greece,cyprus', maxLength: 100, size: 16 });
    els.duration = numInput(f.duration, 1, 86400, 1, { placeholder: 'no limit' });
    els.scan_seconds = numInput(f.scan_seconds, 5, 86400);
    els.capture_seconds = numInput(f.capture_seconds, 0, 60, 1);
    els.capture_max = numInput(f.capture_max, 0, 50);
    els.capture_which = selectOf(['unidentified', 'new'], f.capture_which);
    els.capture_min_snr = numInput(f.capture_min_snr, 0, 60, 0.5);
    els.cycles = numInput(f.cycles, 0, 10000);
    els.bin_width = numInput(f.bin_width, 2445, 5000000, 1);
    els.expire = numInput(f.expire, 1, 1000);
    els.obs_interval = numInput(f.obs_interval, 1, 3600);

    const scanOnly = h('div', { class: 'fields' }, field('Duration (s)', els.duration));
    const runOnly = h('div', { class: 'fields' }, field('Scan seconds per cycle', els.scan_seconds), field('IQ seconds', els.capture_seconds), field('Max captures per cycle', els.capture_max), field('Capture which', els.capture_which));
    const output = h('div');
    const startBtn = h('button', { type: 'submit', class: 'primary' }, 'Start scanning');
    const checkBtn = h('button', { type: 'button' }, 'Check device');

    const currentMode = () => (modeRadios.some((l) => $('input', l).checked && $('input', l).value === 'run') ? 'run' : 'scan');
    function syncMode() {
      const run = currentMode() === 'run';
      runOnly.hidden = !run;
      scanOnly.hidden = run;
    }

    function collect() {
      const mode = currentMode();
      const p = {
        mode, bands: picker.get(), lna: +els.lna.value, vga: +els.vga.value, amp: els.amp.checked,
        snr: +els.snr.value, min_hits: +els.min_hits.value, bin_width: +els.bin_width.value,
        expire: +els.expire.value, obs_interval: +els.obs_interval.value,
      };
      if (els.region_keywords.value.trim()) p.region_keywords = els.region_keywords.value.trim();
      if (mode === 'scan') {
        if (els.duration.value) p.duration = +els.duration.value;
      } else {
        Object.assign(p, {
          scan_seconds: +els.scan_seconds.value, cycles: +els.cycles.value, capture_seconds: +els.capture_seconds.value,
          capture_max: +els.capture_max.value, capture_which: els.capture_which.value, capture_min_snr: +els.capture_min_snr.value,
        });
      }
      return p;
    }

    const form = h('form', {
      class: 'form-grid',
      novalidate: true, // the server validates and explains; a native popup cannot show inside a closed <details>
      onsubmit: async (ev) => {
        ev.preventDefault();
        const params = collect();
        store.set('scout-form', JSON.stringify(Object.assign({}, f, params, { duration: params.duration || '' })));
        startBtn.disabled = true;
        startBtn.textContent = 'Starting…';
        try {
          await api('/api/scanner/start', { method: 'POST', body: params });
          toast('Scanner started');
        } catch (e) {
          toast(e.message, 'err');
        } finally {
          startBtn.disabled = false;
          startBtn.textContent = 'Start scanning';
          refreshStatus();
        }
      },
    },
    h('fieldset', null, h('legend', null, 'Mode'), h('div', { class: 'row' }, modeRadios)),
    h('fieldset', null, h('legend', null, 'Frequencies'), picker.el,
      h('p', { class: 'muted' }, 'Nothing selected scans the whole 1–6000 MHz range. Narrow bands are revisited much faster, which helps with short bursts.')),
    h('fieldset', null, h('legend', null, 'Receiver and detection'),
      h('div', { class: 'fields' }, field('LNA gain (dB)', els.lna), field('VGA gain (dB)', els.vga),
        h('label', { class: 'inline' }, els.amp, 'RF amp (+14 dB)'), field('SNR threshold (dB)', els.snr), field('Hits to confirm', els.min_hits), field('Region keywords', els.region_keywords))),
    scanOnly, runOnly,
    h('details', null, h('summary', null, 'Advanced'),
      h('div', { class: 'fields' }, field('Bin width (Hz)', els.bin_width), field('Expire (sweeps)', els.expire), field('Observation interval (s)', els.obs_interval),
        field('Capture min SNR (dB)', els.capture_min_snr), field('Run cycles (0 = forever)', els.cycles))),
    h('div', { class: 'row' }, startBtn, checkBtn, h('span', { class: 'muted' }, 'Receive only. The tool never transmits.')),
    output);

    checkBtn.addEventListener('click', async () => {
      checkBtn.disabled = true;
      try {
        const r = await api('/api/scanner/check', { method: 'POST' });
        fill(output, h('pre', { class: 'out' }, (r.ok ? '' : 'HackRF not detected.\n') + r.output));
      } catch (e) {
        toast(e.message, 'err');
      } finally {
        checkBtn.disabled = false;
      }
    });
    syncMode();
    return form;
  }

  function runningPanel(s) {
    const sc = s.scanner;
    const busy = sc.state !== 'running';
    const stopBtn = h('button', { class: 'danger', disabled: busy || !s.control_enabled, onclick: async () => {
      stopBtn.disabled = true;
      stopBtn.textContent = 'Stopping…';
      try { await api('/api/scanner/stop', { method: 'POST' }); toast('Scanner stopped'); } catch (e) { toast(e.message, 'err'); }
      refreshStatus();
    } }, 'Stop scanning');
    return h('div', { class: 'form-grid' },
      h('div', { class: 'row' },
        h('span', { class: 'badge ok' }, sc.state), h('span', null, `${sc.mode || 'scan'}${sc.phase ? ' · ' + sc.phase : ''}`),
        h('span', { class: 'muted' }, `pid ${sc.pid || '?'} · started ${fmtTime(sc.started_at)} `, h('span', { id: 'uptime' }, sc.started_at ? '(' + ago(sc.started_at) + ')' : '')),
        h('span', { class: 'spacer' }), s.control_enabled ? stopBtn : null),
      !s.control_enabled ? h('p', { class: 'note' }, 'This scanner was not started from here and the web server runs read-only, so it cannot be stopped from the browser (press Ctrl-C in its terminal, or restart the server with --allow-control).') : null,
      sc.argv ? h('div', null, h('span', { class: 'muted' }, 'command: '), h('code', { class: 'cmd' }, 'hackrf-scout ' + sc.argv.join(' '))) : null,
      sc.db ? h('div', { class: 'muted' }, 'writing to ', h('code', null, sc.db)) : null);
  }

  function controlSig(s) {
    const sc = s.scanner || {};
    return [s.control_enabled, sc.state, sc.pid, sc.phase, sc.last_exit && sc.last_exit.at].join('|');
  }

  function renderControl(card, s) {
    const sc = s.scanner || {};
    fill(card, h('h2', null, 'Scanner'));
    if (sc.state === 'unsupported') { card.append(h('p', { class: 'note' }, sc.message || 'Start/stop is not available on this platform.')); return; }
    if (['running', 'starting', 'stopping'].includes(sc.state)) { card.append(runningPanel(s)); return; }
    if (sc.last_exit && sc.last_exit.code !== 0) {
      card.append(h('div', { class: 'note err' }, `The last scanner run ended with exit code ${sc.last_exit.code} at ${fmtTime(sc.last_exit.at)}.`,
        sc.last_exit.output_tail ? h('pre', { class: 'out' }, sc.last_exit.output_tail) : null));
    }
    if (s.control_enabled) card.append(startForm());
    else {
      card.append(h('p', { class: 'note' }, 'Start/stop from the browser is off. Restart the web server with ', h('code', null, 'hackrf-scout web --allow-control'),
        ' to enable it (a token is generated and printed in the terminal). Meanwhile you can watch scans started elsewhere, and everything below is live.'));
    }
  }

  function buildLog(signal) {
    const LEVELS = { info: 0, warn: 1, error: 2 };
    const body = h('div', { class: 'log', role: 'log', 'aria-label': 'Scanner log', tabindex: 0 });
    const dot = h('span', { class: 'dot' });
    const connText = h('span', null, 'connecting');
    const levelSel = h('select', { 'aria-label': 'Minimum level', onchange: refilter }, option('info', 'All levels'), option('warn', 'Warnings and errors'), option('error', 'Errors only'));
    const textFilter = h('input', { type: 'search', placeholder: 'Filter text', 'aria-label': 'Filter log text', oninput: refilter, size: 18 });
    const follow = h('input', { type: 'checkbox', checked: true });
    const rows = [];
    let last = 0;

    const passes = (it) => (LEVELS[it.level] || 0) >= LEVELS[levelSel.value] && (!textFilter.value || (it.msg + ' ' + it.source).toLowerCase().includes(textFilter.value.toLowerCase()));
    function refilter() { rows.forEach((r) => { r.el.hidden = !passes(r.it); }); if (follow.checked) body.scrollTop = body.scrollHeight; }
    function add(items) {
      for (const it of items) {
        const el = h('div', { class: `logrow lvl-${it.level}` }, h('span', { class: 't' }, (it.ts || '').slice(11, 23)),
          h('span', { class: 'l' }, (it.level || '').toUpperCase().slice(0, 4)), h('span', { class: 's' }, it.source), h('span', { class: 'm' }, it.msg));
        el.hidden = !passes(it);
        body.append(el);
        rows.push({ el, it });
        last = Math.max(last, it.id);
      }
      while (rows.length > 1500) rows.shift().el.remove();
      if (items.length && follow.checked) body.scrollTop = body.scrollHeight;
    }
    function setConn(live) {
      dot.className = 'dot' + (live ? ' live' : '');
      connText.textContent = live ? 'live' : 'reconnecting…';
    }

    async function stream() {
      let backoff = 1000;
      while (!signal.aborted) {
        try {
          const headers = state.token ? { Authorization: 'Bearer ' + state.token } : {};
          const res = await fetch('/api/log/stream' + qs({ after: last }), { headers, signal });
          if (res.status === 401) { await askToken(!!state.token); continue; }
          if (!res.ok) throw new Error('HTTP ' + res.status);
          setConn(true);
          backoff = 1000;
          const reader = res.body.getReader();
          const dec = new TextDecoder();
          let buf = '';
          for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            let i;
            while ((i = buf.indexOf('\n\n')) >= 0) {
              const block = buf.slice(0, i);
              buf = buf.slice(i + 2);
              const data = block.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trim()).join('');
              if (data) { try { add([JSON.parse(data)]); } catch (e) { /* skip a malformed event */ } }
            }
          }
        } catch (e) {
          if (signal.aborted) return;
        }
        setConn(false);
        await sleep(backoff);
        backoff = Math.min(backoff * 2, 15000);
      }
    }

    (async () => {
      try {
        const r = await api('/api/log', { params: { tail: 300 }, signal });
        add(r.items);
      } catch (e) {
        if (signal.aborted) return;
      }
      stream();
    })();

    const el = h('section', { class: 'card log-card' },
      h('div', { class: 'row' }, h('h2', null, 'Log'), h('span', { class: 'muted' }, dot, connText), h('span', { class: 'spacer' }),
        levelSel, textFilter, h('label', { class: 'inline' }, follow, 'Follow'),
        h('button', { type: 'button', class: 'small', onclick: () => { rows.splice(0).forEach((r) => r.el.remove()); } }, 'Clear view'),
        h('button', { type: 'button', class: 'small', onclick: () => download('scout-log.txt', new Blob([rows.map((r) => `${r.it.ts} ${r.it.level} [${r.it.source}] ${r.it.msg}`).join('\n') + '\n'], { type: 'text/plain' })) }, 'Save')),
      body);
    return el;
  }

  function viewLive(root) {
    const ac = new AbortController();
    const statsEl = h('div', { class: 'stats' });
    const ctrl = h('section', { class: 'card' });
    let sig = null;
    const onStatus = (s) => {
      renderStats(statsEl, s);
      const now = controlSig(s);
      if (now !== sig) { sig = now; renderControl(ctrl, s); }
      const up = $('#uptime');
      if (up && s.scanner && s.scanner.started_at) up.textContent = '(' + ago(s.scanner.started_at) + ')';
    };
    root.append(statsEl, ctrl, buildLog(ac.signal));
    statusListeners.add(onStatus);
    if (state.status) onStatus(state.status);
    return () => { ac.abort(); statusListeners.delete(onStatus); };
  }

  // ------------------------------------------------------------------ SIGNALS view
  function viewSignals(root, params) {
    const f = {
      bands: params.getAll('band'), unid: params.get('unid') === '1', hits: params.get('hits') || '', snr: params.get('snr') || '',
      q: params.get('q') || '', mode: params.get('mode') || 'overlap', sort: params.get('sort') || 'center_hz', order: params.get('order') || '',
      page: Math.max(1, +params.get('page') || 1), size: +params.get('size') || 50,
    };
    let seq = 0, timer = null, refreshTimer = null;
    const results = h('div');
    const picker = bandPicker(f.bands, (b) => { f.bands = b; f.page = 1; load(); });
    const q = h('input', { type: 'search', value: f.q, placeholder: 'Search name, service, notes or #id', 'aria-label': 'Search signals', size: 28, oninput: () => { clearTimeout(timer); timer = setTimeout(() => { f.q = q.value; f.page = 1; load(); }, 300); } });
    const unid = h('input', { type: 'checkbox', checked: f.unid, onchange: () => { f.unid = unid.checked; f.page = 1; load(); } });
    const hits = h('input', { type: 'number', min: 0, value: f.hits, placeholder: '0', onchange: () => { f.hits = hits.value; f.page = 1; load(); } });
    const snr = h('input', { type: 'number', min: 0, step: 0.5, value: f.snr, placeholder: '0', onchange: () => { f.snr = snr.value; f.page = 1; load(); } });
    const mode = h('select', { 'aria-label': 'Band matching', onchange: () => { f.mode = mode.value; load(); } }, option('overlap', 'Any overlap with band'), option('center', 'Centre frequency inside band'));
    mode.value = f.mode;
    const auto = h('input', { type: 'checkbox', onchange: () => { clearInterval(refreshTimer); if (auto.checked) refreshTimer = setInterval(() => { if (!$('.drawer')) load(true); }, 5000); } });

    const apiParams = () => ({ band: f.bands, unidentified: f.unid, min_hits: f.hits || null, min_snr: f.snr || null, q: f.q.trim() || null, mode: f.mode });

    function syncHash() {
      const u = new URLSearchParams();
      f.bands.forEach((b) => u.append('band', b));
      if (f.unid) u.set('unid', '1');
      if (f.hits) u.set('hits', f.hits);
      if (f.snr) u.set('snr', f.snr);
      if (f.q) u.set('q', f.q);
      if (f.mode !== 'overlap') u.set('mode', f.mode);
      if (f.sort !== 'center_hz') u.set('sort', f.sort);
      if (f.order) u.set('order', f.order);
      if (f.page > 1) u.set('page', f.page);
      if (f.size !== 50) u.set('size', f.size);
      history.replaceState(null, '', '#/signals' + (u.toString() ? '?' + u : ''));
    }

    const COLS = [
      { key: 'id', label: 'ID', sort: 'id', cls: 'num', cell: (r) => r.id },
      { key: 'mhz', label: 'MHz', sort: 'center_hz', cls: 'num mono', cell: (r) => fmtMHz(r.center_hz) },
      { key: 'bw', label: 'BW kHz', sort: 'bandwidth_hz', cls: 'num', cell: (r) => fmtKHz(r.bandwidth_hz) },
      { key: 'snr', label: 'Max SNR', sort: 'max_snr', cls: 'num', cell: (r) => fmtNum(r.max_snr) },
      { key: 'db', label: 'Peak dB', sort: 'max_db', cls: 'num', cell: (r) => fmtNum(r.max_db) },
      { key: 'hits', label: 'Hits', sort: 'hits', cls: 'num', cell: (r) => fmtInt(r.hits) },
      { key: 'duty', label: 'Duty %', cls: 'num', cell: (r) => fmtNum(r.duty_pct) },
      { key: 'last', label: 'Last seen', sort: 'last_seen', cell: (r) => fmtTime(r.last_seen) },
      { key: 'ident', label: 'Identification', sort: 'ident_name', cls: 'wrap', cell: identCell },
      { key: 'svc', label: 'Service', cls: 'cut', cell: (r) => (r.service || '').split(';')[0] },
    ];

    async function load(quiet) {
      const mine = ++seq;
      syncHash();
      try {
        const d = await api('/api/signals', { params: Object.assign(apiParams(), { sort: f.sort, order: f.order || null, page: f.page, page_size: f.size }) });
        if (mine !== seq) return;
        render(d);
      } catch (e) {
        if (mine !== seq || quiet) return;
        fill(results, errorNote(e));
      }
    }

    function render(d) {
      const pages = Math.max(1, Math.ceil(d.total / d.page_size));
      if (f.page > pages) { f.page = pages; load(); return; }
      const head = h('tr', null, COLS.map((c) => h('th', {
        scope: 'col', class: [c.cls && c.cls.startsWith('num') ? 'num' : '', c.sort ? 'sortable' : ''].filter(Boolean).join(' '),
        'aria-sort': c.sort === d.sort ? (d.order === 'asc' ? 'ascending' : 'descending') : null,
        onclick: c.sort ? () => { f.order = f.sort === c.sort && d.order === 'desc' ? 'asc' : f.sort === c.sort ? 'desc' : ''; f.sort = c.sort; f.page = 1; load(); } : null,
      }, c.label, c.sort === d.sort ? (d.order === 'asc' ? ' ▲' : ' ▼') : '')));
      const body = d.items.map((r) => h('tr', { class: 'click', tabindex: 0, onclick: () => openSignal(r.id), onkeydown: (e) => { if (e.key === 'Enter') openSignal(r.id); } },
        COLS.map((c) => h('td', { class: c.cls || '', title: c.key === 'svc' ? r.service || '' : null }, c.cell(r)))));
      const pager = h('div', { class: 'pager' },
        h('button', { disabled: f.page <= 1, onclick: () => { f.page--; load(); } }, '‹ Prev'),
        h('span', null, `Page ${d.page} of ${pages} · ${fmtInt(d.total)} signals`),
        h('button', { disabled: f.page >= pages, onclick: () => { f.page++; load(); } }, 'Next ›'),
        h('span', { class: 'spacer' }),
        h('label', { class: 'inline' }, 'Rows', (() => { const s = h('select', { onchange: () => { f.size = +s.value; f.page = 1; load(); } }, [25, 50, 100, 200].map((n) => option(n))); s.value = String(f.size); return s; })()));
      fill(results, d.items.length
        ? h('div', { class: 'table-wrap' }, h('table', null, h('thead', null, head), h('tbody', null, body)))
        : h('div', { class: 'empty' }, 'No signals match these filters.'), pager);
    }

    const exportBtn = (fmt) => h('button', { type: 'button', class: 'small', onclick: () => downloadApi('/api/signals/export', Object.assign(apiParams(), { format: fmt }), `signals.${fmt}`).catch((e) => toast(e.message, 'err')) }, 'Export ' + fmt.toUpperCase());

    root.append(h('section', { class: 'card form-grid' },
      h('div', { class: 'row' }, h('h2', null, 'Signals'), h('span', { class: 'spacer' }), h('label', { class: 'inline' }, auto, 'Auto-refresh'), exportBtn('csv'), exportBtn('json')),
      picker.el,
      h('div', { class: 'fields' }, field('Search', q), h('label', { class: 'inline' }, unid, 'Unidentified only'), field('Min hits', hits), field('Min SNR (dB)', snr), field('Band matching', mode))),
    h('section', { class: 'card' }, results));
    load();
    return () => { clearTimeout(timer); clearInterval(refreshTimer); seq++; closeDrawer(); };
  }

  // ------------------------------------------------------------------ signal drawer
  function closeDrawer() {
    document.querySelectorAll('.drawer, .backdrop').forEach((e) => e.remove());
    document.removeEventListener('keydown', onDrawerKey);
  }
  function onDrawerKey(e) { if (e.key === 'Escape') closeDrawer(); }

  async function openSignal(id) {
    closeDrawer();
    const back = h('div', { class: 'backdrop', onclick: closeDrawer });
    const closeBtn = h('button', { onclick: closeDrawer, 'aria-label': 'Close' }, 'Close');
    const drawer = h('aside', { class: 'drawer', role: 'dialog', 'aria-label': `Signal ${id}` }, h('div', { class: 'row' }, h('h2', null, `Signal #${id}`), h('span', { class: 'spacer' }), closeBtn), h('p', { class: 'muted' }, 'Loading…'));
    document.body.append(back, drawer);
    document.addEventListener('keydown', onDrawerKey);
    closeBtn.focus();
    let d;
    try { d = await api('/api/signals/' + id); } catch (e) { drawer.append(errorNote(e)); return; }
    if (!drawer.isConnected) return;

    const facts = [
      ['Bandwidth', fmtKHz(d.bandwidth_hz) + ' kHz'], ['Peak at', fmtMHz(d.peak_hz || d.center_hz) + ' MHz'],
      ['Max level', fmtNum(d.max_db) + ' dB'], ['Avg level', fmtNum(d.avg_db) + ' dB'],
      ['Max SNR', fmtNum(d.max_snr) + ' dB'], ['Last SNR', fmtNum(d.last_snr) + ' dB'],
      ['Hits', fmtInt(d.hits)], ['Duty cycle', fmtNum(d.duty_pct) + ' %'],
      ['First seen', fmtTime(d.first_seen)], ['Last seen', fmtTime(d.last_seen)],
      ['Identified', fmtTime(d.ident_at)], ['IQ captured', d.captured ? 'yes' : 'no'],
    ];
    const snrs = d.observations.slice().reverse().map((o) => o.snr_db).filter((v) => v != null);
    fill(drawer,
      h('div', { class: 'row' }, h('h2', null, `#${d.id} · ${fmtMHz(d.center_hz)} MHz`), h('span', { class: 'spacer' }), closeBtn),
      h('div', null, d.ident_name ? [h('b', null, d.ident_name), ' ', identCell(d).slice(1)] : h('span', { class: 'muted' }, 'Not identified'), d.ident_url ? [' · ', extLink(d.ident_url, 'SigID Wiki')] : null),
      d.label || d.notes ? h('p', null, d.label ? h('b', null, d.label + ' ') : null, d.notes) : null,
      h('dl', { class: 'facts' }, facts.map(([k, v]) => [h('dt', null, k), h('dd', null, v)])),
      h('div', null, h('h3', null, 'Bands'), d.bands.length
        ? h('div', { class: 'chips' }, d.bands.map((k) => h('a', { class: 'chip', href: '#/signals?band=' + encodeURIComponent(k), onclick: closeDrawer }, k)))
        : h('span', { class: 'muted' }, 'outside every preset band')),
      d.services.length ? h('div', null, h('h3', null, 'Bandplan says'), h('ul', null, d.services.map((s) => h('li', null, s)))) : null,
      d.candidates.length ? h('div', null, h('h3', null, 'Candidate matches'), h('div', { class: 'table-wrap' }, h('table', null,
        h('thead', null, h('tr', null, ['Name', 'Score', 'Modulation'].map((t) => h('th', { scope: 'col' }, t)))),
        h('tbody', null, d.candidates.map((c) => h('tr', null, h('td', { class: 'wrap' }, c.url ? extLink(c.url, c.name) : c.name, c.broad ? [' ', h('span', { class: 'badge warn' }, 'broad')] : null),
          h('td', { class: 'num' }, Math.round(c.score) + '%'), h('td', null, (c.modulations || []).join(', ') || '?'))))))) : null,
      h('div', null, h('h3', null, `Signal strength over time (${fmtInt(d.observation_count)} observations${d.observation_count > d.observations.length ? `, last ${d.observations.length} shown` : ''})`), sparkline(snrs)),
      d.captures.length ? h('div', null, h('h3', null, 'IQ captures'), h('ul', null, d.captures.map((c) => h('li', null, h('code', null, c.path), ` · ${fmtTime(c.ts)} · ${fmtNum(c.seconds, 0)} s @ ${fmtNum(c.sample_rate / 1e6, 0)} Msps`)))) : null,
      h('div', null, h('h3', null, 'Recent observations'), h('div', { class: 'table-wrap' }, h('table', null,
        h('thead', null, h('tr', null, [['Time', ''], ['MHz', 'num'], ['BW kHz', 'num'], ['Peak dB', 'num'], ['SNR dB', 'num']].map(([t, c]) => h('th', { scope: 'col', class: c }, t)))),
        h('tbody', null, d.observations.slice(0, 50).map((o) => h('tr', null, h('td', null, fmtTime(o.ts)), h('td', { class: 'num mono' }, fmtMHz(o.center_hz)),
          h('td', { class: 'num' }, fmtKHz(o.bandwidth_hz)), h('td', { class: 'num' }, fmtNum(o.peak_db)), h('td', { class: 'num' }, fmtNum(o.snr_db)))))))));
    drawer.querySelector('button').focus();
  }

  // ------------------------------------------------------------------ BANDS view
  function viewBands(root) {
    let mode = 'overlap', hideEmpty = true, seq = 0;
    const out = h('div', { class: 'form-grid' });
    const modeSel = h('select', { 'aria-label': 'Band matching', onchange: () => { mode = modeSel.value; load(); } }, option('overlap', 'Any overlap with band'), option('center', 'Centre frequency inside band'));
    const hide = h('input', { type: 'checkbox', checked: true, onchange: () => { hideEmpty = hide.checked; load(); } });

    async function load() {
      const mine = ++seq;
      try {
        const d = await api('/api/bands/summary', { params: { mode } });
        if (mine === seq) render(d.bands);
      } catch (e) {
        if (mine === seq) fill(out, errorNote(e));
      }
    }

    function render(list) {
      const shown = hideEmpty ? list.filter((b) => b.signals > 0) : list;
      if (!shown.length) { fill(out, h('div', { class: 'card empty' }, hideEmpty ? 'No signals in any preset band yet.' : 'No bands.')); return; }
      const groups = {};
      shown.forEach((b) => { (groups[b.group] = groups[b.group] || []).push(b); });
      fill(out, ...Object.entries(groups).map(([g, bs]) => h('section', { class: 'card' }, h('div', { class: 'group-title' }, g),
        h('div', { class: 'table-wrap' }, h('table', null,
          h('thead', null, h('tr', null, [['Band', ''], ['Range', ''], ['Signals', 'num'], ['Unidentified', 'num'], ['Best SNR', 'num'], ['Last seen', ''], ['', '']].map(([t, c]) => h('th', { scope: 'col', class: c }, t)))),
          h('tbody', null, bs.map(bandRow)))))));
    }

    function bandRow(b) {
      let open = null;
      const tr = h('tr', null,
        h('td', null, h('b', null, b.name), ' ', h('span', { class: 'badge' }, b.key)),
        h('td', { class: 'mono' }, rangeLabel(b)),
        h('td', { class: 'num' }, b.signals ? h('a', { href: '#/signals?band=' + encodeURIComponent(b.key) + (mode !== 'overlap' ? '&mode=' + mode : '') }, fmtInt(b.signals)) : '0'),
        h('td', { class: 'num' }, fmtInt(b.unidentified)),
        h('td', { class: 'num' }, fmtNum(b.best_snr)),
        h('td', null, b.last_seen ? ago(b.last_seen) : '–'),
        h('td', { class: 'num' }, h('button', { class: 'small', 'aria-expanded': 'false', onclick: toggle }, 'Details')));
      const frag = document.createDocumentFragment();
      frag.append(tr);
      async function toggle(ev) {
        const btn = ev.currentTarget;
        if (open) { open.remove(); open = null; btn.setAttribute('aria-expanded', 'false'); return; }
        btn.setAttribute('aria-expanded', 'true');
        const cell = h('td', { colSpan: 7 }, h('span', { class: 'muted' }, 'Loading…'));
        open = h('tr', { class: 'detail' }, cell);
        tr.after(open);
        const bucket = h('select', { 'aria-label': 'Bucket size', onchange: () => chart() }, option('hour', 'per hour'), option('minute', 'per minute'), option('day', 'per day'));
        const chartBox = h('div');
        async function chart() {
          try {
            const a = await api('/api/bands/activity', { params: { band: b.key, bucket: bucket.value, limit: 72, mode } });
            fill(chartBox, a.items.length ? [barChart(a.items, 'observations', `Observations in ${b.name}`), h('div', { class: 'muted' }, `${a.items[0].t} → ${a.items[a.items.length - 1].t}`)] : h('span', { class: 'muted' }, 'No observations recorded in this band yet.'));
          } catch (e) { fill(chartBox, errorNote(e)); }
        }
        const cmd = b.sweepable ? `hackrf-scout scan -f ${b.sweep_range}` : null;
        fill(cell, h('div', { class: 'form-grid' },
          h('div', null, h('div', { class: 'row' }, h('h3', null, 'Activity'), bucket), chartBox),
          b.strongest ? h('div', null, 'Strongest: ', h('button', { class: 'link', onclick: () => openSignal(b.strongest.id) }, `#${b.strongest.id} at ${fmtMHz(b.strongest.center_hz)} MHz`), b.strongest.ident_name ? ' · ' + b.strongest.ident_name : '') : null,
          cmd ? h('div', null, h('h3', null, 'Scan only this band'), h('p', { class: 'muted' }, 'Narrow scans revisit a band much faster, which catches short bursts that a full sweep misses.'),
            h('div', { class: 'row' }, h('code', { class: 'cmd' }, cmd),
              h('button', { class: 'small', onclick: () => { (navigator.clipboard ? navigator.clipboard.writeText(cmd) : Promise.reject()).then(() => toast('Copied'), () => toast('Select the command and copy it manually', 'err')); } }, 'Copy'),
              state.status && state.status.control_enabled ? h('button', { class: 'small primary', onclick: () => { state.pendingBands = [b.key]; location.hash = '#/live'; } }, 'Use in scanner form') : null)) : null));
        chart();
      }
      return frag;
    }

    root.append(h('section', { class: 'card' }, h('div', { class: 'row' }, h('h2', null, 'Bands'), h('span', { class: 'spacer' }), h('label', { class: 'inline' }, hide, 'Hide empty bands'), modeSel),
      h('p', { class: 'muted' }, 'Counts of stored signals per band. Add your own bands in ', h('code', null, '~/.hackrf-scout/bands.json'), '.')), out);
    load();
    return () => { seq++; };
  }

  // ------------------------------------------------------------------ DATA view (raw tables)
  function viewData(root, params) {
    let table = params.get('table') || 'signals', sort = null, order = 'desc', page = 1, seq = 0;
    const list = h('div', { class: 'tablist' });
    const out = h('div');

    function syncHash() { history.replaceState(null, '', '#/data?table=' + encodeURIComponent(table)); }

    async function loadTables() {
      try {
        const d = await api('/api/tables');
        fill(list, ...d.tables.map((t) => h('button', { 'aria-pressed': String(t.name === table), onclick: () => { table = t.name; sort = null; order = 'desc'; page = 1; loadTables(); load(); } }, t.name, h('span', null, fmtInt(t.rows)))));
      } catch (e) { fill(list, errorNote(e)); }
    }

    async function load() {
      const mine = ++seq;
      syncHash();
      try {
        const d = await api('/api/tables/' + encodeURIComponent(table), { params: { sort, order, page, page_size: 50 } });
        if (mine !== seq) return;
        sort = d.sort;
        const pages = Math.max(1, Math.ceil(d.total / d.page_size));
        fill(out,
          h('div', { class: 'row' }, h('h2', null, d.table), h('span', { class: 'muted' }, `${fmtInt(d.total)} rows`)),
          d.rows.length ? h('div', { class: 'table-wrap' }, h('table', null,
            h('thead', null, h('tr', null, d.columns.map((c) => h('th', { scope: 'col', class: 'sortable', 'aria-sort': c === d.sort ? (d.order === 'asc' ? 'ascending' : 'descending') : null, onclick: () => { order = sort === c && order === 'desc' ? 'asc' : 'desc'; sort = c; page = 1; load(); } }, c, c === d.sort ? (d.order === 'asc' ? ' ▲' : ' ▼') : '')))),
            h('tbody', null, d.rows.map((r) => h('tr', null, r.map((v) => h('td', { class: 'cut' + (typeof v === 'number' ? ' num' : ''), title: v == null ? null : String(v) }, v == null ? h('span', { class: 'muted' }, 'NULL') : typeof v === 'number' && !Number.isInteger(v) ? +v.toFixed(4) : String(v)))))))) : h('div', { class: 'empty' }, 'This table is empty.'),
          h('div', { class: 'pager' }, h('button', { disabled: page <= 1, onclick: () => { page--; load(); } }, '‹ Prev'), h('span', null, `Page ${d.page} of ${pages}`), h('button', { disabled: page >= pages, onclick: () => { page++; load(); } }, 'Next ›')));
      } catch (e) {
        if (mine === seq) fill(out, errorNote(e));
      }
    }

    root.append(h('div', { class: 'split' }, h('section', { class: 'card' }, h('h2', null, 'Tables'), list, h('p', { class: 'muted' }, 'Read-only view of the SQLite file.')), h('section', { class: 'card' }, out)));
    loadTables();
    load();
    return () => { seq++; };
  }

  // ------------------------------------------------------------------ router
  const views = { live: viewLive, signals: viewSignals, bands: viewBands, data: viewData };
  let cleanup = null;
  async function route() {
    if (cleanup) { try { cleanup(); } catch (e) { /* ignore */ } cleanup = null; }
    closeDrawer();
    const [name, query] = location.hash.replace(/^#\/?/, '').split('?');
    const key = views[name] ? name : 'live';
    document.querySelectorAll('.tabs a').forEach((a) => { if (a.dataset.tab === key) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    const root = $('#view');
    root.replaceChildren();
    if (!state.bands.length) await loadBands();
    cleanup = views[key](root, new URLSearchParams(query || '')) || null;
  }

  window.addEventListener('hashchange', route);
  (async function init() {
    await refreshStatus();
    setInterval(() => { if (!document.hidden) refreshStatus(); }, 4000);
    route();
  })();
})();
