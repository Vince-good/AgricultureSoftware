/* 禾眼 HeYan 前端逻辑。
 *
 * 只跟 /api/* 说话，识别怎么算的一概不知 —— 这是"识别核心与界面解耦"在前端的落地。
 * 三处刻意的设计：
 *   1. 结论用 颜色 + 图标 + 文字 三条通道同时给，色弱或识字有限都不会误判；
 *   2. 置信度不足时界面绝不显示病害名，只引导重拍（后端已经强制这么做，前端不猜）；
 *   3. 方言语音缺失、模型未安装、导出失败，一律如实提示，不静默降级。
 */

(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const S = {
    boot: null,
    i18n: {},
    lang: 'zh',
    settings: {},
    profile: {},
    classes: [],
    classById: {},
    sevById: {},
    view: 'scan',
    step: 1,
    stream: null,
    facing: 'environment',
    camOn: false,
    busy: false,
    result: null,
    recents: [],
    records: [],
    recTotal: 0,
    recOffset: 0,
    recFilter: {},
    prompt: null,
  };

  const el = {};
  const els = ['cam', 'preview', 'stage', 'stage-fallback', 'stage-hint', 'scan-notice',
    'btn-shutter', 'btn-album', 'btn-flip', 'file-capture', 'file-album', 'samples-row',
    'steps', 'view-scan', 'view-result', 'view-records', 'view-settings',
    'verdict', 'r-icon', 'r-name', 'r-sev', 'r-crop', 'r-conf', 'r-conf-fill',
    'r-retake', 'r-healthy', 'r-actions', 'btn-replay', 'replay-icon', 'r-voice-text',
    'r-voice-note', 'r-badges', 'btn-retake', 'btn-record', 'r-detail', 'r-more',
    'rec-count', 'rec-filters', 'rec-list', 'rec-empty', 'rec-more',
    'exp-purpose', 'exp-fmt', 'exp-channel', 'exp-anon', 'btn-export', 'btn-integrity',
    'export-out', 'set-lang', 'set-lang-note', 'set-size', 'set-speed', 'set-auto-voice',
    'btn-voice-test', 'voice-status', 'model-kv', 'budget-list', 'paths-kv', 'about-kv',
    'pf-alias', 'pf-farmer-id', 'pf-village', 'pf-town', 'pf-county', 'pf-plot',
    'btn-profile-save', 'btn-adapters', 'btn-schemas',
    'rail-kv', 'rail-budget', 'rail-voice', 'pill-status', 'btn-lang-quick',
    'lang-quick-label', 'tab-badge', 'busy', 'sheet', 'sheet-title', 'sheet-body',
    'sheet-close', 'toast', 'player', 'main'];

  // ---------------------------------------------------------------- 工具
  function t(key, fmt) {
    let s = S.i18n[key];
    if (s == null) return key;
    if (fmt) for (const k in fmt) s = s.split('{' + k + '}').join(fmt[k]);
    return s;
  }

  function icon(name) {
    return `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
  }

  const classIcon = (c) => (c && c.icon) || 'leaf_healthy';

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g,
      (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
  }

  function pct(v) { return `${Math.round((Number(v) || 0) * 100)}%`; }

  function sevName(id) {
    const s = S.sevById[id];
    if (!s) return id || '';
    return S.lang.startsWith('en') ? s.name_en : s.name_zh;
  }

  function cropName(crop) { return t('crop_' + (crop || 'none')); }

  function stressName(stress) { return t('stress_' + (stress || 'healthy')); }

  function localTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (isNaN(d)) return String(iso);
    const p = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }

  function toast(msg, kind = 'ok', iconName) {
    el.toast.className = `toast ${kind}`;
    el.toast.innerHTML = icon(iconName || (kind === 'bad' ? 'alert' : kind === 'warn' ? 'alert' : 'check'))
      + `<span>${esc(msg)}</span>`;
    el.toast.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.toast.hidden = true; }, 4200);
  }

  function setBusy(on, label) {
    S.busy = !!on;
    el.busy.hidden = !on;
    if (label) $('.busy-box p', el.busy).textContent = label;
    else $('.busy-box p', el.busy).textContent = t('analyzing');
  }

  function openSheet(title, html) {
    el['sheet-title'].textContent = title;
    el['sheet-body'].innerHTML = html;
    el.sheet.hidden = false;
    document.body.style.overflow = 'hidden';
    $('#sheet-close').focus();
  }

  function closeSheet() {
    el.sheet.hidden = true;
    el['sheet-body'].innerHTML = '';
    document.body.style.overflow = '';
    stopVoice();
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    const ct = res.headers.get('content-type') || '';
    const body = ct.includes('application/json') ? await res.json() : await res.text();
    if (!res.ok) {
      const err = new Error((body && body.message) || `HTTP ${res.status}`);
      err.status = res.status;
      err.code = body && body.code;
      err.payload = body;
      throw err;
    }
    return body;
  }

  // ---------------------------------------------------------------- 启动
  async function boot() {
    els.forEach((k) => { el[k] = $('#' + k); });
    el.player = $('#player');
    wire();
    try {
      S.boot = await api('/api/bootstrap');
    } catch (e) {
      document.body.innerHTML = `<div style="padding:32px;font:16px system-ui">`
        + `无法连接本机服务：${esc(e.message)}</div>`;
      return;
    }
    S.i18n = S.boot.i18n || {};
    S.lang = S.boot.language || 'zh';
    S.settings = S.boot.settings || {};
    S.profile = S.boot.profile || {};
    S.classes = S.boot.classes || [];
    S.classById = Object.fromEntries(S.classes.map((c) => [c.id, c]));
    S.sevById = Object.fromEntries((S.boot.severities || []).map((s) => [s.id, s]));

    applyLang();
    renderSettings();
    renderModel();
    renderRail();
    renderRecents();
    updateBadge();
    await startCam();
    registerSw();
  }

  function applyLang() {
    document.documentElement.lang = S.lang === 'en' ? 'en' : 'zh-CN';
    document.documentElement.dataset.size = S.settings.text_size || 'large';
    $$('[data-i18n]').forEach((node) => {
      const key = node.dataset.i18n;
      const val = t(key);
      if (node.tagName === 'OPTION') node.textContent = val;
      else node.textContent = val;
    });
    $$('[data-tip-key]').forEach((node) => {
      const label = t(node.dataset.tipKey);
      node.title = label;
      if (!node.textContent.trim()) node.setAttribute('aria-label', label);
    });
    $('#lang-quick-label').textContent = ({
      zh: '普', yue: '粵', hak: '客', teochew: '潮', en: 'EN',
    })[S.lang] || '中';
    document.title = `${t('app_name')} HeYan`;
  }

  function registerSw() {
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker.register('/sw.js').catch(() => {});
    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      S.prompt = e;
    });
  }

  // ---------------------------------------------------------------- 视图切换
  function setView(name) {
    S.view = name;
    ['scan', 'result', 'records', 'settings'].forEach((v) => {
      const node = $('#view-' + v);
      const on = v === name;
      node.hidden = !on;
      node.classList.toggle('is-active', on);
    });
    $$('.tab').forEach((b) => {
      const on = b.dataset.view === (name === 'result' ? 'scan' : name);
      b.classList.toggle('is-on', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    const showSteps = name === 'scan' || name === 'result';
    el.steps.hidden = !showSteps;
    if (name === 'scan') { setStep(1); resumeCam(); } else { pauseCam(); }
    if (name === 'records') loadRecords(true);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function setStep(n) {
    S.step = n;
    $$('.step', el.steps).forEach((li, i) => {
      li.classList.toggle('is-on', i + 1 === n);
      li.classList.toggle('is-done', i + 1 < n);
    });
  }

  // ---------------------------------------------------------------- 相机
  async function startCam() {
    const md = navigator.mediaDevices;
    if (!md || !md.getUserMedia) return camFallback('sys_no_camera');
    try {
      stopStream();
      const stream = await md.getUserMedia({
        audio: false,
        video: {
          facingMode: S.facing,
          width: { ideal: 1600 }, height: { ideal: 1200 },
        },
      });
      S.stream = stream;
      S.camOn = true;
      el.cam.srcObject = stream;
      await el.cam.play().catch(() => {});
      el.cam.hidden = false;
      el['stage-fallback'].hidden = true;
      el['btn-flip'].disabled = false;
      hideScanNotice();
    } catch (e) {
      camFallback('sys_no_camera');
    }
  }

  function camFallback(msgKey) {
    S.camOn = false;
    stopStream();
    el.cam.hidden = true;
    el['stage-fallback'].hidden = false;
    el['btn-flip'].disabled = true;
    if (msgKey) showScanNotice(t(msgKey), 'warn');
  }

  function stopStream() {
    if (S.stream) { S.stream.getTracks().forEach((tr) => tr.stop()); S.stream = null; }
  }

  function pauseCam() {
    if (S.stream && S.camOn) {
      const track = S.stream.getVideoTracks()[0];
      if (track) track.enabled = false;   // 关画面省电，不释放设备，切回来秒开
    }
  }

  function resumeCam() {
    if (S.stream && S.camOn) {
      const track = S.stream.getVideoTracks()[0];
      if (track) track.enabled = true;
      el.cam.hidden = false;
      el.preview.hidden = true;
    }
  }

  function showScanNotice(text, kind) {
    el['scan-notice'].className = 'notice' + (kind === 'warn' ? ' notice-warn' : '');
    el['scan-notice'].innerHTML = icon('alert') + `<span>${esc(text)}</span>`;
    el['scan-notice'].hidden = false;
  }

  function hideScanNotice() { el['scan-notice'].hidden = true; }

  async function shoot() {
    if (S.busy) return;
    if (!S.camOn || !el.cam.videoWidth) { el['file-capture'].click(); return; }
    const vw = el.cam.videoWidth, vh = el.cam.videoHeight;
    const scale = Math.min(1, 1600 / Math.max(vw, vh));
    const cv = document.createElement('canvas');
    cv.width = Math.round(vw * scale);
    cv.height = Math.round(vh * scale);
    cv.getContext('2d').drawImage(el.cam, 0, 0, cv.width, cv.height);
    const blob = await new Promise((r) => cv.toBlob(r, 'image/jpeg', 0.92));
    if (!blob) { toast(t('sys_error'), 'bad'); return; }
    await submit(blob, 'photo.jpg');
  }

  function showPreview(blob) {
    if (S.previewUrl) URL.revokeObjectURL(S.previewUrl);
    S.previewUrl = URL.createObjectURL(blob);
    el.preview.src = S.previewUrl;
    el.preview.hidden = false;
    el.cam.hidden = true;
  }

  function pushRecent(blob, label) {
    const url = URL.createObjectURL(blob);
    S.recents.unshift({ blob, url, label });
    S.recents = S.recents.slice(0, 6);
    renderRecents();
  }

  function renderRecents() {
    const row = el['samples-row'];
    if (!S.recents.length) {
      row.innerHTML = `<p class="samples-empty">${esc(t('history_empty'))}</p>`;
      return;
    }
    row.innerHTML = S.recents.map((r, i) => `
      <button type="button" class="sample" data-i="${i}" title="${esc(r.label)}">
        <img src="${r.url}" alt="">
        <b>${esc(r.label)}</b>
      </button>`).join('');
  }

  // ---------------------------------------------------------------- 识别
  async function submit(blob, filename) {
    if (!S.boot.model || !S.boot.model.available) {
      toast(t('sys_no_model'), 'bad', 'alert');
      return;
    }
    showPreview(blob);
    setBusy(true, t('analyzing'));
    setStep(2);
    const fd = new FormData();
    fd.append('file', blob, filename || 'photo.jpg');
    fd.append('lang', S.lang);
    fd.append('save', S.settings.save_records ? '1' : '0');
    try {
      const data = await api('/api/recognize', { method: 'POST', body: fd });
      S.result = data.result;
      renderResult(data.result);
      pushRecent(blob, S.result.advice ? S.result.advice.name : localTime(new Date().toISOString()));
      setStep(3);
      setView('result');
      if (S.settings.auto_voice) playVoice();
      updateBadge();
    } catch (e) {
      setStep(1);
      setView('scan');
      const msg = e.code === 'no_model' ? t('sys_no_model') : (e.message || t('sys_error'));
      showScanNotice(msg, 'warn');
      toast(msg, 'bad');
    } finally {
      setBusy(false);
      resumeCam();
    }
  }

  function renderResult(r) {
    const advice = r.advice || {};
    const cls = S.classById[r.class_id] || {};
    const sev = (r.severity || {}).id || 'none';
    const retake = !!r.needs_retake;

    el.verdict.dataset.sev = sev;
    $('#r-icon use').setAttribute('href', '#i-' + (retake ? 'retake' : classIcon(cls)));
    el['r-name'].textContent = retake ? t('stress_invalid') : (advice.name || r.class_id);

    el['r-sev'].dataset.sev = sev;
    el['r-sev'].textContent = retake ? t('low_confidence') : sevName(sev);
    el['r-crop'].textContent = `${cropName(advice.crop)} · ${stressName(advice.stress)}`;

    el['r-conf'].textContent = pct(r.confidence);
    el['r-conf-fill'].style.width = `${Math.max(2, Math.round((r.confidence || 0) * 100))}%`;
    el['r-conf-fill'].style.background = retake ? 'var(--amber)' : sevColor(sev);

    el['r-retake'].hidden = !retake;
    const healthy = !retake && advice.stress === 'healthy';
    el['r-healthy'].hidden = !healthy;

    const actions = retake ? [t('aim_hint'), t('low_confidence')] : (advice.actions || []);
    el['r-actions'].innerHTML = actions.length
      ? actions.map((a) => `<li>${icon('chev')}<span>${esc(a)}</span></li>`).join('')
      : `<li>${icon('info')}<span>${esc(advice.summary || t('healthy_note'))}</span></li>`;

    const voice = r.voice || {};
    el['r-voice-text'].textContent = voice.text || advice.voice || '';
    el['btn-replay'].disabled = !voice.ok;
    const note = el['r-voice-note'];
    if (!voice.ok) {
      note.hidden = false;
      note.className = 'notice inline notice-warn';
      note.innerHTML = icon('volume-off') + `<span>${esc(t('sys_voice_unavailable'))}</span>`;
    } else if (voice.degraded) {
      note.hidden = false;
      note.className = 'notice inline notice-warn';
      note.innerHTML = icon('alert') + `<span>${esc(t('sys_dialect_fallback'))}</span>`;
    } else {
      note.hidden = true;
    }

    const badges = [];
    if (advice.insurance_claimable) {
      badges.push(`<span class="badge badge-ins">${icon('shield')}<span>${esc(t('insurance_claimable'))}</span></span>`);
    }
    if (advice.window_days) {
      badges.push(`<span class="badge badge-sub">${icon('clock')}<span>${esc(advice.window_days)}d</span></span>`);
    }
    if ((advice.agro_input || {}).category) {
      badges.push(`<span class="badge badge-agr">${icon('box')}<span>${esc(t('agro_input_hint'))} · ${esc(advice.agro_input.category)}</span></span>`);
    }
    if (!retake && advice.stress !== 'healthy') {
      badges.push(`<span class="badge badge-sub">${icon('database')}<span>${esc(t('subsidy_hint'))}</span></span>`);
    }
    el['r-badges'].innerHTML = badges.join('');

    const saved = !!r.record_id;
    el['btn-record'].hidden = false;
    el['btn-record'].innerHTML = saved
      ? icon('check') + `<span>${esc(t('record_saved'))}</span>`
      : icon('save') + `<span>${esc(t('save_record'))}</span>`;
    el['btn-record'].disabled = !saved;
    el['btn-record'].classList.toggle('btn-primary', saved);

    const rt = r.runtime || {};
    el['r-detail'].innerHTML = [
      [t('confidence'), pct(r.confidence)],
      [t('result_severity'), `${sevName(sev)} (${((r.severity || {}).score || 0).toFixed(2)})`],
      [t('candidates'), (r.candidates || []).map((c) => `${c.name} ${pct(c.probability)}`).join(' / ')],
      [t('backend'), `${rt.backend || '-'} · ${rt.model_id || '-'} v${rt.model_version || '-'}`],
      [t('metric_latency'), `${(rt.latency_ms || 0).toFixed(1)} ms`],
      [t('record_detail'), r.record_id || '-'],
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
    el['r-more'].open = false;
  }

  function sevColor(id) {
    const s = S.sevById[id];
    return (s && s.color) || 'var(--leaf)';
  }

  // ---------------------------------------------------------------- 语音
  function playVoice(url) {
    stopVoice();
    const target = url || (S.result && S.result.voice && S.result.voice.ok
      ? S.result.voice.url : null);
    if (!target) { toast(t('sys_voice_unavailable'), 'warn', 'volume-off'); return; }
    el.player.src = target;
    el.player.play().then(() => {
      setPlaying(true);
    }).catch(() => {
      setPlaying(false);
      toast(t('sys_voice_unavailable'), 'warn', 'volume-off');
    });
  }

  function stopVoice() {
    if (!el.player.paused) el.player.pause();
    el.player.removeAttribute('src');
    el.player.load();
    setPlaying(false);
  }

  function setPlaying(on) {
    el['btn-replay'].classList.toggle('is-playing', !!on);
    $('#replay-icon use').setAttribute('href', on ? '#i-stop' : '#i-volume');
    el['btn-replay'].title = on ? t('stop_voice') : t('replay_voice');
  }

  // ---------------------------------------------------------------- 记录
  function renderFilters() {
    const counts = { all: S.boot.records_total || 0 };
    const defs = [{ k: 'all', label: t('history_filter_all') }]
      .concat(['rice', 'peanut', 'vegetable'].map((c) => ({ k: 'crop:' + c, label: cropName(c) })))
      .concat([{ k: 'retake', label: t('stress_invalid') }]);
    el['rec-filters'].innerHTML = defs.map((d) => {
      const on = (S.recFilter.key || 'all') === d.k;
      return `<button type="button" class="fchip${on ? ' is-on' : ''}" data-k="${d.k}">${esc(d.label)}</button>`;
    }).join('');
    return counts;
  }

  function filterToParams() {
    const key = S.recFilter.key || 'all';
    const p = new URLSearchParams();
    if (key.startsWith('crop:')) p.set('crop', key.slice(5));
    if (key === 'retake') p.set('needs_retake', '1');
    return p;
  }

  async function loadRecords(reset) {
    if (reset) { S.recOffset = 0; S.records = []; }
    renderFilters();
    const p = filterToParams();
    p.set('limit', '30');
    p.set('offset', String(S.recOffset));
    try {
      const data = await api('/api/records?' + p.toString());
      S.records = S.records.concat(data.items || []);
      S.recTotal = data.total || 0;
      S.recOffset = S.records.length;
      renderRecords();
    } catch (e) {
      toast(e.message || t('sys_error'), 'bad');
    }
  }

  function renderRecords() {
    el['rec-count'].textContent = t('records_count', { n: S.recTotal });
    el['rec-empty'].hidden = S.records.length > 0;
    el['rec-empty'].textContent = S.recTotal ? t('no_records_match') : t('history_empty');
    el['rec-more'].hidden = S.records.length >= S.recTotal;
    el['rec-list'].innerHTML = S.records.map((r) => {
      const cls = S.classById[r.class_id] || {};
      const thumb = r.has_image
        ? `<img class="rec-thumb" loading="lazy" src="/api/records/${encodeURIComponent(r.record_id)}/image" alt="">`
        : `<span class="rec-thumb-ph">${icon(classIcon(cls))}</span>`;
      return `<li><button type="button" class="rec-item" data-id="${esc(r.record_id)}" data-sev="${esc(r.severity || 'none')}">
        ${thumb}
        <span class="rec-main">
          <span class="rec-name">${esc(r.needs_retake ? t('stress_invalid') : (r.class_name || r.class_id))}</span>
          <span class="rec-meta">
            <span>${icon('clock')} ${esc(localTime(r.created_at))}</span>
            <span>${esc(cropName(r.crop))}</span>
            ${r.confidence != null ? `<span><b>${pct(r.confidence)}</b></span>` : ''}
            ${r.village ? `<span>${esc(r.village)}</span>` : ''}
          </span>
        </span>
        <span class="rec-sev" data-sev="${esc(r.severity || 'none')}">${esc(r.needs_retake ? '—' : sevName(r.severity))}</span>
      </button></li>`;
    }).join('');
  }

  function updateBadge() {
    api('/api/health').then((h) => {
      const n = Number(h.records || 0);
      if (S.boot) S.boot.records_total = n;
      el['tab-badge'].hidden = n === 0;
      el['tab-badge'].textContent = n > 99 ? '99+' : String(n);
    }).catch(() => {});
  }

  async function openRecord(id) {
    let data;
    try { data = await api('/api/records/' + encodeURIComponent(id)); }
    catch (e) { toast(e.message || t('sys_error'), 'bad'); return; }
    const r = data.record;
    const d = r.diagnosis || {};
    const a = r.advice || {};
    const o = r.observation || {};
    const f = r.followup || {};
    const c = r.consent || {};
    const sev = (d.severity || {}).id || 'none';
    const cls = S.classById[d.class_id] || {};

    const photo = r.image_url
      ? `<img class="sheet-photo" src="${esc(r.image_url)}" alt="${esc(t('photo'))}">`
      : `<p class="notice">${icon('image')}<span>${esc(t('no_image'))}</span></p>`;

    const cands = (d.candidates || []).map((cd) => `
      <div class="cand"><b>${esc(cd.name || cd.class_id)}</b><i>${pct(cd.probability)}</i>
      <small>${esc(cropName(cd.crop))} · ${esc(stressName(cd.stress))}</small></div>`).join('');

    const consentKeys = ['insurance', 'subsidy', 'supplier', 'research'];
    const consents = consentKeys.map((k) => `
      <label class="switch"><input type="checkbox" data-consent="share_${k}" ${c['share_' + k] ? 'checked' : ''}>
      <span class="track"><span class="thumb"></span></span>
      <span>${esc(t('consent_' + k))}</span></label>`).join('');

    openSheet(t('record_detail'), `
      ${photo}
      <dl class="kv">
        <dt>${esc(t('result_name'))}</dt><dd>${esc(d.class_name || d.class_id)}</dd>
        <dt>${esc(t('result_severity'))}</dt><dd>${esc(sevName(sev))} · ${((d.severity || {}).score || 0).toFixed(2)}</dd>
        <dt>${esc(t('confidence'))}</dt><dd>${pct(d.confidence)}</dd>
        <dt>${esc(t('crop_rice').length ? t('result_actions') : '')}</dt><dd>${esc((a.actions || []).join('；') || a.summary || '—')}</dd>
        <dt>${esc(localTime(r.created_at) ? 'time' : '')}</dt><dd>${esc(localTime(r.created_at))}</dd>
        <dt>ID</dt><dd>${esc(r.record_id)}</dd>
        <dt>SHA-256</dt><dd>${esc((r.integrity || {}).record_sha256 || '—')}</dd>
      </dl>
      <div class="btn-row">
        <button type="button" class="btn" id="sh-play" ${data.record.voice && data.record.voice.ok ? '' : 'disabled'}>
          ${icon('volume')}<span>${esc(t('replay_voice'))}</span></button>
      </div>
      <h4>${esc(t('candidates'))}</h4>
      <div>${cands || '—'}</div>
      <h4>${esc(t('followup'))}</h4>
      <div class="field-grid">
        <label><span>${esc(t('action_taken'))}</span><input id="fu-action" value="${esc(f.action_taken || '')}"></label>
        <label><span>${esc(t('product_used'))}</span><input id="fu-product" value="${esc(f.product_used || '')}"></label>
        <label><span>${esc(t('effect'))}</span>
          <select id="fu-effect">
            <option value="">—</option>
            <option value="better" ${f.effect === 'better' ? 'selected' : ''}>${esc(t('effect_better'))}</option>
            <option value="same" ${f.effect === 'same' ? 'selected' : ''}>${esc(t('effect_same'))}</option>
            <option value="worse" ${f.effect === 'worse' ? 'selected' : ''}>${esc(t('effect_worse'))}</option>
          </select></label>
        <label><span>${esc(t('profile_alias'))}</span><input id="fu-by" value="${esc(f.reviewed_by || '')}"></label>
      </div>
      <button type="button" class="btn btn-primary block" id="fu-save">${icon('save')}<span>${esc(t('save_record'))}</span></button>
      <h4>${esc(t('consent_title'))}</h4>
      <p class="notice">${icon('shield')}<span>${esc(t('consent_note'))}</span></p>
      ${consents}
      <button type="button" class="btn block" id="consent-save">${icon('save')}<span>${esc(t('save_record'))}</span></button>
      <button type="button" class="btn btn-danger block" id="rec-del">${icon('trash')}<span>${esc(t('delete'))}</span></button>
    `);

    const body = el['sheet-body'];
    const play = $('#sh-play', body);
    if (play) play.addEventListener('click', () => {
      playVoice(`/api/voice?record_id=${encodeURIComponent(r.record_id)}&lang=${S.lang}`);
    });
    $('#fu-save', body).addEventListener('click', async () => {
      try {
        await api(`/api/records/${encodeURIComponent(r.record_id)}/followup`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action_taken: $('#fu-action', body).value,
            product_used: $('#fu-product', body).value,
            effect: $('#fu-effect', body).value || null,
            reviewed_by: $('#fu-by', body).value,
          }),
        });
        toast(t('sys_saved'));
      } catch (e) { toast(e.message || t('sys_error'), 'bad'); }
    });
    $('#consent-save', body).addEventListener('click', async () => {
      const flags = {};
      $$('[data-consent]', body).forEach((i) => { flags[i.dataset.consent] = i.checked; });
      try {
        await api(`/api/records/${encodeURIComponent(r.record_id)}/consent`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(flags),
        });
        toast(t('sys_saved'));
      } catch (e) { toast(e.message || t('sys_error'), 'bad'); }
    });
    $('#rec-del', body).addEventListener('click', async () => {
      if (!window.confirm(t('confirm_delete'))) return;
      try {
        await api('/api/records/' + encodeURIComponent(r.record_id), { method: 'DELETE' });
        closeSheet();
        toast(t('sys_saved'));
        S.recFilter = {};
        await refreshTotals();
        await loadRecords(true);
      } catch (e) { toast(e.message || t('sys_error'), 'bad'); }
    });
  }

  async function refreshTotals() {
    try {
      const h = await api('/api/health');
      S.boot.records_total = h.records;
      renderRail();
    } catch (e) { /* 忽略：统计刷新失败不影响主流程 */ }
  }

  // ---------------------------------------------------------------- 导出
  async function doExport() {
    const out = el['export-out'];
    out.hidden = false;
    out.innerHTML = `<p class="export-line">${icon('spinner')}<span>${esc(t('loading'))}</span></p>`;
    try {
      const r = await api('/api/export', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fmt: el['exp-fmt'].value,
          purpose: el['exp-purpose'].value,
          channel: el['exp-channel'].value,
          anonymize: el['exp-anon'].checked,
        }),
      });
      renderExport(r);
    } catch (e) {
      renderExport((e && e.payload) || { ok: false, errors: [e.message || t('export_failed')] });
    }
  }

  function renderExport(r) {
    const lines = [];
    const push = (ok, text) => lines.push(
      `<p class="export-line ${ok ? 'ok' : 'bad'}">${icon(ok ? 'check' : 'alert')}<span>${esc(text)}</span></p>`);
    push(!!r.ok, r.ok
      ? `${t('sys_exported')} · ${r.record_count || 0} · ${r.format || ''} · ${r.channel || ''}`
      : t('export_failed'));
    (r.errors || []).forEach((m) => push(false, m));
    (r.messages || []).forEach((m) => push(true, m));
    if (r.destinations && r.destinations.length) {
      push(true, r.destinations.join(' | '));
    }
    if (r.chain_ok === false) push(false, t('integrity_bad'));
    if (r.download_url) {
      lines.push(`<a class="btn btn-primary block" href="${esc(r.download_url)}">
        ${icon('download')}<span>${esc(t('download_now'))}</span></a>`);
    }
    el['export-out'].innerHTML = lines.join('');
  }

  async function doIntegrity() {
    const out = el['export-out'];
    out.hidden = false;
    out.innerHTML = `<p class="export-line">${icon('spinner')}<span>${esc(t('loading'))}</span></p>`;
    try {
      const r = await api('/api/integrity');
      const ok = r.ok !== false && (r.chain_ok !== false) && (r.records_intact !== false);
      out.innerHTML = `<p class="export-line ${ok ? 'ok' : 'bad'}">${icon(ok ? 'shield' : 'alert')}
        <span>${esc(ok ? t('integrity_ok') : t('integrity_bad'))}</span></p>
        <pre class="code">${esc(JSON.stringify(r, null, 2))}</pre>`;
    } catch (e) {
      out.innerHTML = `<p class="export-line bad">${icon('alert')}<span>${esc(e.message)}</span></p>`;
    }
  }

  // ---------------------------------------------------------------- 设置
  function renderSettings() {
    const langs = S.boot.languages || [];
    const review = new Set(S.boot.dialect_review_needed || []);
    el['set-lang'].innerHTML = langs.map((l) => `
      <button type="button" data-v="${esc(l.code)}" class="${l.code === S.lang ? 'is-on' : ''}">
        <span>${esc(l.name_local)}</span>${review.has(l.code) ? `<em class="mini">${esc(t('review_needed_badge'))}</em>` : ''}
      </button>`).join('');

    $$('button', el['set-size']).forEach((b) =>
      b.classList.toggle('is-on', b.dataset.v === (S.settings.text_size || 'large')));
    $$('button', el['set-speed']).forEach((b) =>
      b.classList.toggle('is-on', b.dataset.v === (S.settings.voice_speed || 'normal')));
    el['set-auto-voice'].checked = !!S.settings.auto_voice;

    const note = el['set-lang-note'];
    if (review.has(S.lang)) {
      note.hidden = false;
      note.className = 'notice inline notice-warn';
      note.innerHTML = icon('alert') + `<span>${esc(t('dialect_review_note'))}</span>`;
    } else note.hidden = true;

    el['pf-alias'].value = S.profile.alias || '';
    el['pf-farmer-id'].value = S.profile.farmer_id || '';
    el['pf-village'].value = S.profile.village || '';
    el['pf-town'].value = S.profile.town || '';
    el['pf-county'].value = S.profile.county || '';
    el['pf-plot'].value = S.profile.plot_id || '';

    renderVoiceStatus(el['voice-status']);
  }

  function renderVoiceStatus(node) {
    const v = S.boot.voice || {};
    const names = Object.fromEntries((S.boot.languages || []).map((l) => [l.code, l.name_local]));
    const fmt = (arr) => (arr || []).map((c) => names[c] || c).join('、') || '—';
    const pack = v.voicepack || {};
    node.innerHTML = [
      [t('voice_engine'), `${v.engine || '—'}${v.offline ? ' · offline' : ''}`],
      [t('voice_pack'), pack.clips != null ? `${pack.clips} clips / ${pack.languages ? pack.languages.length : 0} lang` : '—'],
      [t('voice_ready_langs'), fmt(v.voiced_languages)],
      [t('voice_missing_langs'), fmt(v.dialect_needs_recording)],
    ].map(([k, val]) => `<p class="vs-line"><b>${esc(k)}</b><span>${esc(val)}</span></p>`).join('');
  }

  function budgetRow(label, value, limit, ok, note) {
    const cls = ok == null ? 'is-na' : ok ? '' : 'is-bad';
    const state = ok == null ? '—' : ok ? t('budget_pass') : t('budget_fail');
    return `<div class="budget-row"><b>${esc(label)}</b>
      <span class="budget-state ${cls}">${icon(ok === false ? 'x' : 'check')}${esc(state)}</span>
      <small>${esc(value)}${limit ? ` / ${esc(limit)}` : ''}${note ? ` · ${esc(note)}` : ''}</small></div>`;
  }

  function budgetRows() {
    const b = S.boot.benchmark || {};
    const m = (S.boot.model && S.boot.model.metrics) || {};
    const limits = (S.boot.model && S.boot.model.budgets) || {};
    const rows = [];
    rows.push(budgetRow(t('metric_size'),
      b.model_size_mb != null ? `${Number(b.model_size_mb).toFixed(2)} MB` : '—',
      limits.model_size_mb ? `≤ ${limits.model_size_mb} MB` : '',
      b.model_size_mb != null && limits.model_size_mb ? b.model_size_mb <= limits.model_size_mb : null));
    rows.push(budgetRow(t('metric_latency'),
      b.latency_p95_ms != null ? `${Number(b.latency_p95_ms).toFixed(1)} ms (p95)` : '—',
      limits.latency_s ? `≤ ${limits.latency_s * 1000} ms` : '',
      b.latency_p95_ms != null && limits.latency_s ? b.latency_p95_ms <= limits.latency_s * 1000 : null));
    rows.push(budgetRow(t('metric_memory'),
      b.memory_delta_mb != null ? `${Number(b.memory_delta_mb).toFixed(1)} MB` : '—',
      limits.runtime_memory_mb ? `≤ ${limits.runtime_memory_mb} MB` : '',
      b.memory_delta_mb != null && limits.runtime_memory_mb
        ? b.memory_delta_mb <= limits.runtime_memory_mb : null));
    const drop = m.accuracy_drop;
    rows.push(budgetRow(t('metric_accuracy'),
      drop != null ? `${(drop * 100).toFixed(2)}%` : (m.top1_int8 != null ? `${pct(m.top1_int8)} (INT8)` : '—'),
      m.accuracy_drop_max != null ? `≤ ${(m.accuracy_drop_max * 100).toFixed(0)}%` : '',
      drop != null && m.accuracy_drop_max != null ? drop <= m.accuracy_drop_max : null,
      m.top1_fp32 != null && m.top1_int8 != null
        ? `FP32 ${pct(m.top1_fp32)} → INT8 ${pct(m.top1_int8)}` : ''));
    return rows.join('');
  }

  function renderModel() {
    const mo = S.boot.model || {};
    const m = mo.metrics || {};
    const rows = [
      [t('model_info'), `${mo.model_id || '—'} v${mo.version || '—'}`],
      ['arch', mo.arch || '—'],
      [t('backend'), (mo.backend && (mo.backend.backend || mo.backend.name)) || '—'],
      ['classes', mo.num_classes || S.classes.length],
      [t('metric_size'), mo.model_size_mb != null ? `${Number(mo.model_size_mb).toFixed(2)} MB` : '—'],
      ['top1 (INT8)', m.top1_int8 != null ? pct(m.top1_int8) : '—'],
      ['top1 (FP32)', m.top1_fp32 != null ? pct(m.top1_fp32) : '—'],
      ['quant', m.quant_method || m.quantization || '—'],
    ];
    if (!mo.available) rows.unshift([t('sys_no_model'), mo.error || '—']);
    el['model-kv'].innerHTML = rows
      .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
    el['budget-list'].innerHTML = budgetRows();

    el['paths-kv'].innerHTML = [
      ['DB', S.boot.paths.records_db],
      ['exports', S.boot.paths.exports],
      ['outbox', S.boot.paths.outbox],
      ['bundle', mo.bundle_dir || '—'],
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');

    el['about-kv'].innerHTML = [
      [t('app_name'), `${S.boot.app.name} / ${S.boot.app.name_en} v${S.boot.app.version}`],
      ['region', S.boot.app.region],
      ['flow', (S.boot.app.flow || []).join(' → ')],
      ['offline', t('offline_ready')],
      ['server_time', S.boot.server_time],
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
  }

  function renderRail() {
    const mo = S.boot.model || {};
    el['rail-kv'].innerHTML = [
      [t('rail_status'), mo.available ? t('model_ready') : t('sys_no_model')],
      [t('backend'), (mo.backend && mo.backend.backend) || '—'],
      [t('metric_latency'), (S.boot.benchmark || {}).latency_p50_ms != null
        ? `${Number(S.boot.benchmark.latency_p50_ms).toFixed(1)} ms (p50)` : '—'],
      ['records', t('records_saved_n', { n: S.boot.records_total || 0 })],
      [t('language'), (S.boot.languages || []).find((l) => l.code === S.lang)?.name_local || S.lang],
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
    el['rail-kv'].className = 'kv';
    el['rail-budget'].innerHTML = budgetRows();
    renderVoiceStatus(el['rail-voice']);
  }

  async function saveSettings(patch, profile) {
    const body = {};
    if (patch) Object.assign(body, patch);
    if (profile) body.profile = profile;
    try {
      const r = await api('/api/settings', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      S.settings = r.settings;
      S.profile = r.profile;
      if (r.i18n) S.i18n = r.i18n;
      if (patch && patch.language) {
        S.lang = r.settings.language;
        S.boot.language = S.lang;
        applyLang();
        renderSettings();
        renderModel();
        renderRail();
        if (S.view === 'records') renderRecords();
        if (S.result) renderResult(S.result);
      }
      return true;
    } catch (e) {
      toast(e.message || t('sys_error'), 'bad');
      return false;
    }
  }

  // ---------------------------------------------------------------- 对接
  async function showAdapters() {
    openSheet(t('view_adapters'), `<p class="export-line">${icon('spinner')}<span>${esc(t('loading'))}</span></p>`);
    try {
      const r = await api('/api/adapters');
      const list = r.adapters || [];
      const items = list.map((a) => {
        const st = (r.status || {})[a.name] || {};
        return `<div class="block" style="gap:8px">
          <h3>${esc(a.title || a.name)}</h3>
          <p class="vs-line"><span>${esc(a.description || '')}</span></p>
          <dl class="kv">
            <dt>schema</dt><dd>${esc(a.schema_id || a.schema || '—')}</dd>
            <dt>channel</dt><dd>${esc(a.channel || '—')}</dd>
            <dt>pending</dt><dd>${esc(st.pending != null ? st.pending : '—')}</dd>
          </dl>
          <button type="button" class="btn btn-sm" data-dispatch="${esc(a.name)}">
            ${icon('box')}<span>${esc(a.name)}</span></button>
        </div>`;
      }).join('');
      el['sheet-body'].innerHTML = items || `<p class="notice">${icon('info')}<span>—</span></p>`;
      $$('[data-dispatch]', el['sheet-body']).forEach((b) => b.addEventListener('click', async () => {
        try {
          const res = await api('/api/adapters/dispatch', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ names: b.dataset.dispatch }),
          });
          toast(t('sys_exported'));
          b.insertAdjacentHTML('afterend',
            `<pre class="code">${esc(JSON.stringify(res.results, null, 2))}</pre>`);
        } catch (e) { toast(e.message || t('sys_error'), 'bad'); }
      }));
    } catch (e) {
      el['sheet-body'].innerHTML = `<p class="notice notice-bad">${icon('alert')}<span>${esc(e.message)}</span></p>`;
    }
  }

  async function showSchemas() {
    openSheet(t('view_schemas'), `<p class="export-line">${icon('spinner')}<span>${esc(t('loading'))}</span></p>`);
    try {
      const r = await api('/api/schemas');
      el['sheet-body'].innerHTML = `
        <h4>record.schema.json</h4>
        <pre class="code">${esc(JSON.stringify(r.record, null, 2))}</pre>
        <h4>export.schema.json</h4>
        <pre class="code">${esc(JSON.stringify(r.export, null, 2))}</pre>`;
    } catch (e) {
      el['sheet-body'].innerHTML = `<p class="notice notice-bad">${icon('alert')}<span>${esc(e.message)}</span></p>`;
    }
  }

  // ---------------------------------------------------------------- 事件
  function wire() {
    el['btn-shutter'].addEventListener('click', shoot);
    el['stage-fallback'].addEventListener('click', () => el['file-capture'].click());
    el['btn-album'].addEventListener('click', () => el['file-album'].click());
    el['btn-flip'].addEventListener('click', () => {
      S.facing = S.facing === 'environment' ? 'user' : 'environment';
      startCam();
    });
    el['file-capture'].addEventListener('change', onFile);
    el['file-album'].addEventListener('change', onFile);
    el['samples-row'].addEventListener('click', (e) => {
      const btn = e.target.closest('.sample');
      if (!btn) return;
      const item = S.recents[Number(btn.dataset.i)];
      if (item) submit(item.blob, 'recent.jpg');
    });

    el['btn-retake'].addEventListener('click', () => { stopVoice(); setView('scan'); });
    el['btn-replay'].addEventListener('click', () => {
      if (!el.player.paused) { stopVoice(); return; }
      playVoice();
    });
    el['btn-record'].addEventListener('click', () => {
      if (S.result && S.result.record_id) openRecord(S.result.record_id);
    });
    el.player.addEventListener('ended', () => setPlaying(false));
    el.player.addEventListener('pause', () => setPlaying(false));

    $$('.tab').forEach((b) => b.addEventListener('click', () => {
      stopVoice();
      setView(b.dataset.view);
    }));

    el['rec-filters'].addEventListener('click', (e) => {
      const chip = e.target.closest('.fchip');
      if (!chip) return;
      S.recFilter = { key: chip.dataset.k };
      loadRecords(true);
    });
    el['rec-list'].addEventListener('click', (e) => {
      const item = e.target.closest('.rec-item');
      if (item) openRecord(item.dataset.id);
    });
    el['rec-more'].addEventListener('click', () => loadRecords(false));
    el['btn-export'].addEventListener('click', doExport);
    el['btn-integrity'].addEventListener('click', doIntegrity);

    el['set-lang'].addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      saveSettings({ language: b.dataset.v }).then((ok) => {
        if (ok) {
          $$('button', el['set-lang']).forEach((x) => x.classList.toggle('is-on', x === b));
          toast(t('sys_saved'));
        }
      });
    });
    el['set-size'].addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      document.documentElement.dataset.size = b.dataset.v;
      $$('button', el['set-size']).forEach((x) => x.classList.toggle('is-on', x === b));
      saveSettings({ text_size: b.dataset.v });
    });
    el['set-speed'].addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      $$('button', el['set-speed']).forEach((x) => x.classList.toggle('is-on', x === b));
      saveSettings({ voice_speed: b.dataset.v });
    });
    el['set-auto-voice'].addEventListener('change', (e) =>
      saveSettings({ auto_voice: e.target.checked }));

    el['btn-voice-test'].addEventListener('click', () => {
      if (!el.player.paused) { stopVoice(); return; }
      playVoice(`/api/voice?slug=ui_sys_ready&lang=${S.lang}`);
    });

    el['btn-profile-save'].addEventListener('click', () => {
      saveSettings(null, {
        alias: el['pf-alias'].value, farmer_id: el['pf-farmer-id'].value,
        village: el['pf-village'].value, town: el['pf-town'].value,
        county: el['pf-county'].value, plot_id: el['pf-plot'].value,
      }).then((ok) => ok && toast(t('sys_saved')));
    });

    el['btn-adapters'].addEventListener('click', showAdapters);
    el['btn-schemas'].addEventListener('click', showSchemas);
    el['btn-lang-quick'].addEventListener('click', () => setView('settings'));

    el['sheet-close'].addEventListener('click', closeSheet);
    $('.sheet-back').addEventListener('click', closeSheet);
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !el.sheet.hidden) closeSheet();
      if (e.key === ' ' && S.view === 'scan' && document.activeElement === document.body) {
        e.preventDefault();
        shoot();
      }
    });

    window.addEventListener('online', () => setPill(true));
    window.addEventListener('offline', () => setPill(false));
  }

  function setPill(online) {
    // 这个系统本来就是离线的，联网只是"多了个选择"，不是"恢复了功能"
    el['pill-status'].className = 'pill';
    el['pill-status'].classList.add(online ? 'pill-ok' : 'pill-ok');
  }

  async function onFile(e) {
    const input = e.target;
    const file = input.files && input.files[0];
    input.value = '';
    if (!file) return;
    await submit(file, file.name);
  }

  document.addEventListener('DOMContentLoaded', boot);
})();
