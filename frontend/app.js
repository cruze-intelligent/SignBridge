/**
 * SignBridge — app.js  (v4 — Uganda Sign Language · configurable backend)
 *
 * Vanilla JS, no frameworks, no bundlers.
 *
 * Module layout:
 *   1.  CONFIG             — all constants in one place
 *   2.  Utils              — $(), $$(), showToast(), appendLog(), timeStr()
 *   3.  OnboardingWizard   — first-run 4-step tutorial
 *   4.  HelpDrawer         — floating ? FAB → bottom sheet
 *   5.  SettingsPanel      — backend URL configuration, persisted to localStorage
 *   6.  LandmarkPipeline   — 225-float extraction (Holistic results → array)
 *   7.  LandmarkBuffer     — 30-frame sliding window, auto-flush callback
 *   8.  WebSocketClient    — configurable WS URL + auto-reconnect
 *   9.  HolisticController — MediaPipe Holistic wrapper + canvas rendering
 *  10.  UIController       — state machine that wires all modules together
 *  11.  App.init()
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   1. CONFIG
   ═══════════════════════════════════════════════════════════════════════ */
/**
 * Resolve the WebSocket URL.
 * Priority: localStorage key → ?backend= query param → <meta> default → fallback.
 */
function resolveWsUrl() {
  const stored = localStorage.getItem('sb_backend_url');
  if (stored) return stored;

  const param = new URLSearchParams(window.location.search).get('backend');
  if (param) return param;

  const meta = document.querySelector('meta[name="sb-default-ws"]');
  if (meta?.content) return meta.content;

  return 'ws://localhost:8000/ws/landmarks';
}

const CONFIG = Object.freeze({
  // ── Server ────────────────────────────────────────────────────────────
  WS_URL:          resolveWsUrl(),
  WS_RECONNECT_MS: 3_000,

  // ── 225-float frame spec (MUST match backend) ─────────────────────────
  //   Right Hand : 21 landmarks × (x, y, z) =  63 floats
  //   Left  Hand : 21 landmarks × (x, y, z) =  63 floats
  //   Pose       : 33 landmarks × (x, y, z) =  99 floats
  //                                          ────────────
  //   Total                                 = 225 floats
  HAND_LM:            21,
  POSE_LM:            33,
  HAND_FLOATS:        63,   // HAND_LM × 3
  POSE_FLOATS:        99,   // POSE_LM × 3
  FRAME_SIZE:        225,   // total per frame

  // ── Sequence buffer ───────────────────────────────────────────────────
  FRAME_WINDOW:       30,   // flush every N frames

  // ── Pose landmark indices (MediaPipe Holistic convention) ─────────────
  NOSE_IDX:            0,   // used as spatial-normalisation anchor

  // ── MediaPipe model settings ──────────────────────────────────────────
  MODEL_COMPLEXITY:    1,   // 0 = fast, 1 = balanced, 2 = accurate
  MIN_DETECTION:      0.7,
  MIN_TRACKING:       0.6,

  // ── UI ────────────────────────────────────────────────────────────────
  ONBOARDING_KEY:  'sb_onboarding_v3',
  TOAST_MS:        3_500,
  LOG_MAX:           100,
});

/* ═══════════════════════════════════════════════════════════════════════════
   2. UTILS
   ═══════════════════════════════════════════════════════════════════════ */

/** Shorthand querySelector. */
const $ = (sel, ctx = document) => ctx.querySelector(sel);

/** Shorthand querySelectorAll → Array. */
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

/**
 * Display a self-dismissing toast notification.
 * @param {string} msg
 * @param {'info'|'success'|'error'|'warning'} [type='info']
 */
function showToast(msg, type = 'info') {
  const icons = { info: 'ℹ️', success: '✅', error: '❌', warning: '⚠️' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.setAttribute('role', 'status');
  el.innerHTML = `<span aria-hidden="true">${icons[type] ?? '•'}</span><span>${msg}</span>`;
  $('#toast-container').appendChild(el);
  setTimeout(() => {
    el.classList.add('out');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, CONFIG.TOAST_MS);
}

/**
 * Append a timestamped line to the server log panel.
 * @param {string} text
 * @param {'ok'|'err'|'warn'|'info'|''} [cls='']
 */
function appendLog(text, cls = '') {
  const log = $('#server-log');
  const line = document.createElement('span');
  line.className = cls ? `log-${cls}` : '';
  line.textContent = `[${new Date().toLocaleTimeString()}] ${text}\n`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
  // Trim oldest entries
  while (log.children.length > CONFIG.LOG_MAX) log.firstChild.remove();
}

/** Current time as HH:MM:SS string. */
function timeStr() {
  return new Date().toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   3. ONBOARDING WIZARD
   Shown once on first load. Stored dismissal in localStorage.
   ═══════════════════════════════════════════════════════════════════════ */
class OnboardingWizard {
  constructor() {
    this._overlay = $('#onboarding-overlay');
    this._dots    = $$('.wizard-dot');
    this._slides  = $$('.wizard-slide');
    this._btnNext = $('#wizard-next');
    this._btnSkip = $('#wizard-skip');
    this._idx     = 0;

    this._btnNext.addEventListener('click', () => this._advance());
    this._btnSkip.addEventListener('click', () => this.close());
    this._overlay.addEventListener('click', (e) => {
      if (e.target === this._overlay) this.close();
    });
  }

  /** Show if the user has never dismissed. */
  showIfNeeded() {
    if (!localStorage.getItem(CONFIG.ONBOARDING_KEY)) this.show();
  }

  show() {
    this._idx = 0;
    this._render();
    this._overlay.removeAttribute('hidden');
    this._btnNext.focus();
  }

  close() {
    localStorage.setItem(CONFIG.ONBOARDING_KEY, '1');
    this._overlay.setAttribute('hidden', '');
  }

  _advance() {
    if (this._idx < this._slides.length - 1) {
      this._idx++;
      this._render();
    } else {
      this.close();
    }
  }

  _render() {
    const isLast = this._idx === this._slides.length - 1;
    this._btnNext.textContent = isLast ? 'Get Started 🚀' : 'Next';
    this._slides.forEach((s, i) =>
      i === this._idx ? s.removeAttribute('hidden') : s.setAttribute('hidden', '')
    );
    this._dots.forEach((d, i) => d.classList.toggle('active', i === this._idx));
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   4. HELP DRAWER
   Floating ? button opens a bottom-sheet with quick-reference tips.
   ═══════════════════════════════════════════════════════════════════════ */
class HelpDrawer {
  constructor(wizard) {
    this._wizard  = wizard;
    this._overlay = $('#help-overlay');
    this._fab     = $('#fab-help');

    this._fab.addEventListener('click',          () => this.open());
    $('#help-close').addEventListener('click',   () => this.close());
    $('#replay-tutorial').addEventListener('click', () => {
      this.close();
      this._wizard.show();
    });
    this._overlay.addEventListener('click', (e) => {
      if (e.target === this._overlay) this.close();
    });
  }

  open() {
    this._overlay.removeAttribute('hidden');
    this._overlay.removeAttribute('aria-hidden');
    this._fab.setAttribute('aria-expanded', 'true');
    $('#help-close').focus();
  }

  close() {
    this._overlay.setAttribute('hidden', '');
    this._overlay.setAttribute('aria-hidden', 'true');
    this._fab.setAttribute('aria-expanded', 'false');
    this._fab.focus();
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   5. SETTINGS PANEL
   Allows the user to override the backend WebSocket URL and save it to
   localStorage — essential when the frontend is hosted on GitHub Pages and
   the backend is deployed on a separate service (Render, Railway, Fly.io).
   ═══════════════════════════════════════════════════════════════════════ */
class SettingsPanel {
  constructor() {
    this._overlay     = $('#settings-overlay');
    this._input       = $('#backend-url-input');
    this._currentLbl  = $('#settings-current-url');
    this._navSettings = $('#nav-settings');

    $('#settings-close').addEventListener('click',     () => this.close());
    $('#settings-save-url').addEventListener('click',  () => this._save());
    $('#settings-reset-url').addEventListener('click', () => this._reset());
    this._overlay.addEventListener('click', (e) => {
      if (e.target === this._overlay) this.close();
    });
    this._navSettings.addEventListener('click', () => this.open());
  }

  open() {
    this._input.value = localStorage.getItem('sb_backend_url') || CONFIG.WS_URL;
    this._currentLbl.textContent = CONFIG.WS_URL;
    this._overlay.removeAttribute('hidden');
    this._overlay.removeAttribute('aria-hidden');
    this._input.focus();
  }

  close() {
    this._overlay.setAttribute('hidden', '');
    this._overlay.setAttribute('aria-hidden', 'true');
    this._navSettings.focus();
  }

  _save() {
    const url = this._input.value.trim();
    if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
      showToast('URL must start with ws:// or wss://', 'error');
      return;
    }
    localStorage.setItem('sb_backend_url', url);
    showToast('Backend URL saved — reloading…', 'success');
    setTimeout(() => window.location.reload(), 1_000);
  }

  _reset() {
    localStorage.removeItem('sb_backend_url');
    showToast('Reset to default — reloading…', 'info');
    setTimeout(() => window.location.reload(), 1_000);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   6. LANDMARK PIPELINE
   ──────────────────────────────────────────────────────────────────────────
   Converts one MediaPipe Holistic result into an exact 225-float array:

     [ Right Hand (63) | Left Hand (63) | Pose (99) ]

   Extraction rules:
   ┌─────────────────────────────────────────────────────────────────────┐
   │ • Right / Left Hand: 21 landmarks × (x, y, z)                      │
   │   – x and y are expressed RELATIVE to the nose landmark             │
   │     (Pose landmark index 0) to achieve spatial normalisation.       │
   │   – z is kept as raw sensor-relative depth (no offset applied).     │
   │ • If a hand is absent (null / wrong count) → inject 63 × 0.0.      │
   │ • Pose: 33 landmarks × (x, y, z) in raw normalised image coords.   │
   │   – If pose is absent → inject 99 × 0.0.                           │
   └─────────────────────────────────────────────────────────────────────┘
   ═══════════════════════════════════════════════════════════════════════ */
class LandmarkPipeline {
  /**
   * Extract a 225-float frame from a Holistic result object.
   *
   * @param {object} results - MediaPipe Holistic onResults payload
   * @returns {{
   *   frame:        number[],   // exactly 225 floats
   *   rhDetected:   boolean,
   *   lhDetected:   boolean,
   *   poseDetected: boolean
   * }}
   */
  static extract(results) {
    // ── Resolve nose anchor for spatial normalisation ──────────────────
    const poseDetected =
      Array.isArray(results.poseLandmarks) &&
      results.poseLandmarks.length === CONFIG.POSE_LM;

    let noseX = 0;
    let noseY = 0;
    if (poseDetected) {
      const nose = results.poseLandmarks[CONFIG.NOSE_IDX];
      noseX = nose.x;
      noseY = nose.y;
    }

    // ── Right-hand block (63 floats) ───────────────────────────────────
    const { coords: rhCoords, detected: rhDetected } =
      LandmarkPipeline._handBlock(results.rightHandLandmarks, noseX, noseY);

    // ── Left-hand block (63 floats) ────────────────────────────────────
    const { coords: lhCoords, detected: lhDetected } =
      LandmarkPipeline._handBlock(results.leftHandLandmarks, noseX, noseY);

    // ── Pose block (99 floats) — raw normalised image coordinates ──────
    const poseCoords = [];
    if (poseDetected) {
      for (const lm of results.poseLandmarks) {
        poseCoords.push(lm.x, lm.y, lm.z);
      }
    } else {
      for (let i = 0; i < CONFIG.POSE_FLOATS; i++) poseCoords.push(0);
    }

    // ── Assemble [ RH | LH | POSE ] ───────────────────────────────────
    const frame = [...rhCoords, ...lhCoords, ...poseCoords];

    if (frame.length !== CONFIG.FRAME_SIZE) {
      console.error(
        `[LandmarkPipeline] frame length ${frame.length} ≠ ${CONFIG.FRAME_SIZE}`,
      );
    }

    return { frame, rhDetected, lhDetected, poseDetected };
  }

  /**
   * Extract 63 nose-relative floats for one hand, or return 63 zeros.
   *
   * @param {Array|null|undefined} landmarks - 21 hand landmarks from Holistic
   * @param {number} noseX
   * @param {number} noseY
   * @returns {{ coords: number[], detected: boolean }}
   */
  static _handBlock(landmarks, noseX, noseY) {
    if (!Array.isArray(landmarks) || landmarks.length !== CONFIG.HAND_LM) {
      return {
        coords: new Array(CONFIG.HAND_FLOATS).fill(0),
        detected: false,
      };
    }
    const coords = [];
    for (const lm of landmarks) {
      coords.push(
        lm.x - noseX,  // normalised, centred on nose
        lm.y - noseY,
        lm.z,           // raw depth — no anchor subtraction per spec
      );
    }
    return { coords, detected: true };
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   7. LANDMARK BUFFER
   30-frame sliding window. Invokes the onFlush callback with a complete
   batch of frames each time it reaches CONFIG.FRAME_WINDOW entries.
   ═══════════════════════════════════════════════════════════════════════ */
class LandmarkBuffer {
  constructor(windowSize = CONFIG.FRAME_WINDOW) {
    this._win    = windowSize;
    this._frames = [];
    this._flush  = null;
  }

  /**
   * Register the callback fired on each complete window.
   * @param {function(number[][]): void} fn
   */
  onFlush(fn) {
    this._flush = fn;
    return this;
  }

  /**
   * Push one 225-float frame. Fires the flush callback when full.
   * @param {number[]} frame - exactly 225 floats
   */
  push(frame) {
    if (!Array.isArray(frame) || frame.length !== CONFIG.FRAME_SIZE) {
      console.warn('[LandmarkBuffer] skipped invalid frame, length =', frame?.length);
      return;
    }
    this._frames.push(frame);
    if (this._frames.length >= this._win) {
      const batch = this._frames.splice(0, this._win);
      this._flush?.(batch);
    }
  }

  /**
   * Return and clear all buffered frames regardless of window fullness.
   * Used when Stop is pressed to flush any trailing partial window.
   * @returns {number[][]}
   */
  drain() {
    return this._frames.splice(0);
  }

  reset() { this._frames = []; }

  get length() { return this._frames.length; }
}

/* ═══════════════════════════════════════════════════════════════════════════
   8. WEBSOCKET CLIENT
   Manages the persistent connection to the configured backend URL.
   Auto-reconnects every CONFIG.WS_RECONNECT_MS on unexpected closure.
   ═══════════════════════════════════════════════════════════════════════ */
class WebSocketClient {
  constructor(url) {
    this._url   = url;
    this._ws    = null;
    this._ready = false;
    this._timer = null;

    /** @type {function(object): void} */
    this.onMessage = null;
    /** @type {function('connected'|'connecting'|'disconnected'): void} */
    this.onStatusChange = null;
  }

  connect() {
    this._emit('connecting');
    appendLog(`Connecting → ${this._url}`, 'info');
    try {
      this._ws = new WebSocket(this._url);
    } catch (err) {
      appendLog(`WebSocket construction failed: ${err.message}`, 'err');
      this._scheduleReconnect();
      return;
    }

    this._ws.onopen = () => {
      this._ready = true;
      this._emit('connected');
      appendLog('WebSocket connected ✓', 'ok');
      showToast('Connected to server', 'success');
      clearTimeout(this._timer);
      this._timer = null;
    };

    this._ws.onmessage = ({ data }) => {
      try {
        const parsed = JSON.parse(data);
        appendLog(
          `← ${JSON.stringify(parsed)}`,
          parsed.status === 'error' ? 'err' : 'ok',
        );
        this.onMessage?.(parsed);
      } catch {
        appendLog(`← (unparsed) ${data}`, 'warn');
      }
    };

    this._ws.onerror = () => appendLog('WebSocket error.', 'err');

    this._ws.onclose = ({ code }) => {
      this._ready = false;
      this._emit('disconnected');
      appendLog(`WebSocket closed (code ${code}).`, 'warn');
      this._scheduleReconnect();
    };
  }

  /**
   * Send a JSON-serialisable payload.
   * @param {object} payload
   * @returns {boolean} true if the message was enqueued
   */
  send(payload) {
    if (!this._ready || this._ws?.readyState !== WebSocket.OPEN) {
      appendLog('Cannot send — not connected.', 'err');
      return false;
    }
    const msg = JSON.stringify(payload);
    this._ws.send(msg);
    // Log a truncated version to keep the log readable for large frames
    appendLog(`→ ${msg.length > 120 ? msg.slice(0, 120) + '…' : msg}`, 'info');
    return true;
  }

  /**
   * Stream an array of 225-float frames as individual per-frame messages.
   * Wire format: { frame: number[225] }
   *
   * @param {number[][]} frames
   * @returns {number} count of messages actually sent
   */
  streamFrames(frames) {
    let sent = 0;
    for (const frame of frames) {
      if (this.send({ frame })) sent++;
    }
    return sent;
  }

  /** Send {"action": "save"} — flush server buffer, keep session open. */
  sendSave() { return this.send({ action: 'save' }); }

  /** Send {"action": "end"} — flush server buffer and close session. */
  sendEnd()  { return this.send({ action: 'end' }); }

  get isConnected() { return this._ready; }

  _emit(status) { this.onStatusChange?.(status); }

  _scheduleReconnect() {
    if (this._timer) return;
    this._timer = setTimeout(() => {
      this._timer = null;
      appendLog('Attempting reconnect…', 'info');
      this.connect();
    }, CONFIG.WS_RECONNECT_MS);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   9. HOLISTIC CONTROLLER
   Initialises MediaPipe Holistic and the Camera utility.
   On every processed frame it calls LandmarkPipeline.extract() and emits
   the resulting 225-float array via the onFrame callback.
   ═══════════════════════════════════════════════════════════════════════ */
class HolisticController {
  constructor() {
    this._video    = $('#input-video');
    this._canvas   = $('#output-canvas');
    this._ctx      = this._canvas.getContext('2d');
    this._holistic = null;
    this._camera   = null;

    // Set true only while recording — gates the onFrame emission
    this._active = false;

    // Diagnostic counters (read by UIController for the debug grid)
    this.totalFrames  = 0;
    this.goodFrames   = 0;
    this.rhZeroFrames = 0;
    this.lhZeroFrames = 0;

    /** @type {function(number[], {rhDetected:boolean, lhDetected:boolean, poseDetected:boolean}): void} */
    this.onFrame  = null;
    /** @type {function(boolean, boolean, boolean): void} */
    this.onDetect = null;
    /** @type {function(Error): void} */
    this.onError  = null;
  }

  /** Initialise Holistic + Camera and request webcam permission. */
  async init() {
    this._setPill('⏳', 'Requesting camera…');

    // Keep canvas in sync with video resolution
    this._video.addEventListener('loadedmetadata', () => {
      this._canvas.width  = this._video.videoWidth  || 640;
      this._canvas.height = this._video.videoHeight || 480;
    });

    // Create and configure the Holistic model
    this._holistic = new Holistic({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    this._holistic.setOptions({
      modelComplexity:        CONFIG.MODEL_COMPLEXITY,
      smoothLandmarks:        true,
      enableSegmentation:     false,
      smoothSegmentation:     false,
      refineFaceLandmarks:    false,  // not needed for 225-float spec
      minDetectionConfidence: CONFIG.MIN_DETECTION,
      minTrackingConfidence:  CONFIG.MIN_TRACKING,
    });

    this._holistic.onResults((r) => this._onResults(r));

    // Start the camera loop
    try {
      this._camera = new Camera(this._video, {
        onFrame: async () => {
          await this._holistic.send({ image: this._video });
        },
        width: 640, height: 480,
        facingMode: 'user',
      });
      await this._camera.start();
      this._setPill('📷', 'Camera active — Holistic running');
      appendLog('MediaPipe Holistic initialised ✓', 'ok');
    } catch (err) {
      this._setPill('❌', 'Camera unavailable');
      appendLog(`Camera error: ${err.message}`, 'err');
      showToast('Camera permission denied', 'error');
      this.onError?.(err);
    }
  }

  /**
   * Allow (true) or block (false) the onFrame emission.
   * Called by UIController when recording starts / stops.
   */
  setActive(active) { this._active = active; }

  // ── Private ─────────────────────────────────────────────────────────

  _onResults(results) {
    this.totalFrames++;

    // Resize canvas if video dimensions changed
    const w = this._video.videoWidth  || 640;
    const h = this._video.videoHeight || 480;
    if (this._canvas.width  !== w) this._canvas.width  = w;
    if (this._canvas.height !== h) this._canvas.height = h;

    // Draw landmarks on the canvas overlay
    this._draw(results, w, h);

    // Run the 225-float extraction pipeline
    const { frame, rhDetected, lhDetected, poseDetected } =
      LandmarkPipeline.extract(results);

    // Update diagnostic counters
    if (frame.length === CONFIG.FRAME_SIZE) this.goodFrames++;
    if (!rhDetected) this.rhZeroFrames++;
    if (!lhDetected) this.lhZeroFrames++;

    // Notify UI of detection state (updates RH / LH / POSE chips)
    this.onDetect?.(rhDetected, lhDetected, poseDetected);

    // Emit the frame only when the user is actively recording
    if (this._active) {
      this.onFrame?.(frame, { rhDetected, lhDetected, poseDetected });
    }
  }

  _draw(results, w, h) {
    const ctx = this._ctx;
    ctx.save();
    ctx.clearRect(0, 0, w, h);

    // Pose skeleton (white, thin)
    if (results.poseLandmarks) {
      drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
        color: 'rgba(255,255,255,0.22)', lineWidth: 2,
      });
      drawLandmarks(ctx, results.poseLandmarks, {
        color: 'rgba(255,255,255,0.45)', fillColor: 'rgba(255,255,255,0.12)',
        lineWidth: 1, radius: 2,
      });
    }

    // Right hand (cyan)
    if (results.rightHandLandmarks) {
      drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, {
        color: 'rgba(0,229,200,0.8)', lineWidth: 2.5,
      });
      drawLandmarks(ctx, results.rightHandLandmarks, {
        color: '#00e5c8', fillColor: 'rgba(0,229,200,0.75)',
        lineWidth: 1, radius: 4,
      });
    }

    // Left hand (purple)
    if (results.leftHandLandmarks) {
      drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, {
        color: 'rgba(124,92,252,0.8)', lineWidth: 2.5,
      });
      drawLandmarks(ctx, results.leftHandLandmarks, {
        color: '#7c5cfc', fillColor: 'rgba(124,92,252,0.75)',
        lineWidth: 1, radius: 4,
      });
    }

    // Orange dot on nose — visual anchor for the normalisation reference
    const nose = results.poseLandmarks?.[CONFIG.NOSE_IDX];
    if (nose) {
      ctx.beginPath();
      ctx.arc(nose.x * w, nose.y * h, 6, 0, Math.PI * 2);
      ctx.fillStyle   = 'rgba(255,165,2,0.88)';
      ctx.strokeStyle = '#fff';
      ctx.lineWidth   = 1.5;
      ctx.fill();
      ctx.stroke();
    }

    ctx.restore();
  }

  _setPill(icon, text) {
    const i = $('#cam-pill-icon');
    const t = $('#cam-pill-text');
    if (i) i.textContent = icon;
    if (t) t.textContent = text;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   ACHOLI VOCABULARY
   Keys match backend gesture labels (lowercase/underscore).
   Translations are community-verified Acholi (Luo) terms.
   Entries marked ★ are Uganda Sign Language (USL) greetings confirmed via
   the SignMaster reference dataset.
   ═══════════════════════════════════════════════════════════════════════ */
const ACHOLI_DICT = {
  // ── Greetings (USL-verified ★) ────────────────────────────────────────
  hello:              'Iboŋo',           // ★
  goodbye:            'Wot ki kuc',      // ★
  good_morning:       'Odikinin maber',  // ★
  good_night:         'Dyewor maber',    // ★
  how_are_you:        'Ityeko nining',   // ★
  i_am_fine:          'Atye maber',      // ★
  thank_you:          'Apwoyo matek',    // ★
  please:             'Alegi',
  sorry:              'Tika',
  welcome:            'Ibin matek',      // ★
  congratulations:    'Gum ngolo',       // ★

  // ── Basic responses ───────────────────────────────────────────────────
  yes:        'Eyo',
  no:         'Ku',

  // ── Common verbs / actions ────────────────────────────────────────────
  help:       'Konnya',
  stop:       'Juk',
  go:         'Cit',
  come:       'Bin',
  wait:       'Kur',
  understand: 'Niang',
  repeat:     'Dok cobo',

  // ── Identity ──────────────────────────────────────────────────────────
  my_name_is: 'Nyinga tye',
  name:       'Nying',

  // ── Feelings ──────────────────────────────────────────────────────────
  good:   'Maber',
  bad:    'Marac',
  happy:  'Yom cwinyi',
  love:   'Maro',

  // ── People ────────────────────────────────────────────────────────────
  friend:  'Lareme',
  family:  'Kaka',
  mother:  'Mama',
  father:  'Baba',
  brother: 'Omin',
  sister:  'Lamin',
  child:   'Latin',

  // ── Everyday nouns ────────────────────────────────────────────────────
  water:  'Pii',
  food:   'Cam',
  home:   'Gang',
  school: 'Cuk',
  road:   'Yo',

  // ── Numbers (Acholi) ─────────────────────────────────────────────────
  one:   'Acel',
  two:   'Ariyo',
  three: 'Adek',
  four:  'Aŋwen',
  five:  'Abic',
};

function toAcholi(label) {
  if (!label) return '—';
  return ACHOLI_DICT[label.toLowerCase().trim()] ?? `[${label}]`;
}

function toEnglish(label) {
  return (label || 'Unknown')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/* ═══════════════════════════════════════════════════════════════════════════
   10. UI CONTROLLER
   Central state machine. Owns the recording state and wires every module
   to the correct DOM elements in index.html.
   ═══════════════════════════════════════════════════════════════════════ */
class UIController {
  constructor() {
    // ── Recording controls ────────────────────────────────────────────
    this._btnStart = $('#btn-start');
    this._btnStop  = $('#btn-stop');
    this._btnCopy  = $('#btn-copy');

    // ── Camera HUD ────────────────────────────────────────────────────
    this._frameBadge = $('#frame-badge');
    this._frameCount = $('#frame-count');

    // ── Detection chips ───────────────────────────────────────────────
    this._chipRh   = $('#ind-rh');
    this._chipLh   = $('#ind-lh');
    this._chipPose = $('#ind-pose');

    // ── WS status ─────────────────────────────────────────────────────
    this._wsDot   = $('#ws-dot');
    this._wsLabel = $('#ws-label');

    // ── Translation card ──────────────────────────────────────────────
    this._outEn     = $('#out-english');
    this._outAch    = $('#out-acholi');
    this._transCard = $('#trans-card');
    this._achBlock  = $('#lang-ach-block');
    this._langSep   = $('.lang-sep');

    // ── Debug grid ────────────────────────────────────────────────────
    this._dbgTotal  = $('#dbg-total');
    this._dbgFrames = $('#dbg-frames');
    this._dbgRhZero = $('#dbg-rh-zero');
    this._dbgLhZero = $('#dbg-lh-zero');
    this._dbgWsSent = $('#dbg-ws-sent');
    this._dbgBuf    = $('#dbg-buf');

    // ── Internal state ────────────────────────────────────────────────
    this._recording  = false;
    this._wsSent     = 0;
    this._showAcholi = true;

    // ── Modules ───────────────────────────────────────────────────────
    this._buffer   = new LandmarkBuffer(CONFIG.FRAME_WINDOW);
    this._ws       = new WebSocketClient(CONFIG.WS_URL);
    this._holistic = new HolisticController();

    this._wire();
  }

  _wire() {
    // ── Buffer → auto-stream every complete 30-frame window ───────────
    this._buffer.onFlush((frames) => {
      if (!this._recording) return;
      const sent = this._ws.streamFrames(frames);
      this._wsSent += sent;
      this._dbgWsSent.textContent = this._wsSent;
    });

    // ── WebSocket ─────────────────────────────────────────────────────
    this._ws.onStatusChange = (s) => this._onWsStatus(s);
    this._ws.onMessage      = (d) => this._onServerMsg(d);

    // ── Holistic frame emission ────────────────────────────────────────
    this._holistic.onFrame = (frame) => {
      this._buffer.push(frame);
      this._frameCount.textContent = this._buffer.length;
      this._frameBadge.classList.add('active');
      this._dbgBuf.textContent = `${this._buffer.length}/30`;
    };

    this._holistic.onDetect = (rh, lh, pose) => {
      this._setChip(this._chipRh,   rh,   'live', 'zeroed');
      this._setChip(this._chipLh,   lh,   'live', 'zeroed');
      this._setChip(this._chipPose, pose, 'live', 'zeroed');
      // Sync debug grid from controller counters
      this._dbgTotal.textContent  = this._holistic.totalFrames;
      this._dbgFrames.textContent = this._holistic.goodFrames;
      this._dbgRhZero.textContent = this._holistic.rhZeroFrames;
      this._dbgLhZero.textContent = this._holistic.lhZeroFrames;
    };

    this._holistic.onError = () => { this._btnStart.disabled = true; };

    // ── Buttons ───────────────────────────────────────────────────────
    this._btnStart.addEventListener('click', () => this._startRecording());
    this._btnStop.addEventListener('click',  () => this._stopRecording());
    this._btnCopy.addEventListener('click',  () => this._copyTranslation());

    // ── Language toggle ───────────────────────────────────────────────
    $('#lang-toggle-btn').addEventListener('click', () => this._toggleLang());

    // ── Nav tab highlights ────────────────────────────────────────────
    $$('.nav-item').forEach((btn) => {
      btn.addEventListener('click', () => {
        $$('.nav-item').forEach((b) => {
          b.classList.remove('active');
          b.removeAttribute('aria-current');
        });
        btn.classList.add('active');
        btn.setAttribute('aria-current', 'page');
      });
    });
  }

  /** Called from App.init() — connects WS and starts camera. */
  async start() {
    this._ws.connect();
    await this._holistic.init();
    this._btnStart.disabled = false;
    appendLog('SignBridge ready — 225-float Holistic pipeline active.', 'ok');
  }

  // ── Recording state machine ──────────────────────────────────────────

  _startRecording() {
    if (this._recording) return;
    this._recording = true;
    this._buffer.reset();
    this._holistic.setActive(true);

    this._btnStart.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <circle cx="12" cy="12" r="10"/>
      </svg>
      Start Translating`;
    this._btnStart.classList.add('recording');
    this._btnStart.disabled = true;
    this._btnStop.disabled  = false;
    this._frameBadge.classList.add('active');

    showToast('Translation started — stream live', 'info');
    appendLog('Translation started.', 'info');
  }


  _stopRecording() {
    if (!this._recording) return;
    this._recording = false;
    this._holistic.setActive(false);

    // Drain any partial window before closing
    const partial = this._buffer.drain();
    if (partial.length > 0) {
      const sent = this._ws.streamFrames(partial);
      this._wsSent += sent;
      this._dbgWsSent.textContent = this._wsSent;
    }
    this._ws.sendEnd();

    // Reset button to initial state
    this._btnStart.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <circle cx="12" cy="12" r="10"/>
      </svg>
      Start Translating`;
    this._btnStart.classList.remove('recording');
    this._btnStart.disabled = false;
    this._btnStop.disabled  = true;
    this._frameBadge.classList.remove('active');
    this._frameCount.textContent = '0';
    this._dbgBuf.textContent = '0/30';

    showToast('Session ended', 'success');
    appendLog('Translation stopped.', 'ok');
  }

  // ── WebSocket event handlers ─────────────────────────────────────────

  _onWsStatus(status) {
    const labels = { connected: 'Online', connecting: 'Connecting…', disconnected: 'Offline' };
    this._wsDot.className = `ws-dot ${status}`;
    this._wsDot.setAttribute('aria-label', labels[status] ?? status);
    this._wsLabel.textContent = labels[status] ?? status;
  }

  _onServerMsg(data) {
    if (data.status === 'translated' && data.text) {
      this._showTranslation(data.text);
    }
    if (data.status === 'error') {
      showToast(`Server: ${data.detail}`, 'error');
    }
  }

  // ── Translation display ──────────────────────────────────────────────

  _showTranslation(label, frames) {
    const en  = toEnglish(label);
    const ach = toAcholi(label);

    this._outEn.textContent  = en;
    this._outAch.textContent = ach;
    this._btnCopy.disabled   = false;

    // Briefly highlight the card
    this._transCard.classList.add('highlight');
    setTimeout(() => this._transCard.classList.remove('highlight'), 1_600);

    appendLog(`Translation: ${en} / ${ach}`, 'ok');

    // NEW: Speak the English translation aloud
    this._speakTranslation(en);
  }

  // ── Copy & language toggle ───────────────────────────────────────────

  _copyTranslation() {
    const text = `${this._outEn.textContent}\n${this._outAch.textContent}`;
    navigator.clipboard.writeText(text)
      .then(() => showToast('Copied to clipboard', 'success'))
      .catch(() => showToast('Copy failed', 'error'));
  }

  _toggleLang() {
    this._showAcholi = !this._showAcholi;
    this._achBlock.style.display = this._showAcholi ? '' : 'none';
    this._langSep.style.display  = this._showAcholi ? '' : 'none';
    showToast(this._showAcholi ? 'Showing EN + ACH' : 'English only', 'info');
  }

  // ── Audio Output ─────────────────────────────────────────────────────

  _speakTranslation(text) {
    // 1. Cancel any currently playing audio so rapid signs don't overlap
    window.speechSynthesis.cancel();

    // 2. Create the speech request
    const utterance = new SpeechSynthesisUtterance(text);

    // 3. Configure the voice (slightly slower rate usually sounds more natural)
    utterance.rate   = 0.95;
    utterance.pitch  = 1.0;
    utterance.volume = 1.0;

    // Optional: Try to find a good English voice if multiple are available
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(
      (v) => v.lang.startsWith('en-GB') || v.lang.startsWith('en-US'),
    );
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    // 4. Play the audio
    window.speechSynthesis.speak(utterance);
  }

  // ── Helpers ──────────────────────────────────────────────────────────

  /**
   * Update a detection chip's CSS class based on whether it is live or zeroed.
   * @param {HTMLElement} chip
   * @param {boolean}     detected
   * @param {string}      liveClass   - applied when detected === true
   * @param {string}      zeroedClass - applied when detected === false
   */
  _setChip(chip, detected, liveClass, zeroedClass) {
    chip.className = `det-chip ${detected ? liveClass : zeroedClass}`;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   11. APP INIT
   ═══════════════════════════════════════════════════════════════════════ */
const App = {
  async init() {
    // First-run tutorial
    const wizard = new OnboardingWizard();
    wizard.showIfNeeded();

    // Floating help drawer
    new HelpDrawer(wizard);

    // Settings panel (backend URL config)
    new SettingsPanel();

    // Main UI (also starts WS + Holistic)
    const ui = new UIController();
    await ui.start();
  },
};

// Boot once the DOM is fully parsed
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => App.init());
} else {
  App.init();
}
