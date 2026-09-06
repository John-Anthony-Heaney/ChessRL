/* ============================================================================
   ChessRL -- browser front-end.
   Vanilla ES2020, no build step, no dependencies, no network beyond /api/*.

   Contract: docs/API.md.  Legality is NEVER computed here -- every move the UI
   offers comes straight out of `state.legal`.
   ========================================================================== */

'use strict';

import { pieceSVG, pieceGlyph } from './pieces.js';

/* ========================================================================== */
/* helpers                                                                    */
/* ========================================================================== */

const $ = (id) => document.getElementById(id);
const FILES = 'abcdefgh';
const SQN = [];
for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) SQN.push(FILES[f] + (r + 1));
const SQI = {};
SQN.forEach((n, i) => { SQI[n] = i; });

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function fmtInt(n) {
  if (!isFinite(n)) return '—';
  return Math.round(n).toLocaleString('en-US');
}

function fmtCompact(n) {
  if (!isFinite(n)) return '—';
  const a = Math.abs(n);
  if (a >= 1e9) return (n / 1e9).toFixed(a >= 1e10 ? 0 : 1) + 'B';
  if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M';
  if (a >= 1e4) return (n / 1e3).toFixed(0) + 'k';
  if (a >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(Math.round(n));
}

function fmtNum(n, d) {
  if (n == null || !isFinite(n)) return '—';
  return n.toFixed(d == null ? (Math.abs(n) >= 100 ? 0 : Math.abs(n) >= 10 ? 1 : 2) : d);
}

function fmtSigned(n, d) {
  if (n == null || !isFinite(n)) return '—';
  const s = n.toFixed(d == null ? 2 : d);
  return n > 0 ? '+' + s : s;
}

function fmtPct(x, d) { return (x == null || !isFinite(x)) ? '—' : (x * 100).toFixed(d == null ? 1 : d) + '%'; }

function fmtMs(ms) {
  if (ms == null || !isFinite(ms)) return '—';
  return ms >= 1000 ? (ms / 1000).toFixed(2) + 's' : Math.round(ms) + 'ms';
}

function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
}

function store(key, val) {
  try {
    if (val === undefined) return localStorage.getItem('chessrl.' + key);
    localStorage.setItem('chessrl.' + key, val);
  } catch (_) { /* private mode, blocked storage -- ignore */ }
  return null;
}

/* ------------------------------------------------------------------ toasts */

const toastHost = $('toasts');

function toast(title, message, kind) {
  const t = el('div', 'toast ' + (kind || 'err'));
  t.innerHTML =
    '<div class="toast-body"><strong>' + esc(title) + '</strong>' +
    (message ? '<span>' + esc(message) + '</span>' : '') + '</div>' +
    '<button aria-label="Dismiss">&times;</button>';
  const kill = () => {
    if (!t.isConnected) return;
    t.classList.add('out');
    setTimeout(() => t.remove(), 200);
  };
  t.querySelector('button').addEventListener('click', kill);
  toastHost.appendChild(t);
  setTimeout(kill, kind === 'err' ? 7000 : 4000);
  while (toastHost.children.length > 4) toastHost.firstChild.remove();
}

/* --------------------------------------------------------------------- api */

class ApiError extends Error {
  constructor(msg, where) { super(msg); this.where = where; }
}

async function api(path, opts) {
  let res;
  try {
    res = await fetch(path, Object.assign({ cache: 'no-store' }, opts));
  } catch (_) {
    throw new ApiError('The server is not responding.', path);
  }
  let text = '';
  try { text = await res.text(); } catch (_) { text = ''; }
  let data = null;
  if (text) { try { data = JSON.parse(text); } catch (_) { data = null; } }
  if (!res.ok) {
    const msg = (data && (data.error || data.message)) || ('HTTP ' + res.status + ' ' + res.statusText);
    throw new ApiError(msg, path);
  }
  if (data === null) throw new ApiError('The server returned a malformed response.', path);
  return data;
}

function apiPost(path, body) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
}

function reportError(e, what) {
  const msg = e instanceof ApiError ? e.message : (e && e.message) || String(e);
  toast(what, msg, 'err');
  // eslint-disable-next-line no-console
  console.error(what, e);
}

/* ========================================================================== */
/* chess helpers -- DISPLAY ONLY.  Never used to decide legality.             */
/* ========================================================================== */

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const PIECE_VALUE = { P: 1, N: 3, B: 3, R: 5, Q: 9, K: 0 };

/** FEN -> { board: (code|null)[64], turn } with a1 at index 0. */
function parseFen(fen) {
  const board = new Array(64).fill(null);
  if (typeof fen !== 'string' || !fen) return { board, turn: 'white' };
  const parts = fen.trim().split(/\s+/);
  const rows = parts[0].split('/');
  for (let i = 0; i < rows.length && i < 8; i++) {
    const rank = 7 - i;
    let file = 0;
    for (const ch of rows[i]) {
      if (ch >= '1' && ch <= '8') { file += +ch; continue; }
      if (file > 7) break;
      const isWhite = ch === ch.toUpperCase();
      board[rank * 8 + file] = (isWhite ? 'w' : 'b') + ch.toUpperCase();
      file++;
    }
  }
  return { board, turn: parts[1] === 'b' ? 'black' : 'white' };
}

/** Apply an already-validated UCI move to a board array (mutates, returns it). */
function applyUci(board, uci) {
  if (typeof uci !== 'string' || uci.length < 4) return board;
  const from = SQI[uci.slice(0, 2)], to = SQI[uci.slice(2, 4)];
  if (from == null || to == null) return board;
  const p = board[from];
  board[from] = null;
  if (!p) { return board; }
  const color = p[0], type = p[1];
  if (type === 'P' && (from % 8) !== (to % 8) && board[to] == null) {
    board[(to % 8) + Math.floor(from / 8) * 8] = null;          /* en passant */
  }
  if (type === 'K' && Math.abs((from % 8) - (to % 8)) === 2) {   /* castling  */
    const base = Math.floor(from / 8) * 8;
    if (to % 8 === 6) { board[base + 5] = board[base + 7]; board[base + 7] = null; }
    else { board[base + 3] = board[base + 0]; board[base + 0] = null; }
  }
  const promo = uci.length > 4 ? uci[4].toUpperCase() : '';
  board[to] = promo ? color + promo : p;
  return board;
}

/** All positions of a game: index i = position after i plies. */
function replayBoards(startFen, moves) {
  const out = [];
  let cur = parseFen(startFen || START_FEN).board;
  out.push({ board: cur.slice(), last: null });
  for (let i = 0; i < moves.length; i++) {
    cur = applyUci(cur, moves[i]);
    out.push({
      board: cur.slice(),
      last: { from: moves[i].slice(0, 2), to: moves[i].slice(2, 4) }
    });
  }
  return out;
}

function findKing(board, color) {
  const want = (color === 'white' ? 'w' : 'b') + 'K';
  for (let i = 0; i < 64; i++) if (board[i] === want) return SQN[i];
  return null;
}

function materialOf(board) {
  let w = 0, b = 0;
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p) continue;
    const v = PIECE_VALUE[p[1]] || 0;
    if (p[0] === 'w') w += v; else b += v;
  }
  return { white: w, black: b };
}

/** Pieces missing from a board, relative to the standard array. */
function derivedCaptures(board) {
  const full = { P: 8, N: 2, B: 2, R: 2, Q: 1 };
  const have = { w: { P: 0, N: 0, B: 0, R: 0, Q: 0 }, b: { P: 0, N: 0, B: 0, R: 0, Q: 0 } };
  for (const p of board) if (p && have[p[0]] && have[p[0]][p[1]] != null) have[p[0]][p[1]]++;
  const out = { white: [], black: [] };
  for (const t of ['Q', 'R', 'B', 'N', 'P']) {
    for (let i = have.w[t]; i < full[t]; i++) out.black.push(t);        /* black took a white piece */
    for (let i = have.b[t]; i < full[t]; i++) out.white.push(t.toLowerCase());
  }
  return out;
}

/* ========================================================================== */
/* Board view                                                                 */
/* ========================================================================== */

const ORDER = { Q: 0, R: 1, B: 2, N: 3, P: 4, K: 5 };

class Board {
  /**
   * @param {HTMLElement} root .board element containing the five layers
   * @param {object} opts { interactive, onMove(uci), onSelect() }
   */
  constructor(root, opts) {
    this.root = root;
    this.opts = opts || {};
    this.sqLayer = root.querySelector('.squares');
    this.markLayer = root.querySelector('.marks');
    this.pieceLayer = root.querySelector('.pieces');
    this.hintLayer = root.querySelector('.hints');
    this.coordLayer = root.querySelector('.coords');
    this.promoEl = root.querySelector('.promo-overlay');

    this.flipped = false;
    this.enabled = false;
    this.board = new Array(64).fill(null);
    this.pieces = new Map();          /* square -> { el, code } */
    this.legal = new Map();           /* from -> Map(to -> [uci,...]) */
    this.selected = null;
    this.marks = {};
    this.drag = null;
    this.promo = null;

    this.buildSquares();
    this.buildCoords();
    if (this.opts.interactive) this.attach();
  }

  /* ------------------------------------------------------------ geometry */

  buildSquares() {
    const frag = document.createDocumentFragment();
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        /* Rotating the board 180 degrees preserves the checker parity, so the
           squares layer never has to be rebuilt when the board is flipped. */
        const dark = (col + row) % 2 === 1;
        frag.appendChild(el('div', 'sq ' + (dark ? 'd' : 'l')));
      }
    }
    this.sqLayer.appendChild(frag);
  }

  buildCoords() {
    const c = this.coordLayer;
    c.textContent = '';
    for (let i = 0; i < 8; i++) {
      const f = el('span', 'cf');
      f.style.left = (i * 12.5) + '%';
      c.appendChild(f);
      const r = el('span', 'cr');
      r.style.top = (i * 12.5) + '%';
      c.appendChild(r);
    }
    this.updateCoords();
  }

  updateCoords() {
    const spans = this.coordLayer.children;
    for (let i = 0; i < 8; i++) {
      const fileSpan = spans[i * 2], rankSpan = spans[i * 2 + 1];
      const file = this.flipped ? 7 - i : i;
      const bottomRank = this.flipped ? 7 : 0;
      fileSpan.textContent = FILES[file];
      fileSpan.className = 'cf ' + ((file + bottomRank) % 2 === 0 ? 'on-d' : 'on-l');
      const rank = this.flipped ? i : 7 - i;
      const leftFile = this.flipped ? 7 : 0;
      rankSpan.textContent = String(rank + 1);
      rankSpan.className = 'cr ' + ((leftFile + rank) % 2 === 0 ? 'on-d' : 'on-l');
    }
  }

  setFlipped(v) {
    if (this.flipped === v) return;
    this.flipped = v;
    this.updateCoords();
    this.noAnim(() => {
      for (const [sq, rec] of this.pieces) this.place(rec.el, sq);
      this.renderMarks();
      this.renderHints();
    });
    if (this.promo) this.positionPromo();
  }

  colRow(sq) {
    const f = FILES.indexOf(sq[0]);
    const r = +sq[1] - 1;
    return this.flipped ? [7 - f, r] : [f, 7 - r];
  }

  place(node, sq) {
    const [c, r] = this.colRow(sq);
    node.style.transform = 'translate(' + (c * 100) + '%,' + (r * 100) + '%)';
  }

  squareAt(clientX, clientY) {
    const b = this.root.getBoundingClientRect();
    if (!b.width) return null;
    const x = clientX - b.left, y = clientY - b.top;
    if (x < 0 || y < 0 || x > b.width || y > b.height) return null;
    const col = clamp(Math.floor(x / (b.width / 8)), 0, 7);
    const row = clamp(Math.floor(y / (b.height / 8)), 0, 7);
    const f = this.flipped ? 7 - col : col;
    const r = this.flipped ? row : 7 - row;
    return FILES[f] + (r + 1);
  }

  noAnim(fn) {
    this.root.classList.add('no-anim');
    fn();
    void this.root.offsetWidth;                     /* force a reflow */
    this.root.classList.remove('no-anim');
  }

  /* ------------------------------------------------------------- pieces */

  makePiece(code, sq) {
    const node = el('div', 'piece appearing', pieceSVG(code));
    this.place(node, sq);
    this.pieceLayer.appendChild(node);
    setTimeout(() => node.classList.remove('appearing'), 160);
    return { el: node, code: code };
  }

  dropPiece(rec) {
    rec.el.classList.add('fading');
    setTimeout(() => rec.el.remove(), 160);
  }

  /**
   * Diff the current board against `next` and animate the difference.
   * @param {(string|null)[]} next 64 entries, a1 first
   * @param {object} meta { last: {from,to}|null, animate: boolean }
   */
  setPosition(next, meta) {
    meta = meta || {};
    const apply = () => {
      const prev = new Map(this.pieces);
      const pending = new Map();
      for (let i = 0; i < 64; i++) if (next[i]) pending.set(SQN[i], next[i]);
      const result = new Map();

      /* 1. squares whose occupant did not change */
      for (const [sq, code] of Array.from(pending)) {
        const rec = prev.get(sq);
        if (rec && rec.code === code) {
          result.set(sq, rec); prev.delete(sq); pending.delete(sq);
        }
      }
      /* 2. the move the server just reported -- also covers promotion,
            where the arriving code differs from the departing one */
      const last = meta.last;
      if (last && last.from && last.to && pending.has(last.to) && prev.has(last.from)) {
        const rec = prev.get(last.from);
        const code = pending.get(last.to);
        if (rec.code !== code) { rec.el.innerHTML = pieceSVG(code); rec.code = code; }
        rec.el.style.zIndex = '4';
        this.place(rec.el, last.to);
        result.set(last.to, rec);
        prev.delete(last.from); pending.delete(last.to);
        setTimeout(() => { rec.el.style.zIndex = ''; }, 180);
      }
      /* 3. anything else: nearest piece of the same kind (castling rook, undo) */
      for (const [sq, code] of Array.from(pending)) {
        let best = null, bestD = Infinity;
        const ti = SQI[sq];
        for (const [osq, rec] of prev) {
          if (rec.code !== code) continue;
          const oi = SQI[osq];
          const d = Math.abs((oi & 7) - (ti & 7)) + Math.abs((oi >> 3) - (ti >> 3));
          if (d < bestD) { bestD = d; best = osq; }
        }
        if (best != null) {
          const rec = prev.get(best);
          this.place(rec.el, sq);
          result.set(sq, rec);
          prev.delete(best); pending.delete(sq);
        }
      }
      /* 4. leftovers */
      for (const rec of prev.values()) this.dropPiece(rec);
      for (const [sq, code] of pending) result.set(sq, this.makePiece(code, sq));

      this.pieces = result;
      this.board = next.slice();
    };

    if (meta.animate === false) this.noAnim(apply); else apply();
  }

  /* -------------------------------------------------------------- marks */

  setMarks(m) { this.marks = m || {}; this.renderMarks(); }

  renderMarks() {
    const m = this.marks;
    const frag = document.createDocumentFragment();
    const add = (sq, cls) => {
      if (!sq || SQI[sq] == null) return;
      const n = el('div', 'mark ' + cls);
      this.place(n, sq);
      frag.appendChild(n);
    };
    if (m.last) { add(m.last.from, 'last'); add(m.last.to, 'last'); }
    if (m.hint) { add(m.hint.from, 'hintm'); add(m.hint.to, 'hintm'); }
    if (m.sel) add(m.sel, 'sel');
    if (m.check) add(m.check, 'check');
    this.markLayer.textContent = '';
    this.markLayer.appendChild(frag);
  }

  /* -------------------------------------------------------------- hints */

  setLegal(list) {
    this.legal = new Map();
    if (!Array.isArray(list)) return;
    for (const uci of list) {
      if (typeof uci !== 'string' || uci.length < 4) continue;
      const from = uci.slice(0, 2), to = uci.slice(2, 4);
      if (SQI[from] == null || SQI[to] == null) continue;
      let m = this.legal.get(from);
      if (!m) { m = new Map(); this.legal.set(from, m); }
      const arr = m.get(to);
      if (arr) arr.push(uci); else m.set(to, [uci]);
    }
  }

  targets(from) { return this.legal.get(from) || null; }

  renderHints() {
    this.hintLayer.textContent = '';
    if (!this.selected || !this.enabled) return;
    const t = this.targets(this.selected);
    if (!t) return;
    const src = this.board[SQI[this.selected]];
    const frag = document.createDocumentFragment();
    for (const to of t.keys()) {
      const occupied = !!this.board[SQI[to]];
      const epLike = src && src[1] === 'P' && this.selected[0] !== to[0] && !occupied;
      const n = el('div', 'hint' + (occupied || epLike ? ' cap' : ''));
      n.dataset.sq = to;
      this.place(n, to);
      frag.appendChild(n);
    }
    this.hintLayer.appendChild(frag);
  }

  select(sq) {
    this.selected = sq;
    const m = Object.assign({}, this.marks, { sel: sq });
    this.marks = m;
    this.renderMarks();
    this.renderHints();
  }

  setEnabled(v) {
    this.enabled = !!v;
    this.root.classList.toggle('is-live', this.enabled);
    if (!v) { this.cancelDrag(); this.select(null); }
    this.updateCursors();
  }

  updateCursors() {
    for (const [sq, rec] of this.pieces) {
      rec.el.classList.toggle('grabbable', this.enabled && this.legal.has(sq));
    }
  }

  /* ------------------------------------------------------------ pointer */

  attach() {
    this.root.addEventListener('pointerdown', (e) => this.onDown(e));
    this.root.addEventListener('pointermove', (e) => this.onMove(e));
    this.root.addEventListener('pointerup', (e) => this.onUp(e));
    this.root.addEventListener('pointercancel', () => this.cancelDrag());
    this.root.addEventListener('contextmenu', (e) => {
      if (this.selected || this.drag) { e.preventDefault(); this.cancelDrag(); this.select(null); }
    });
  }

  onDown(e) {
    if (this.promo) return;
    if (!this.enabled || e.button > 0) return;
    const sq = this.squareAt(e.clientX, e.clientY);
    if (!sq) return;
    e.preventDefault();

    if (this.selected) {
      const t = this.targets(this.selected);
      if (t && t.has(sq)) { this.commit(this.selected, sq); return; }
    }
    const rec = this.pieces.get(sq);
    if (rec && this.legal.has(sq)) {
      const wasSelected = this.selected === sq;
      this.select(sq);
      this.drag = {
        from: sq, rec: rec, id: e.pointerId,
        x0: e.clientX, y0: e.clientY, active: false, wasSelected: wasSelected
      };
      try { this.root.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    } else {
      this.select(null);
    }
  }

  onMove(e) {
    const d = this.drag;
    if (!d || e.pointerId !== d.id) return;
    if (!d.active) {
      if (Math.abs(e.clientX - d.x0) + Math.abs(e.clientY - d.y0) < 5) return;
      d.active = true;
      d.rec.el.classList.add('dragging');
      d.ghost = el('div', 'piece ghost', d.rec.el.innerHTML);
      this.place(d.ghost, d.from);
      this.pieceLayer.insertBefore(d.ghost, d.rec.el);
    }
    const b = this.root.getBoundingClientRect();
    const s = b.width / 8;
    d.rec.el.style.transform =
      'translate(' + (e.clientX - b.left - s / 2) + 'px,' + (e.clientY - b.top - s / 2) + 'px) scale(1.06)';

    const over = this.squareAt(e.clientX, e.clientY);
    if (over !== d.over) {
      d.over = over;
      for (const n of this.hintLayer.children) n.classList.toggle('hot', n.dataset.sq === over);
    }
  }

  onUp(e) {
    const d = this.drag;
    if (!d || e.pointerId !== d.id) return;
    this.drag = null;
    try { this.root.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    const sq = this.squareAt(e.clientX, e.clientY);

    if (d.active) {
      if (d.ghost) d.ghost.remove();
      d.rec.el.classList.remove('dragging');
      const t = this.targets(d.from);
      if (sq && t && t.has(sq)) {
        this.noAnim(() => this.place(d.rec.el, d.from));
        this.commit(d.from, sq);
      } else {
        this.place(d.rec.el, d.from);            /* animated snap-back */
        for (const n of this.hintLayer.children) n.classList.remove('hot');
      }
    } else if (d.wasSelected && sq === d.from) {
      this.select(null);                         /* click the same piece twice */
    }
  }

  cancelDrag() {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    if (d.ghost) d.ghost.remove();
    d.rec.el.classList.remove('dragging');
    this.place(d.rec.el, d.from);
  }

  /* ------------------------------------------------------------- commit */

  commit(from, to) {
    const t = this.targets(from);
    const options = t && t.get(to);
    if (!options || !options.length) { this.select(null); return; }
    if (options.length === 1) {
      this.select(null);
      if (this.opts.onMove) this.opts.onMove(options[0]);
      return;
    }
    this.openPromo(from, to, options);
  }

  /* ---------------------------------------------------------- promotion */

  openPromo(from, to, options) {
    const color = (this.board[SQI[from]] || 'w')[0];
    const order = ['q', 'r', 'b', 'n'];
    const avail = order.filter((c) => options.some((u) => u.length > 4 && u[4].toLowerCase() === c));
    if (!avail.length) { if (this.opts.onMove) this.opts.onMove(options[0]); return; }

    this.promo = { from: from, to: to, options: options };
    const picker = el('div', 'promo-picker');
    for (const c of avail) {
      const b = el('button', '', pieceSVG(color + c.toUpperCase()));
      b.title = 'Promote to ' + { q: 'queen', r: 'rook', b: 'bishop', n: 'knight' }[c];
      b.addEventListener('click', (ev) => {
        ev.stopPropagation();
        const uci = options.find((u) => u.length > 4 && u[4].toLowerCase() === c) || options[0];
        this.closePromo();
        this.select(null);
        if (this.opts.onMove) this.opts.onMove(uci);
      });
      picker.appendChild(b);
    }
    const cancel = el('button', 'promo-cancel', 'esc');
    cancel.addEventListener('click', (ev) => { ev.stopPropagation(); this.closePromo(); });
    picker.appendChild(cancel);

    const scrim = el('div', 'scrim');
    scrim.addEventListener('pointerdown', (ev) => { ev.stopPropagation(); this.closePromo(); });

    this.promoEl.textContent = '';
    this.promoEl.appendChild(scrim);
    this.promoEl.appendChild(picker);
    this.promoEl.hidden = false;
    this.promo.picker = picker;
    this.positionPromo();
  }

  positionPromo() {
    if (!this.promo || !this.promo.picker) return;
    const [c, r] = this.colRow(this.promo.to);
    const p = this.promo.picker;
    p.style.left = (c * 12.5) + '%';
    const n = p.children.length - 1;          /* piece buttons, cancel excluded */
    const downRoom = 8 - r;
    if (downRoom >= n) { p.style.top = (r * 12.5) + '%'; p.style.bottom = 'auto'; }
    else { p.style.bottom = ((7 - r) * 12.5) + '%'; p.style.top = 'auto'; }
  }

  closePromo() {
    this.promo = null;
    this.promoEl.hidden = true;
    this.promoEl.textContent = '';
  }
}

/* ========================================================================== */
/* charts -- plain inline SVG, no library                                     */
/* ========================================================================== */

const CHART_COLORS = ['var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)', 'var(--c5)', 'var(--c6)'];

function niceTicks(min, max, count) {
  if (!isFinite(min) || !isFinite(max)) return [0, 1];
  if (min === max) { min -= 0.5; max += 0.5; }
  const span = max - min;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out.length ? out : [min, max];
}

/**
 * @param {object} o { xs, series:[{name,color,values}], width, height, yMin, yMax,
 *                     fmtY, fmtX, area, zeroLine, hit }
 * @returns {string} svg markup
 */
function lineChart(o) {
  const W = o.width || 360, H = o.height || 170;
  const pad = Object.assign({ l: 34, r: 8, t: 8, b: 20 }, o.pad);
  const xs = o.xs, n = xs.length;
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  if (!n) return '<svg viewBox="0 0 ' + W + ' ' + H + '"></svg>';

  let lo = o.yMin, hi = o.yMax;
  if (lo == null || hi == null) {
    let mn = Infinity, mx = -Infinity;
    for (const s of o.series) for (const v of s.values) {
      if (v == null || !isFinite(v)) continue;
      if (v < mn) mn = v; if (v > mx) mx = v;
    }
    if (!isFinite(mn)) { mn = 0; mx = 1; }
    const pad2 = (mx - mn) * 0.12 || Math.abs(mx) * 0.1 || 1;
    lo = lo == null ? mn - pad2 : lo;
    hi = hi == null ? mx + pad2 : hi;
  }
  if (hi === lo) { hi = lo + 1; }

  const xmin = xs[0], xmax = xs[n - 1] === xs[0] ? xs[0] + 1 : xs[n - 1];
  const X = (v) => pad.l + ((v - xmin) / (xmax - xmin)) * pw;
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * ph;

  const fmtY = o.fmtY || ((v) => fmtNum(v));
  const fmtX = o.fmtX || ((v) => String(Math.round(v)));

  let s = '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet">';

  for (const t of niceTicks(lo, hi, 4)) {
    const y = Y(t);
    if (y < pad.t - 1 || y > H - pad.b + 1) continue;
    s += '<line class="grid-line' + (t === 0 && o.zeroLine ? ' zero' : '') + '" x1="' + pad.l +
      '" y1="' + y.toFixed(1) + '" x2="' + (W - pad.r) + '" y2="' + y.toFixed(1) + '"/>';
    s += '<text x="' + (pad.l - 5) + '" y="' + (y + 3.2).toFixed(1) + '" text-anchor="end">' +
      esc(fmtY(t)) + '</text>';
  }
  const xticks = [];
  const steps = Math.min(5, Math.max(2, n));
  for (let i = 0; i < steps; i++) xticks.push(xs[Math.round(i * (n - 1) / (steps - 1))]);
  for (const t of Array.from(new Set(xticks))) {
    s += '<text x="' + X(t).toFixed(1) + '" y="' + (H - pad.b + 12) + '" text-anchor="middle">' +
      esc(fmtX(t)) + '</text>';
  }
  s += '<line class="ax-line" x1="' + pad.l + '" y1="' + (H - pad.b) + '" x2="' + (W - pad.r) +
    '" y2="' + (H - pad.b) + '"/>';

  o.series.forEach((ser, si) => {
    const color = ser.color || CHART_COLORS[si % CHART_COLORS.length];
    let d = '', open = false, area = '';
    for (let i = 0; i < n; i++) {
      const v = ser.values[i];
      if (v == null || !isFinite(v)) { open = false; continue; }
      const x = X(xs[i]).toFixed(1), y = Y(clamp(v, lo, hi)).toFixed(1);
      d += (open ? 'L' : 'M') + x + ',' + y + ' ';
      area += (open ? 'L' : 'M') + x + ',' + y + ' ';
      open = true;
    }
    if (o.area) {
      const base = Y(clamp(0, lo, hi)).toFixed(1);
      const first = X(xs[0]).toFixed(1), lastX = X(xs[n - 1]).toFixed(1);
      s += '<path class="area" d="' + area + 'L' + lastX + ',' + base + ' L' + first + ',' + base +
        ' Z" style="fill:' + color + ';opacity:.12"/>';
    }
    s += '<path class="series" d="' + d.trim() + '" style="stroke:' + color + '"/>';
  });

  if (o.markerIndex != null && o.markerIndex >= 0 && o.markerIndex < n) {
    const v = o.series[0].values[o.markerIndex];
    const x = X(xs[o.markerIndex]).toFixed(1);
    s += '<line class="ax-line" x1="' + x + '" y1="' + pad.t + '" x2="' + x + '" y2="' + (H - pad.b) +
      '" style="stroke:var(--accent);opacity:.55"/>';
    if (v != null && isFinite(v)) {
      s += '<circle class="marker" cx="' + x + '" cy="' + Y(clamp(v, lo, hi)).toFixed(1) + '" r="3"/>';
    }
  }
  if (o.hit) {
    s += '<rect class="plot-hit" x="' + pad.l + '" y="' + pad.t + '" width="' + pw + '" height="' + ph +
      '" fill="transparent" style="cursor:crosshair"/>';
  }
  return s + '</svg>';
}

/** Stacked area, values are fractions of 1 per layer. */
function stackedArea(o) {
  const W = o.width || 520, H = o.height || 170;
  const pad = Object.assign({ l: 34, r: 8, t: 8, b: 20 }, o.pad);
  const xs = o.xs, n = xs.length;
  if (!n) return '<svg viewBox="0 0 ' + W + ' ' + H + '"></svg>';
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const xmin = xs[0], xmax = xs[n - 1] === xs[0] ? xs[0] + 1 : xs[n - 1];
  const X = (v) => pad.l + ((v - xmin) / (xmax - xmin)) * pw;
  const Y = (v) => pad.t + (1 - v) * ph;

  let s = '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet">';
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    s += '<line class="grid-line" x1="' + pad.l + '" y1="' + Y(t).toFixed(1) + '" x2="' + (W - pad.r) +
      '" y2="' + Y(t).toFixed(1) + '"/>' +
      '<text x="' + (pad.l - 5) + '" y="' + (Y(t) + 3.2).toFixed(1) + '" text-anchor="end">' +
      Math.round(t * 100) + '%</text>';
  }
  const base = new Array(n).fill(0);
  o.layers.forEach((layer, li) => {
    const color = layer.color || CHART_COLORS[li % CHART_COLORS.length];
    let top = '', bot = '';
    for (let i = 0; i < n; i++) {
      const v = clamp(layer.values[i] || 0, 0, 1);
      const y0 = base[i], y1 = base[i] + v;
      top += (i ? 'L' : 'M') + X(xs[i]).toFixed(1) + ',' + Y(clamp(y1, 0, 1)).toFixed(1) + ' ';
      bot = 'L' + X(xs[i]).toFixed(1) + ',' + Y(clamp(y0, 0, 1)).toFixed(1) + ' ' + bot;
      base[i] = y1;
    }
    s += '<path d="' + top + bot + 'Z" style="fill:' + color + ';opacity:.82"/>';
  });
  const steps = Math.min(5, Math.max(2, n));
  const seen = new Set();
  for (let i = 0; i < steps; i++) {
    const t = xs[Math.round(i * (n - 1) / (steps - 1))];
    if (seen.has(t)) continue;
    seen.add(t);
    s += '<text x="' + X(t).toFixed(1) + '" y="' + (H - pad.b + 12) + '" text-anchor="middle">' +
      esc(String(Math.round(t))) + '</text>';
  }
  return s + '</svg>';
}

function legend(items) {
  return '<div class="legend">' + items.map((i) =>
    '<span class="legend-item"><span class="legend-swatch" style="background:' + i.color + '"></span>' +
    esc(i.name) + (i.value != null ? ' <b style="color:var(--text);font-weight:600">' + esc(i.value) + '</b>' : '') +
    '</span>').join('') + '</div>';
}

/* ========================================================================== */
/* PLAY view                                                                  */
/* ========================================================================== */

const LEVELS = {
  1: { depth: 1, movetime: 120, blunder: 0.35, name: 'Novice' },
  2: { depth: 2, movetime: 250, blunder: 0.18, name: 'Casual' },
  3: { depth: 3, movetime: 450, blunder: 0.07, name: 'Club' },
  4: { depth: 4, movetime: 800, blunder: 0.02, name: 'Strong' },
  5: { depth: 5, movetime: 1500, blunder: 0.0, name: 'Champion' }
};

const REASON_TEXT = {
  checkmate: 'Checkmate',
  stalemate: 'Stalemate',
  'fifty-move': 'Draw by the fifty-move rule',
  fifty: 'Draw by the fifty-move rule',
  'threefold repetition': 'Draw by threefold repetition',
  repetition: 'Draw by repetition',
  'insufficient material': 'Draw by insufficient material',
  insufficient: 'Draw by insufficient material',
  'move limit': 'Draw — move limit reached',
  maxplies: 'Draw — move limit reached',
  adjudicated: 'Adjudicated',
  resign: 'Resignation'
};

const Play = {
  board: null,
  gid: null,
  state: null,
  startFen: START_FEN,
  human: 'white',
  level: 4,
  flipped: false,
  busy: false,
  viewPly: null,        /* null = live */
  resigned: null,
  engine: null,
  hint: null,
  positions: null,

  init() {
    this.board = new Board($('board'), {
      interactive: true,
      onMove: (uci) => this.playHuman(uci)
    });

    $('btn-new').addEventListener('click', () => this.newGame());
    $('banner-again').addEventListener('click', () => this.newGame());
    $('btn-undo').addEventListener('click', () => this.undo());
    $('btn-flip').addEventListener('click', () => this.setFlipped(!this.flipped));
    $('btn-hint').addEventListener('click', () => this.askHint());
    $('btn-resign').addEventListener('click', () => this.resign());
    $('sel-level').addEventListener('change', () => {
      this.level = +$('sel-level').value;
      store('level', this.level);
      this.paintNames();
    });

    $('mv-start').addEventListener('click', () => this.goto(0));
    $('mv-prev').addEventListener('click', () => this.goto(this.curPly() - 1));
    $('mv-next').addEventListener('click', () => this.goto(this.curPly() + 1));
    $('mv-end').addEventListener('click', () => this.goto(null));
    $('movelist').addEventListener('click', (e) => {
      const b = e.target.closest('.mv');
      if (b && b.dataset.ply) this.goto(+b.dataset.ply);
    });

    const savedLevel = store('level');
    if (savedLevel && LEVELS[+savedLevel]) {
      this.level = +savedLevel; $('sel-level').value = String(this.level);
    }
    const savedColor = store('color');
    if (savedColor) $('sel-color').value = savedColor;

    this.paintNames();
    this.newGame();
  },

  /* ------------------------------------------------------------- game flow */

  async newGame() {
    let want = $('sel-color').value;
    store('color', want);
    if (want === 'random') want = Math.random() < 0.5 ? 'white' : 'black';
    this.level = +$('sel-level').value || 4;

    this.busy = true;
    this.setThinking(false);
    try {
      const r = await api('/api/new?human=' + encodeURIComponent(want));
      const st = r && r.state ? r.state : r;
      if (!st || !st.fen) throw new ApiError('The server did not return a game state.', '/api/new');
      this.gid = r.gid != null ? r.gid : st.gid;
      this.human = want;
      this.startFen = st.fen;
      this.resigned = null;
      this.engine = null;
      this.hint = null;
      this.viewPly = null;
      this.positions = null;
      this.setFlipped(want === 'black');
      this.board.noAnim(() => { });
      this.clearEngine();
      this.apply(st, false);
      toast('New game', 'You are ' + want + ' against the champion at level ' + this.level + '.', 'info');
    } catch (e) {
      reportError(e, 'Could not start a game');
      this.gid = null;
    } finally {
      this.busy = false;
      this.maybeEngine();
    }
  },

  async playHuman(uci) {
    if (this.gid == null || this.busy) return;
    this.busy = true;
    this.hint = null;
    try {
      const r = await apiPost('/api/move', { gid: this.gid, move: uci });
      const st = r && r.state ? r.state : r;
      this.apply(st, true);
    } catch (e) {
      reportError(e, 'Move rejected');
      await this.refresh();
    } finally {
      this.busy = false;
      this.maybeEngine();
    }
  },

  maybeEngine() {
    if (this.gid == null || this.busy || !this.state) return;
    if (this.state.result) return;
    if (this.resigned) return;
    if (this.state.turn === this.human) return;
    this.engineMove();
  },

  async engineMove() {
    const lv = LEVELS[this.level] || LEVELS[4];
    this.busy = true;
    this.setThinking(true);
    const t0 = performance.now();
    try {
      const r = await apiPost('/api/ai', {
        gid: this.gid, depth: lv.depth, movetime: lv.movetime, blunder: lv.blunder
      });
      this.engine = {
        move: r.move, san: r.san, score: r.score, depth: r.depth, nodes: r.nodes,
        ms: r.ms != null ? r.ms : Math.round(performance.now() - t0),
        value: r.value, top: Array.isArray(r.top) ? r.top : [],
        color: this.state ? this.state.turn : (this.human === 'white' ? 'black' : 'white')
      };
      this.paintEngine();
      const st = r && r.state ? r.state : null;
      if (st) this.apply(st, true); else await this.refresh();
    } catch (e) {
      reportError(e, 'The engine failed to move');
    } finally {
      this.setThinking(false);
      this.busy = false;
      if (this.state && !this.state.result && !this.resigned && this.state.turn !== this.human) {
        /* the engine still has to move (e.g. after a failed attempt) -- do not
           loop: leave it to the user to retry via undo / new game. */
      }
    }
  },

  async refresh() {
    if (this.gid == null) return;
    try {
      const st = await api('/api/state?gid=' + encodeURIComponent(this.gid));
      this.apply(st && st.state ? st.state : st, false);
    } catch (e) { reportError(e, 'Could not read the game state'); }
  },

  async undo() {
    if (this.gid == null || this.busy || !this.state) return;
    const played = (this.state.moves || []).length;
    if (!played) return;
    this.busy = true;
    this.setThinking(false);
    try {
      let plies = Math.min(2, played);
      let r = await apiPost('/api/undo', { gid: this.gid, plies: plies });
      let st = r && r.state ? r.state : r;
      /* make sure it is the human's turn again */
      let guard = 0;
      while (st && !st.result && st.turn !== this.human && (st.moves || []).length && guard++ < 2) {
        r = await apiPost('/api/undo', { gid: this.gid, plies: 1 });
        st = r && r.state ? r.state : r;
      }
      this.resigned = null;
      this.engine = null;
      this.hint = null;
      this.viewPly = null;
      this.clearEngine();
      this.apply(st, true);
    } catch (e) {
      reportError(e, 'Undo failed');
    } finally {
      this.busy = false;
      this.maybeEngine();
    }
  },

  async askHint() {
    if (this.gid == null || this.busy || !this.state || this.state.result) return;
    if (this.state.turn !== this.human) return;
    this.busy = true;
    this.setThinking(true);
    try {
      const r = await api('/api/hint?gid=' + encodeURIComponent(this.gid));
      if (r && typeof r.move === 'string' && r.move.length >= 4) {
        this.hint = { from: r.move.slice(0, 2), to: r.move.slice(2, 4), san: r.san };
        this.paintMarks();
        toast('Hint', (r.san || r.move) + ' looks best to the champion.', 'info');
      } else {
        toast('Hint', 'The engine had no suggestion.', 'info');
      }
    } catch (e) {
      reportError(e, 'Hint failed');
    } finally {
      this.setThinking(false);
      this.busy = false;
    }
  },

  resign() {
    if (!this.state || this.state.result || this.resigned) return;
    this.resigned = this.human;
    this.board.setEnabled(false);
    this.paintBanner();
  },

  /* --------------------------------------------------------------- render */

  apply(st, animate) {
    if (!st || !st.fen) return;
    this.state = st;
    this.positions = null;
    this.viewPly = null;
    this.hint = null;
    this.render(animate !== false);
  },

  curPly() {
    if (!this.state) return 0;
    return this.viewPly == null ? (this.state.moves || []).length : this.viewPly;
  },

  allPositions() {
    if (!this.positions) {
      this.positions = replayBoards(this.startFen, (this.state && this.state.moves) || []);
    }
    return this.positions;
  },

  goto(ply) {
    if (!this.state) return;
    const total = (this.state.moves || []).length;
    if (ply == null || ply >= total) this.viewPly = null;
    else this.viewPly = clamp(ply, 0, total);
    this.render(true);
  },

  render(animate) {
    const st = this.state;
    if (!st) return;
    const live = this.viewPly == null;
    const moves = st.moves || [];
    let board, last;

    if (live) {
      const p = parseFen(st.fen);
      board = p.board;
      last = st.last || null;
    } else {
      const pos = this.allPositions()[clamp(this.viewPly, 0, moves.length)];
      board = pos.board;
      last = pos.last;
    }

    this.board.setLegal(live ? (st.legal || []) : []);
    this.board.setPosition(board, { last: last, animate: animate !== false });

    const canPlay = live && !st.result && !this.resigned && st.turn === this.human;
    this.board.setEnabled(canPlay);
    this.board.updateCursors();

    this.lastBoard = board;
    this.paintMarks();
    this.paintPlayers(board);
    this.paintMoves();
    this.paintBanner();
    this.paintEval();
    $('btn-hint').disabled = !canPlay;
    $('btn-undo').disabled = !live || !moves.length || this.busy;
    $('btn-resign').disabled = !!st.result || !!this.resigned;
    $('mv-prev').disabled = this.curPly() <= 0;
    $('mv-start').disabled = this.curPly() <= 0;
    $('mv-next').disabled = live;
    $('mv-end').disabled = live;
  },

  paintMarks() {
    const st = this.state;
    if (!st) return;
    const live = this.viewPly == null;
    const board = this.lastBoard || parseFen(st.fen).board;
    let last = live ? st.last : this.allPositions()[this.curPly()].last;
    const marks = { last: last || null, sel: this.board.selected };
    if (live && st.check && !st.result) marks.check = findKing(board, st.turn);
    if (live && this.hint) marks.hint = this.hint;
    this.board.setMarks(marks);
  },

  paintNames() {
    const lv = LEVELS[this.level] || LEVELS[4];
    this.engineLabel = 'Champion';
    this.engineSub = 'depth ' + lv.depth + ' · ' + lv.name;
  },

  paintPlayers(board) {
    this.paintNames();
    const st = this.state;
    const humanTop = this.flipped ? (this.human === 'white') : (this.human === 'black');
    const topIsHuman = humanTop;
    const topColor = this.flipped ? 'white' : 'black';
    const botColor = this.flipped ? 'black' : 'white';

    const setStrip = (suffix, color, isHuman) => {
      $('name-' + suffix).textContent = isHuman ? 'You' : this.engineLabel;
      $('sub-' + suffix).textContent = isHuman
        ? (color === 'white' ? 'white · human' : 'black · human')
        : (color === 'white' ? 'white · ' : 'black · ') + this.engineSub;
      $('avatar-' + suffix).className = 'pl-avatar' + (color === 'black' ? ' dark' : '');
    };
    setStrip('top', topColor, topIsHuman);
    setStrip('bot', botColor, !topIsHuman);

    const captured = (st && st.captured && (st.captured.white || st.captured.black))
      ? st.captured : derivedCaptures(board);
    $('tray-' + (topColor === 'white' ? 'top' : 'bot')).innerHTML = trayHTML(captured.white);
    $('tray-' + (topColor === 'black' ? 'top' : 'bot')).innerHTML = trayHTML(captured.black);

    const mat = (st && st.material && st.material.white != null) ? st.material : materialOf(board);
    const diff = (mat.white || 0) - (mat.black || 0);
    $('adv-' + (topColor === 'white' ? 'top' : 'bot')).textContent = diff > 0 ? '+' + diff : '';
    $('adv-' + (topColor === 'black' ? 'top' : 'bot')).textContent = diff < 0 ? '+' + (-diff) : '';

    const live = this.viewPly == null;
    const turn = st ? st.turn : 'white';
    const active = live && st && !st.result && !this.resigned;
    $('strip-top').classList.toggle('to-move', !!active && turn === topColor);
    $('strip-bottom').classList.toggle('to-move', !!active && turn === botColor);

    const plies = st ? (st.moves || []).length : 0;
    $('clock-top').textContent = $('clock-bot').textContent =
      Math.floor(plies / 2) + (plies % 2 ? '½' : '') + ' moves';
  },

  paintMoves() {
    const st = this.state;
    const list = $('movelist');
    const sans = (st && (st.history_san || st.historySan)) ||
      (st && st.moves ? st.moves.slice() : []);
    const total = sans.length;
    $('moves-empty').hidden = total > 0;
    list.hidden = total === 0;
    if (!total) { list.textContent = ''; return; }

    let html = '';
    for (let i = 0; i < total; i += 2) {
      html += '<li class="num">' + (i / 2 + 1) + '.</li>';
      html += '<li class="cell"><button class="mv" data-ply="' + (i + 1) + '">' + esc(sans[i]) + '</button></li>';
      html += i + 1 < total
        ? '<li class="cell"><button class="mv" data-ply="' + (i + 2) + '">' + esc(sans[i + 1]) + '</button></li>'
        : '<li class="cell"><span class="mv blank">—</span></li>';
    }
    list.innerHTML = html;
    const cur = this.curPly();
    const btn = list.querySelector('.mv[data-ply="' + cur + '"]');
    if (btn) {
      btn.classList.add('cur');
      const top = btn.offsetTop, h = list.clientHeight;
      if (top < list.scrollTop || top > list.scrollTop + h - 30) list.scrollTop = top - h / 2;
    }
  },

  paintBanner() {
    const st = this.state;
    const b = $('banner');
    if (!st) { b.hidden = true; return; }
    let title = null, sub = '', cls = '';

    if (this.resigned) {
      title = 'Resigned';
      sub = (this.resigned === 'white' ? 'Black' : 'White') + ' wins';
      cls = 'loss';
    } else if (st.result) {
      const reason = REASON_TEXT[st.reason] || (st.reason ? st.reason : '');
      if (st.result === 1 || st.result === 2) {
        const winner = st.result === 1 ? 'white' : 'black';
        title = reason || 'Game over';
        sub = (winner === 'white' ? 'White' : 'Black') + ' wins' +
          (winner === this.human ? ' — that is you' : '');
        cls = winner === this.human ? 'win' : 'loss';
      } else {
        title = 'Draw';
        sub = reason || 'The game is drawn';
      }
    }
    if (!title) { b.hidden = true; return; }
    $('banner-title').textContent = title;
    $('banner-sub').textContent = sub;
    const icon = $('banner-icon');
    icon.className = 'banner-icon ' + cls;
    icon.textContent = cls === 'win' ? '★' : cls === 'loss' ? '·' : '½';
    b.hidden = false;
  },

  clearEngine() {
    this.engine = null;
    $('st-depth').textContent = '—';
    $('st-nodes').textContent = '—';
    $('st-time').textContent = '—';
    $('st-nps').textContent = '—';
    $('cands').innerHTML =
      '<li class="cand empty"><span class="cand-san">—</span></li>'.repeat(3);
    $('evalbar-white').style.width = '50%';
    $('eval-pill').textContent = '0.00';
    $('eval-pill').className = 'pill';
  },

  paintEngine() {
    const e = this.engine;
    if (!e) return;
    $('st-depth').textContent = e.depth != null ? String(e.depth) : '—';
    $('st-nodes').textContent = e.nodes != null ? fmtCompact(e.nodes) : '—';
    $('st-time').textContent = fmtMs(e.ms);
    $('st-nps').textContent = (e.nodes && e.ms) ? fmtCompact(e.nodes / (e.ms / 1000) / 1000) : '—';

    const top = (e.top || []).slice(0, 3);
    if (!top.length) {
      $('cands').innerHTML = '<li class="cand empty"><span class="cand-san">' +
        esc(e.san || e.move || '—') + '</span></li>' +
        '<li class="cand empty"><span class="cand-san">—</span></li>'.repeat(2);
    } else {
      const max = Math.max.apply(null, top.map((t) => t.prob || 0)) || 1;
      let html = '';
      top.forEach((t, i) => {
        const p = t.prob != null ? t.prob : 0;
        html += '<li class="cand' + (i === 0 ? ' top' : '') + '">' +
          '<span class="bar" style="width:' + (100 * p / max).toFixed(1) + '%"></span>' +
          '<span class="cand-san">' + esc(t.san || t.move || '') + '</span>' +
          '<span class="cand-prob">' + fmtPct(p, 0) + '</span></li>';
      });
      for (let i = top.length; i < 3; i++) html += '<li class="cand empty"><span class="cand-san">—</span></li>';
      $('cands').innerHTML = html;
    }
    this.paintEval();
  },

  paintEval() {
    const e = this.engine;
    const bar = $('evalbar-white');
    const pill = $('eval-pill');
    if (!e) { bar.style.width = '50%'; pill.textContent = '0.00'; pill.className = 'pill'; return; }
    const sign = e.color === 'white' ? 1 : -1;      /* engine's view -> White's view */
    let cp = null;
    if (e.score != null && isFinite(e.score)) cp = sign * e.score / 100;
    let v = (e.value != null && isFinite(e.value)) ? sign * e.value : null;
    if (v == null && cp != null) v = Math.tanh(cp / 4);
    if (v == null) v = 0;
    bar.style.width = (50 + 50 * clamp(v, -1, 1) * 0.94).toFixed(1) + '%';
    const shown = cp != null ? fmtSigned(cp, 2) : fmtSigned(v, 2);
    pill.textContent = shown;
    pill.className = 'pill ' + (parseFloat(shown) > 0.15 ? 'pos' : parseFloat(shown) < -0.15 ? 'neg' : '');
  },

  setThinking(on) { $('thinking').hidden = !on; },

  setFlipped(v) {
    this.flipped = v;
    this.board.setFlipped(v);
    if (this.state) this.paintPlayers(this.lastBoard || parseFen(this.state.fen).board);
  }
};

function trayHTML(list) {
  if (!Array.isArray(list) || !list.length) return '';
  const codes = list.map((s) => {
    const ch = String(s).trim();
    if (!ch) return null;
    const t = ch[0];
    const white = t === t.toUpperCase() && /[A-Z]/.test(t);
    return (white ? 'w' : 'b') + t.toUpperCase();
  }).filter(Boolean);
  codes.sort((a, b) => (ORDER[a[1]] == null ? 9 : ORDER[a[1]]) - (ORDER[b[1]] == null ? 9 : ORDER[b[1]]));
  return codes.map((c) => pieceGlyph(c)).join('');
}

/* ========================================================================== */
/* WATCH view                                                                 */
/* ========================================================================== */

const Watch = {
  board: null,
  game: null,
  positions: null,
  ply: 0,
  playing: false,
  speed: 4,
  timer: null,
  flipped: false,
  loading: false,

  init() {
    this.board = new Board($('w-board'), { interactive: false });
    $('w-go').addEventListener('click', () => this.generate());
    $('w-play').addEventListener('click', () => this.toggle());
    $('w-first').addEventListener('click', () => { this.pause(); this.seek(0); });
    $('w-last').addEventListener('click', () => { this.pause(); this.seek(this.maxPly()); });
    $('w-back').addEventListener('click', () => { this.pause(); this.seek(this.ply - 1); });
    $('w-fwd').addEventListener('click', () => { this.pause(); this.seek(this.ply + 1); });
    $('w-flip').addEventListener('click', () => {
      this.flipped = !this.flipped; this.board.setFlipped(this.flipped); this.paintPlayers();
    });
    $('w-seek').addEventListener('input', (e) => { this.pause(); this.seek(+e.target.value); });
    $('w-speed').addEventListener('input', (e) => {
      this.speed = +e.target.value;
      $('w-speed-out').textContent = this.speed + '/s';
      if (this.playing) { this.pause(); this.play(); }
    });
    $('w-movelist').addEventListener('click', (e) => {
      const b = e.target.closest('.mv');
      if (b && b.dataset.ply) { this.pause(); this.seek(+b.dataset.ply); }
    });
    $('w-evalchart').addEventListener('click', (e) => {
      const hit = $('w-evalchart').querySelector('.plot-hit');
      if (!hit || !this.positions) return;
      const r = hit.getBoundingClientRect();
      if (!r.width) return;
      const f = clamp((e.clientX - r.left) / r.width, 0, 1);
      this.pause();
      this.seek(Math.round(f * this.maxPly()));
    });
    this.loadAgents();
  },

  async loadAgents() {
    try {
      const m = await api('/api/model');
      const agents = Array.isArray(m && m.agents) ? m.agents : [];
      const opts = agents.slice(0, 256).map((a) => {
        const idx = a.i != null ? a.i : (a.index != null ? a.index : 0);
        const elo = a.elo != null ? ' · ' + Math.round(a.elo) + ' Elo' : '';
        return '<option value="' + esc(idx) + '">agent ' + esc(idx) + esc(elo) + '</option>';
      }).join('');
      $('w-sel-a').innerHTML = '<option value="">best agent</option>' + opts;
      $('w-sel-b').innerHTML = '<option value="">random agent</option>' + opts;
      if (agents.length > 1) $('w-sel-b').value = String(agents[Math.min(1, agents.length - 1)].i);
    } catch (_) { /* the match-up selects simply stay on their defaults */ }
  },

  maxPly() { return this.positions ? this.positions.length - 1 : 0; },

  async generate() {
    if (this.loading) return;
    this.loading = true;
    this.pause();
    $('w-thinking').hidden = false;
    $('w-go').disabled = true;
    try {
      const a = $('w-sel-a').value, b = $('w-sel-b').value;
      const q = [];
      if (a !== '') q.push('a=' + encodeURIComponent(a));
      if (b !== '') q.push('b=' + encodeURIComponent(b));
      q.push('plies=400');
      const g = await api('/api/watch?' + q.join('&'));
      this.setGame(g);
    } catch (e) {
      reportError(e, 'Could not generate a game');
    } finally {
      this.loading = false;
      $('w-thinking').hidden = true;
      $('w-go').disabled = false;
    }
  },

  setGame(g) {
    this.game = g || {};
    const moves = Array.isArray(g.moves) ? g.moves : [];
    const fens = Array.isArray(g.fens) ? g.fens : [];
    if (fens.length >= moves.length + 1) {
      this.positions = fens.slice(0, moves.length + 1).map((f, i) => ({
        board: parseFen(f).board,
        last: i > 0 && moves[i - 1] ? { from: moves[i - 1].slice(0, 2), to: moves[i - 1].slice(2, 4) } : null
      }));
    } else if (fens.length === moves.length && moves.length) {
      /* fens[i] = position AFTER move i; synthesise the start position */
      this.positions = [{ board: parseFen(START_FEN).board, last: null }].concat(
        fens.map((f, i) => ({
          board: parseFen(f).board,
          last: { from: moves[i].slice(0, 2), to: moves[i].slice(2, 4) }
        })));
    } else {
      this.positions = replayBoards(START_FEN, moves);
    }
    this.ply = 0;
    $('w-seek').max = String(this.maxPly());
    $('w-seek').value = '0';
    this.paintMoves();
    this.paintResult();
    this.paintChart();
    this.render(false);
    if (this.maxPly() > 0) this.play();
  },

  toggle() { if (this.playing) this.pause(); else this.play(); },

  play() {
    if (!this.positions || this.maxPly() === 0) return;
    if (this.ply >= this.maxPly()) this.seek(0);
    this.playing = true;
    $('w-play').classList.add('playing');
    const step = () => {
      if (!this.playing) return;
      if (this.ply >= this.maxPly()) { this.pause(); return; }
      this.seek(this.ply + 1);
      this.timer = setTimeout(step, Math.max(50, 1000 / this.speed));
    };
    this.timer = setTimeout(step, Math.max(50, 1000 / this.speed));
  },

  pause() {
    this.playing = false;
    $('w-play').classList.remove('playing');
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
  },

  seek(p) {
    if (!this.positions) return;
    this.ply = clamp(p, 0, this.maxPly());
    $('w-seek').value = String(this.ply);
    this.render(true);
  },

  render(animate) {
    if (!this.positions) return;
    const pos = this.positions[this.ply];
    this.board.setPosition(pos.board, { last: pos.last, animate: animate !== false });
    this.board.setMarks({ last: pos.last });
    $('w-plylabel').textContent = this.ply + ' / ' + this.maxPly();
    this.paintPlayers();
    this.highlightMove();
    this.paintChart();
  },

  paintPlayers() {
    const g = this.game || {};
    const pos = this.positions ? this.positions[this.ply] : null;
    const board = pos ? pos.board : parseFen(START_FEN).board;
    const topColor = this.flipped ? 'white' : 'black';
    const name = (c) => esc(c === 'white' ? (g.white || 'White') : (g.black || 'Black'));

    $('w-name-top').innerHTML = name(topColor);
    $('w-name-bot').innerHTML = name(topColor === 'white' ? 'black' : 'white');
    $('w-sub-top').textContent = topColor === 'white' ? 'white' : 'black';
    $('w-sub-bot').textContent = topColor === 'white' ? 'black' : 'white';
    $('w-avatar-top').className = 'pl-avatar' + (topColor === 'black' ? ' dark' : '');
    $('w-avatar-bot').className = 'pl-avatar' + (topColor === 'white' ? ' dark' : '');

    const cap = derivedCaptures(board);
    $('w-tray-top').innerHTML = trayHTML(topColor === 'white' ? cap.white : cap.black);
    $('w-tray-bot').innerHTML = trayHTML(topColor === 'white' ? cap.black : cap.white);
    const mat = materialOf(board);
    const diff = mat.white - mat.black;
    $('w-adv-top').textContent = topColor === 'white' ? (diff > 0 ? '+' + diff : '') : (diff < 0 ? '+' + (-diff) : '');
    $('w-adv-bot').textContent = topColor === 'white' ? (diff < 0 ? '+' + (-diff) : '') : (diff > 0 ? '+' + diff : '');
  },

  paintMoves() {
    const g = this.game || {};
    const sans = Array.isArray(g.sans) && g.sans.length ? g.sans : (g.moves || []);
    const list = $('w-movelist');
    $('w-moves-empty').hidden = sans.length > 0;
    list.hidden = sans.length === 0;
    if (!sans.length) { list.textContent = ''; return; }
    let html = '';
    for (let i = 0; i < sans.length; i += 2) {
      html += '<li class="num">' + (i / 2 + 1) + '.</li>';
      html += '<li class="cell"><button class="mv" data-ply="' + (i + 1) + '">' + esc(sans[i]) + '</button></li>';
      html += i + 1 < sans.length
        ? '<li class="cell"><button class="mv" data-ply="' + (i + 2) + '">' + esc(sans[i + 1]) + '</button></li>'
        : '<li class="cell"><span class="mv blank">—</span></li>';
    }
    list.innerHTML = html;
  },

  highlightMove() {
    const list = $('w-movelist');
    const prev = list.querySelector('.mv.cur');
    if (prev) prev.classList.remove('cur');
    const b = list.querySelector('.mv[data-ply="' + this.ply + '"]');
    if (b) {
      b.classList.add('cur');
      const top = b.offsetTop, h = list.clientHeight;
      if (top < list.scrollTop || top > list.scrollTop + h - 30) list.scrollTop = top - h / 2;
    }
  },

  paintResult() {
    const g = this.game || {};
    const r = g.result;
    const reason = REASON_TEXT[g.reason] || g.reason || '';
    let txt = '—';
    if (r === 1) txt = '1–0';
    else if (r === 2) txt = '0–1';
    else if (r === 3) txt = '½–½';
    $('w-result').textContent = txt + (reason ? ' · ' + reason : '');
  },

  paintChart() {
    const g = this.game || {};
    const host = $('w-evalchart');
    let evals = Array.isArray(g.evals) ? g.evals.filter((v) => v == null || isFinite(v)) : [];
    if (!evals.length) {
      host.innerHTML = '<p class="empty-note">No evaluation trace for this game.</p>';
      $('w-eval-pill').textContent = '—';
      return;
    }
    const xs = evals.map((_, i) => i);
    host.innerHTML = lineChart({
      xs: xs,
      series: [{ name: 'eval', color: 'var(--accent)', values: evals }],
      width: 320, height: 120, area: true, zeroLine: true, hit: true,
      yMin: Math.min(-1, Math.min.apply(null, evals)),
      yMax: Math.max(1, Math.max.apply(null, evals)),
      fmtY: (v) => fmtNum(v, 1),
      fmtX: (v) => String(Math.round(v)),
      markerIndex: clamp(this.ply, 0, evals.length - 1)
    });
    const v = evals[clamp(this.ply, 0, evals.length - 1)];
    $('w-eval-pill').textContent = fmtSigned(v, 2);
    $('w-eval-pill').className = 'pill ' + (v > 0.1 ? 'pos' : v < -0.1 ? 'neg' : '');
  }
};

/* ========================================================================== */
/* REPORT view                                                                */
/* ========================================================================== */

/* The exact shape of /api/report is produced by py/report.py; the normaliser
   below accepts every reasonable spelling of the telemetry documented in
   docs/ALGORITHM.md, including the raw per-generation JSONL rows. */

function walk(root, visit, depth) {
  depth = depth || 0;
  if (!root || typeof root !== 'object' || depth > 6) return;
  if (visit(root, depth) === true) return;
  const vals = Array.isArray(root) ? root : Object.values(root);
  for (const v of vals) if (v && typeof v === 'object') walk(v, visit, depth + 1);
}

function findArray(root, test) {
  let hit = null;
  walk(root, (node) => {
    if (hit) return true;
    if (Array.isArray(node) && node.length && test(node)) { hit = node; return true; }
    if (!Array.isArray(node)) {
      for (const v of Object.values(node)) {
        if (Array.isArray(v) && v.length && test(v)) { hit = v; return true; }
      }
    }
    return false;
  });
  return hit;
}

function findObject(root, test) {
  let hit = null;
  walk(root, (node) => {
    if (hit) return true;
    if (!Array.isArray(node) && test(node)) { hit = node; return true; }
    return false;
  });
  return hit;
}

function pick(obj, names, dflt) {
  if (!obj) return dflt;
  for (const n of names) {
    if (obj[n] != null) return obj[n];
    const lower = n.toLowerCase();
    for (const k of Object.keys(obj)) {
      if (k.toLowerCase() === lower && obj[k] != null) return obj[k];
    }
  }
  return dflt;
}

function seriesFrom(rows, names) {
  const out = rows.map((r) => {
    const v = pick(r, names, null);
    return typeof v === 'number' && isFinite(v) ? v : null;
  });
  return out.some((v) => v != null) ? out : null;
}

function toNumberMap(obj) {
  const out = [];
  if (!obj || typeof obj !== 'object') return out;
  for (const [k, v] of Object.entries(obj)) {
    if (typeof v === 'number' && isFinite(v)) out.push({ label: k, value: v });
  }
  return out;
}

function normaliseReport(raw) {
  const R = { raw: raw };

  /* --- per-generation rows -------------------------------------------- */
  const rows = findArray(raw, (a) =>
    typeof a[0] === 'object' && a[0] !== null && !Array.isArray(a[0]) &&
    (a[0].gen != null || a[0].generation != null) &&
    Object.keys(a[0]).length > 2) || [];
  R.rows = rows;
  R.gens = rows.length
    ? rows.map((r, i) => {
      const g = pick(r, ['gen', 'generation'], i + 1);
      return typeof g === 'number' ? g : i + 1;
    })
    : [];

  const S = (names) => (rows.length ? seriesFrom(rows, names) : null);
  const direct = (names) => {
    for (const n of names) {
      const v = pick(raw, [n], null);
      if (Array.isArray(v) && v.length && typeof v[0] === 'number') return v;
    }
    return null;
  };

  R.eloBest = S(['elo_best', 'eloBest', 'best_elo']) || direct(['elo_best']);
  R.eloMean = S(['elo_mean', 'eloMean', 'mean_elo']) || direct(['elo_mean']);
  R.eloP10 = S(['elo_p10', 'eloP10']) || direct(['elo_p10']);
  const eloObj = findObject(raw, (o) => o && (Array.isArray(o.best) || Array.isArray(o.mean)) &&
    (o.best || o.mean).length && typeof (o.best || o.mean)[0] === 'number');
  if (!R.eloBest && eloObj) R.eloBest = Array.isArray(eloObj.best) ? eloObj.best : null;
  if (!R.eloMean && eloObj) R.eloMean = Array.isArray(eloObj.mean) ? eloObj.mean : null;
  if (!R.eloP10 && eloObj) R.eloP10 = Array.isArray(eloObj.p10) ? eloObj.p10 : null;

  R.whiteWin = S(['white_win', 'whiteWin', 'white_wins', 'white_win_rate']);
  R.blackWin = S(['black_win', 'blackWin', 'black_wins', 'black_win_rate']);
  R.draw = S(['draw', 'draws', 'draw_rate']);
  R.avgLen = S(['avg_len', 'avgLen', 'average_length', 'avg_plies']);
  R.gps = S(['gps', 'games_per_sec', 'games_sec']);
  R.captures = S(['captures_per_game', 'captures']);
  R.checks = S(['checks_per_game', 'checks']);
  R.castle = S(['castle_rate', 'castles']);
  R.promo = S(['promo_rate', 'promotions']);
  R.ep = S(['ep_rate']);
  R.entropy = S(['piece_dest_entropy', 'entropy']);
  R.material = S(['avg_final_material']);
  R.gradNorm = S(['grad_norm', 'gradNorm']);

  const lossOf = (k) => rows.length
    ? rows.map((r) => {
      const l = pick(r, ['loss'], null);
      const v = l && typeof l === 'object' ? pick(l, [k], null) : pick(r, ['loss_' + k], null);
      return typeof v === 'number' && isFinite(v) ? v : null;
    })
    : null;
  R.lossPolicy = lossOf('policy');
  R.lossValue = lossOf('value');
  R.lossEntropy = lossOf('entropy');
  if (R.lossPolicy && !R.lossPolicy.some((v) => v != null)) R.lossPolicy = null;
  if (R.lossValue && !R.lossValue.some((v) => v != null)) R.lossValue = null;
  if (R.lossEntropy && !R.lossEntropy.some((v) => v != null)) R.lossEntropy = null;

  if (!R.gens.length) {
    const anyLen = [R.eloBest, R.eloMean, R.whiteWin].find((a) => Array.isArray(a) && a.length);
    if (anyLen) R.gens = anyLen.map((_, i) => i + 1);
  }

  /* --- terminations ---------------------------------------------------- */
  const TERM_KEYS = ['checkmate', 'stalemate', 'fifty', 'repetition', 'insufficient', 'maxplies'];
  let term = findObject(raw, (o) => {
    const keys = Object.keys(o).map((k) => k.toLowerCase());
    let hits = 0;
    for (const t of TERM_KEYS) if (keys.includes(t)) hits++;
    return hits >= 3 && Object.values(o).every((v) => typeof v === 'number');
  });
  if (!term && rows.length) {
    term = {};
    for (const r of rows) {
      const t = pick(r, ['term', 'terminations', 'termination'], null);
      if (t && typeof t === 'object') {
        for (const [k, v] of Object.entries(t)) {
          if (typeof v === 'number') term[k] = (term[k] || 0) + v;
        }
      }
    }
    if (!Object.keys(term).length) term = null;
  }
  R.terminations = term ? toNumberMap(term).filter((d) => d.value >= 0) : [];

  /* --- openings -------------------------------------------------------- */
  let openings = [];
  const openArr = findArray(raw, (a) => {
    const f = a[0];
    if (Array.isArray(f)) return f.length >= 2 && typeof f[0] === 'string';
    return f && typeof f === 'object' &&
      (f.san != null || f.move != null || f.opening != null) &&
      (f.count != null || f.pct != null || f.n != null || f.freq != null || f.share != null);
  });
  if (openArr) {
    openings = openArr.map((e) => Array.isArray(e)
      ? { label: String(e[0]), value: +e[1] || 0 }
      : {
        label: String(pick(e, ['san', 'move', 'opening', 'name'], '?')),
        value: +pick(e, ['count', 'n', 'pct', 'freq', 'share', 'value'], 0) || 0
      });
  } else {
    const openObj = findObject(raw, (o) => {
      const ks = Object.keys(o);
      return ks.length >= 3 && ks.length <= 40 &&
        ks.every((k) => /^[a-h][1-8][a-h][1-8][qrbn]?$|^[KQRBNa-h][a-h1-8x=+#-]{1,6}$/.test(k)) &&
        Object.values(o).every((v) => typeof v === 'number');
    });
    if (openObj) openings = toNumberMap(openObj);
  }
  if (!openings.length && rows.length) {
    const agg = {};
    for (const r of rows) {
      const t = pick(r, ['opening_top', 'openings', 'opening'], null);
      if (Array.isArray(t)) {
        for (const e of t) {
          if (Array.isArray(e)) agg[e[0]] = (agg[e[0]] || 0) + (+e[1] || 0);
          else if (e && typeof e === 'object') {
            const k = String(pick(e, ['san', 'move', 'name'], '?'));
            agg[k] = (agg[k] || 0) + (+pick(e, ['count', 'n', 'pct', 'freq', 'value'], 0) || 0);
          }
        }
      } else if (t && typeof t === 'object') {
        for (const [k, v] of Object.entries(t)) if (typeof v === 'number') agg[k] = (agg[k] || 0) + v;
      }
    }
    openings = toNumberMap(agg);
  }
  openings.sort((a, b) => b.value - a.value);
  R.openings = openings.slice(0, 10);

  /* --- piece-square heatmaps ------------------------------------------- */
  const PIECES = ['P', 'N', 'B', 'R', 'Q', 'K'];
  const NAMES = { p: 'P', pawn: 'P', n: 'N', knight: 'N', b: 'B', bishop: 'B',
    r: 'R', rook: 'R', q: 'Q', queen: 'Q', k: 'K', king: 'K' };
  let heat = null;
  const is64 = (a) => Array.isArray(a) && a.length === 64 && a.every((v) => typeof v === 'number');
  const heatArr = findArray(raw, (a) => a.length >= 1 && a.length <= 6 && is64(a[0]));
  if (heatArr) {
    heat = heatArr.map((cells, i) => ({ piece: PIECES[i] || '?', cells: cells }));
  } else {
    const heatObj = findObject(raw, (o) => {
      const vals = Object.values(o);
      return vals.length >= 1 && vals.length <= 12 && vals.some(is64) &&
        Object.keys(o).every((k) => NAMES[k.toLowerCase()] != null);
    });
    if (heatObj) {
      heat = Object.entries(heatObj)
        .filter(([, v]) => is64(v))
        .map(([k, v]) => ({ piece: NAMES[k.toLowerCase()] || '?', cells: v }));
      heat.sort((a, b) => PIECES.indexOf(a.piece) - PIECES.indexOf(b.piece));
    }
  }
  R.heat = heat || [];

  /* --- headline numbers ------------------------------------------------ */
  const last = rows.length ? rows[rows.length - 1] : null;
  R.meta = {
    run: pick(raw, ['run', 'run_dir', 'name'], null),
    generation: pick(raw, ['generation', 'generations', 'gens'], null) ||
      (R.gens.length ? R.gens[R.gens.length - 1] : null),
    agents: pick(raw, ['n_agents', 'agents_count', 'population'], null),
    games: pick(raw, ['games', 'total_games'], null) ||
      (rows.length ? rows.reduce((s, r) => s + (+pick(r, ['games'], 0) || 0), 0) : null),
    plies: rows.length ? rows.reduce((s, r) => s + (+pick(r, ['plies'], 0) || 0), 0) : null,
    seconds: rows.length ? rows.reduce((s, r) => s + (+pick(r, ['sec', 'seconds'], 0) || 0), 0) : null,
    eloBest: R.eloBest ? lastNum(R.eloBest) : pick(raw, ['elo_best'], null),
    eloMean: R.eloMean ? lastNum(R.eloMean) : null,
    gps: R.gps ? lastNum(R.gps) : null,
    avgLen: R.avgLen ? lastNum(R.avgLen) : null,
    best: last ? pick(last, ['best_agent'], null) : null
  };
  if (typeof R.meta.agents !== 'number') {
    const n = pick(raw, ['n_agents'], null);
    R.meta.agents = typeof n === 'number' ? n : null;
  }
  return R;
}

function lastNum(arr) {
  for (let i = arr.length - 1; i >= 0; i--) if (arr[i] != null && isFinite(arr[i])) return arr[i];
  return null;
}

const Report = {
  loaded: false,
  loading: false,

  async load(force) {
    if (this.loading || (this.loaded && !force)) return;
    this.loading = true;
    const wrap = $('report-wrap');
    wrap.innerHTML = '<div class="report-loading"><span class="spinner"></span>' +
      '<span>loading the strategy analysis…</span></div>';
    try {
      const raw = await api('/api/report');
      const R = normaliseReport(raw);
      this.render(R);
      this.loaded = true;
    } catch (e) {
      wrap.innerHTML =
        '<div class="report-error"><strong>The strategy report is not available.</strong>' +
        '<span>' + esc(e && e.message ? e.message : String(e)) + '</span>' +
        '<span style="max-width:44ch">It is generated by <code>py/report.py</code> from ' +
        '<code>runs/&lt;name&gt;/telemetry.jsonl</code>; run a training job first.</span>' +
        '<button class="btn" id="report-retry">Try again</button></div>';
      const b = $('report-retry');
      if (b) b.addEventListener('click', () => this.load(true));
    } finally {
      this.loading = false;
    }
  },

  render(R) {
    const wrap = $('report-wrap');
    const m = R.meta;
    let html = '';

    html += '<div class="report-head"><div>' +
      '<h1>Strategy analysis</h1>' +
      '<p>' + (m.run ? esc(m.run) + ' · ' : '') +
      (m.generation != null ? fmtInt(m.generation) + ' generations' : 'population self-play') +
      (m.agents != null ? ' · ' + fmtInt(m.agents) + ' agents' : '') + '</p></div>' +
      '<button class="btn btn-sm" id="report-reload">Refresh</button></div>';

    /* ---- KPI row ---- */
    const kpis = [];
    if (m.eloBest != null) kpis.push(k('Best Elo', fmtNum(m.eloBest, 0), m.eloMean != null ? 'mean ' + fmtNum(m.eloMean, 0) : ''));
    if (m.games != null && m.games > 0) kpis.push(k('Games played', fmtCompact(m.games), m.plies ? fmtCompact(m.plies) + ' plies' : ''));
    if (m.gps != null) kpis.push(k('Throughput', fmtInt(m.gps), 'games / second', 'games/s'));
    if (m.avgLen != null) kpis.push(k('Game length', fmtNum(m.avgLen, 1), 'plies on average'));
    const dr = R.draw ? lastNum(R.draw) : null;
    if (dr != null) kpis.push(k('Draw rate', fmtPct(dr > 1 ? dr / 100 : dr, 1), 'latest generation'));
    if (m.seconds) kpis.push(k('Compute', fmtNum(m.seconds / 3600, 1), 'core-hours of self-play', 'h'));
    if (kpis.length) html += '<div class="kpis">' + kpis.join('') + '</div>';

    html += '<div class="report-grid">';

    /* ---- Elo curve ---- */
    if (R.gens.length && (R.eloBest || R.eloMean || R.eloP10)) {
      const series = [];
      if (R.eloBest) series.push({ name: 'best', color: 'var(--c1)', values: R.eloBest });
      if (R.eloMean) series.push({ name: 'population mean', color: 'var(--c2)', values: R.eloMean });
      if (R.eloP10) series.push({ name: '10th percentile', color: 'var(--c3)', values: R.eloP10 });
      html += card('Elo over training',
        'Hall-of-fame agents are frozen, so the scale is absolute rather than drifting with the mean.',
        '<div class="chart">' + lineChart({
          xs: R.gens, series: series, width: 520, height: 190,
          fmtY: (v) => fmtNum(v, 0), fmtX: (v) => 'g' + Math.round(v)
        }) + '</div>' + legend(series.map((s) => ({
          name: s.name, color: s.color, value: fmtNum(lastNum(s.values), 0)
        }))), 'span-2');
    }

    /* ---- result mix ---- */
    if (R.gens.length && (R.whiteWin || R.blackWin || R.draw)) {
      const norm = (a) => a ? a.map((v) => (v == null ? 0 : v > 1.0001 ? v / 100 : v)) : null;
      const w = norm(R.whiteWin) || R.gens.map(() => 0);
      const d = norm(R.draw) || R.gens.map(() => 0);
      const b = norm(R.blackWin) || R.gens.map(() => 0);
      const layers = [
        { name: 'White wins', color: 'var(--c1)', values: w },
        { name: 'Draws', color: 'var(--c3)', values: d },
        { name: 'Black wins', color: 'var(--c4)', values: b }
      ];
      html += card('Result mix by generation',
        'The share of games ending in a White win, a draw, or a Black win.',
        '<div class="chart">' + stackedArea({ xs: R.gens, layers: layers, width: 520, height: 190 }) + '</div>' +
        legend(layers.map((l) => ({ name: l.name, color: l.color, value: fmtPct(lastNum(l.values) || 0, 0) }))),
        'span-2');
    }

    /* ---- openings ---- */
    if (R.openings.length) {
      const total = R.openings.reduce((s, o) => s + o.value, 0) || 1;
      const max = R.openings[0].value || 1;
      const bars = R.openings.map((o) =>
        '<div class="bar-row"><span class="b-name">' + esc(o.label) + '</span>' +
        '<span class="b-track"><span class="b-fill" style="width:' +
        (100 * o.value / max).toFixed(1) + '%"></span></span>' +
        '<span class="b-val">' + (o.value <= 1 ? fmtPct(o.value, 1) : fmtPct(o.value / total, 1)) +
        '</span></div>').join('');
      html += card('Opening preference',
        'How often the population opens with each first move.',
        '<div class="bars">' + bars + '</div>');
    }

    /* ---- terminations ---- */
    if (R.terminations.length) {
      const order = ['checkmate', 'stalemate', 'repetition', 'fifty', 'insufficient', 'maxplies'];
      const items = R.terminations.slice().sort((a, b) => {
        const ai = order.indexOf(a.label.toLowerCase()), bi = order.indexOf(b.label.toLowerCase());
        return (ai < 0 ? 9 : ai) - (bi < 0 ? 9 : bi);
      });
      const total = items.reduce((s, i) => s + i.value, 0) || 1;
      const segs = items.map((it, i) =>
        '<span class="stack-seg" title="' + esc(it.label) + '" style="width:' +
        (100 * it.value / total).toFixed(2) + '%;background:' +
        CHART_COLORS[i % CHART_COLORS.length] + '"></span>').join('');
      html += card('How games end',
        'Termination reasons across the whole run.',
        '<div class="stack-bar">' + segs + '</div>' +
        legend(items.map((it, i) => ({
          name: termLabel(it.label), color: CHART_COLORS[i % CHART_COLORS.length],
          value: fmtPct(it.value / total, 1)
        }))));
    }

    /* ---- heatmaps ---- */
    if (R.heat.length) {
      html += card('Learned piece-square preference',
        'Where each piece type ends up moving, aggregated over self-play. Brighter is more visited.',
        '<div class="heatgrid">' + R.heat.map(heatBoard).join('') + '</div>' +
        '<div class="scale-strip"><span>rare</span>' +
        '<span class="scale-grad" style="background:linear-gradient(90deg,var(--bg-sunk),var(--accent))"></span>' +
        '<span>frequent</span></div>', 'span-2');
    }

    /* ---- style / dynamics ---- */
    const dyn = [];
    if (R.avgLen) dyn.push({ name: 'game length (plies)', color: 'var(--c1)', values: R.avgLen });
    if (R.captures) dyn.push({ name: 'captures per game', color: 'var(--c2)', values: R.captures });
    if (R.checks) dyn.push({ name: 'checks per game', color: 'var(--c3)', values: R.checks });
    if (dyn.length && R.gens.length) {
      html += card('Game dynamics',
        'Longer games with more captures and checks mean the population is fighting rather than shuffling.',
        '<div class="chart">' + lineChart({
          xs: R.gens, series: dyn, width: 360, height: 170, fmtX: (v) => 'g' + Math.round(v)
        }) + '</div>' + legend(dyn.map((s) => ({
          name: s.name, color: s.color, value: fmtNum(lastNum(s.values), 1)
        }))));
    }

    const rates = [];
    if (R.castle) rates.push({ name: 'castling rate', color: 'var(--c5)', values: R.castle });
    if (R.promo) rates.push({ name: 'promotion rate', color: 'var(--c6)', values: R.promo });
    if (R.ep) rates.push({ name: 'en-passant rate', color: 'var(--c4)', values: R.ep });
    if (R.entropy) rates.push({ name: 'destination entropy', color: 'var(--c2)', values: R.entropy });
    if (rates.length && R.gens.length) {
      html += card('Technique acquisition',
        'Rates of the moves that only appear once the population understands them.',
        '<div class="chart">' + lineChart({
          xs: R.gens, series: rates, width: 360, height: 170, fmtX: (v) => 'g' + Math.round(v),
          fmtY: (v) => fmtNum(v, 2)
        }) + '</div>' + legend(rates.map((s) => ({
          name: s.name, color: s.color, value: fmtNum(lastNum(s.values), 3)
        }))));
    }

    const losses = [];
    if (R.lossPolicy) losses.push({ name: 'policy', color: 'var(--c1)', values: R.lossPolicy });
    if (R.lossValue) losses.push({ name: 'value', color: 'var(--c4)', values: R.lossValue });
    if (R.lossEntropy) losses.push({ name: 'entropy', color: 'var(--c3)', values: R.lossEntropy });
    if (losses.length && R.gens.length) {
      html += card('Optimisation',
        'A2C loss components per generation.',
        '<div class="chart">' + lineChart({
          xs: R.gens, series: losses, width: 360, height: 170, fmtX: (v) => 'g' + Math.round(v),
          fmtY: (v) => fmtNum(v, 2)
        }) + '</div>' + legend(losses.map((s) => ({
          name: s.name, color: s.color, value: fmtNum(lastNum(s.values), 3)
        }))));
    }

    if (m.best && typeof m.best === 'object') {
      const b = m.best;
      const rowsHtml = Object.entries(b).map(([kk, v]) =>
        '<tr><td>' + esc(kk) + '</td><td class="num">' +
        esc(typeof v === 'number' ? fmtNum(v, 3) : String(v)) + '</td></tr>').join('');
      html += card('Champion hyper-parameters',
        'The best agent of the final generation, as selected by Elo.',
        '<table class="agent-table"><tbody>' + rowsHtml + '</tbody></table>');
    }

    html += '</div>';

    if (!R.gens.length && !R.openings.length && !R.terminations.length && !R.heat.length) {
      html += '<div class="report-error"><strong>The report contained no recognisable telemetry.</strong>' +
        '<span>Expected per-generation rows with <code>gen</code>, <code>elo_best</code>, ' +
        '<code>opening_top</code>, <code>term</code> and piece-square counts.</span></div>';
    }

    wrap.innerHTML = html;
    const rl = $('report-reload');
    if (rl) rl.addEventListener('click', () => this.load(true));
  }
};

function k(label, value, sub, unit) {
  return '<div class="kpi"><div class="k-label">' + esc(label) + '</div>' +
    '<div class="k-value">' + esc(value) + (unit ? '<span class="unit">' + esc(unit) + '</span>' : '') + '</div>' +
    (sub ? '<div class="k-sub">' + esc(sub) + '</div>' : '') + '</div>';
}

function card(title, sub, body, cls) {
  return '<section class="card ' + (cls || '') + '">' +
    '<div class="card-head"><h3>' + esc(title) + '</h3>' +
    (sub ? '<p>' + esc(sub) + '</p>' : '') + '</div>' +
    '<div class="card-body">' + body + '</div></section>';
}

function termLabel(s) {
  const map = {
    checkmate: 'checkmate', stalemate: 'stalemate', fifty: 'fifty-move',
    repetition: 'repetition', insufficient: 'insufficient material', maxplies: 'move limit'
  };
  return map[String(s).toLowerCase()] || String(s);
}

function heatBoard(h) {
  const cells = h.cells;
  let max = 0;
  for (const v of cells) if (v > max) max = v;
  if (!max) max = 1;
  let grid = '';
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const idx = (7 - row) * 8 + col;                 /* rank 8 first */
      const t = Math.pow(clamp(cells[idx] / max, 0, 1), 0.6);
      grid += '<div class="heat-cell" title="' + SQN[idx] + ': ' + fmtCompact(cells[idx]) + '">' +
        '<div style="position:absolute;inset:0;background:var(--accent);opacity:' + t.toFixed(3) + '"></div></div>';
    }
  }
  return '<div class="heat"><div class="heat-title">' + pieceGlyph('w' + h.piece) +
    '<span>' + esc({ P: 'Pawns', N: 'Knights', B: 'Bishops', R: 'Rooks', Q: 'Queens', K: 'Kings' }[h.piece] || h.piece) +
    '</span></div><div class="heat-board" style="background:var(--bg-sunk)">' + grid + '</div></div>';
}

/* ========================================================================== */
/* shell                                                                      */
/* ========================================================================== */

const Views = { play: 'view-play', watch: 'view-watch', report: 'view-report' };
let currentView = 'play';

function showView(name) {
  if (!Views[name]) return;
  currentView = name;
  for (const [n, id] of Object.entries(Views)) {
    const sec = $(id);
    const on = n === name;
    sec.classList.toggle('is-active', on);
    sec.hidden = !on;
    const tab = $('tab-' + n);
    tab.classList.toggle('is-active', on);
    tab.setAttribute('aria-selected', on ? 'true' : 'false');
  }
  if (name === 'report') Report.load(false);
  if (name !== 'watch') Watch.pause();
  try { history.replaceState(null, '', '#' + name); } catch (_) { /* ignore */ }
}

function applyTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode);
  store('theme', mode);
}

function initShell() {
  for (const n of Object.keys(Views)) {
    $('tab-' + n).addEventListener('click', () => showView(n));
  }
  const saved = store('theme');
  applyTheme(saved === 'light' || saved === 'dark' || saved === 'auto' ? saved : 'auto');
  $('theme-btn').addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const next = cur === 'auto' ? (dark ? 'light' : 'dark') : (cur === 'dark' ? 'light' : 'dark');
    applyTheme(next);
  });

  document.addEventListener('keydown', (e) => {
    if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (e.key === 'Escape') {
      if (Play.board && Play.board.promo) { Play.board.closePromo(); return; }
      if (Play.board && Play.board.selected) { Play.board.select(null); return; }
    }
    if (currentView === 'play') {
      if (e.key === 'ArrowLeft') { e.preventDefault(); Play.goto(Play.curPly() - 1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); Play.goto(Play.curPly() + 1); }
      if (e.key === 'f') Play.setFlipped(!Play.flipped);
    } else if (currentView === 'watch') {
      if (e.key === 'ArrowLeft') { e.preventDefault(); Watch.pause(); Watch.seek(Watch.ply - 1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); Watch.pause(); Watch.seek(Watch.ply + 1); }
      if (e.key === ' ') { e.preventDefault(); Watch.toggle(); }
    }
  });

  const hash = (location.hash || '').replace('#', '');
  if (Views[hash]) showView(hash); else showView('play');
}

async function loadModelChip() {
  const chip = $('model-chip'), label = $('model-label');
  try {
    const m = await api('/api/model');
    const gen = m && m.generation != null ? 'gen ' + fmtInt(m.generation) : '';
    const n = m && m.n_agents != null ? fmtInt(m.n_agents) + ' agents' : '';
    label.textContent = [gen, n].filter(Boolean).join(' · ') || 'model loaded';
    chip.classList.add('ok');
    chip.title = (m && m.path) ? String(m.path) : 'model';
    const best = Array.isArray(m && m.agents)
      ? m.agents.reduce((a, b) => ((b && b.elo != null && (!a || b.elo > a.elo)) ? b : a), null)
      : null;
    if (best && best.elo != null) {
      $('brand-sub').textContent = 'champion · ' + Math.round(best.elo) + ' Elo';
    }
  } catch (e) {
    label.textContent = 'no model';
    chip.classList.add('err');
    chip.title = e && e.message ? e.message : 'model unavailable';
  }
}

function boot() {
  initShell();
  Play.init();
  Watch.init();
  loadModelChip();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
