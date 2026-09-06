/* pieces.js -- hand-drawn Staunton-silhouette chess pieces as inline SVG.
 *
 * Every piece lives in a 45x45 box (the conventional chess-SVG canvas) and is
 * built from three kinds of element:
 *
 *   .pc-body    filled with var(--pc-fill), outlined with var(--pc-line)
 *   .pc-detail  filled with var(--pc-line)   (eyes, cross, crown balls)
 *   .pc-stroke  stroked with var(--pc-ink)   (mitre slit, mane, collar lines)
 *
 * The two colour variables are set by `.piece.w` / `.piece.b` in style.css, so
 * the same geometry serves both sides and both themes.  No external assets.
 */

'use strict';

/* Shared plinth: a collar and a foot.  Drawn last so it paints over the seam
 * where the body meets the base. */
const PC_BASE =
  '<rect class="pc-body" x="10.4" y="33.4" width="24.2" height="3.7" rx="1.3"/>' +
  '<rect class="pc-body" x="7.3" y="37.1" width="30.4" height="4.5" rx="1.8"/>';

const PC_SHAPES = {
  /* ------------------------------------------------------------------ pawn */
  P:
    '<path class="pc-body" d="M 22.5,8.2 C 19.8,8.2 17.6,10.4 17.6,13.1 ' +
    'C 17.6,14.3 18.1,15.5 18.9,16.3 C 17.1,17.4 15.9,19.2 15.9,21.3 ' +
    'C 15.9,23 16.7,24.6 18,25.6 C 15.1,27.2 12.6,30.2 12.6,34 ' +
    'L 32.4,34 C 32.4,30.2 29.9,27.2 27,25.6 C 28.3,24.6 29.1,23 29.1,21.3 ' +
    'C 29.1,19.2 27.9,17.4 26.1,16.3 C 26.9,15.5 27.4,14.3 27.4,13.1 ' +
    'C 27.4,10.4 25.2,8.2 22.5,8.2 Z"/>' + PC_BASE,

  /* ------------------------------------------------------------------ rook */
  R:
    '<path class="pc-body" d="M 11.5,34 L 11.5,31.4 L 14.1,29.4 L 14.1,17.6 ' +
    'L 11.5,15 L 11.5,9 L 16,9 L 16,11.5 L 19.8,11.5 L 19.8,9 L 25.2,9 ' +
    'L 25.2,11.5 L 29,11.5 L 29,9 L 33.5,9 L 33.5,15 L 30.9,17.6 L 30.9,29.4 ' +
    'L 33.5,31.4 L 33.5,34 Z"/>' +
    '<path class="pc-stroke" d="M 14.1,17.6 L 30.9,17.6 M 14.1,29.4 L 30.9,29.4"/>' +
    PC_BASE,

  /* ---------------------------------------------------------------- knight */
  N:
    '<path class="pc-body" d="M 20.5,9.5 L 22.4,5.6 L 24.7,9.9 L 27.6,7.2 ' +
    'L 28.9,11.5 C 31.7,15.1 33.6,20.3 33.6,26.2 C 33.6,29.4 33.2,31.9 32.6,34 ' +
    'L 13.4,34 C 13.4,30.7 14.5,28.1 16.4,25.9 C 18,24 19.5,22.7 19.5,21.4 ' +
    'C 19.5,20.5 18.9,20 18.1,20 C 16.7,20 15.7,21.2 14.5,22.6 ' +
    'C 13.4,23.9 12.3,24.8 11.1,24.8 C 9.5,24.8 8.6,23.6 8.6,21.8 ' +
    'C 8.6,19.2 9.7,16.6 11.7,14.2 C 13.9,11.5 17.1,10 20.5,9.5 Z"/>' +
    '<ellipse class="pc-detail" cx="15.7" cy="17.2" rx="1.35" ry="2.05" ' +
    'transform="rotate(32 15.7 17.2)"/>' +
    '<circle class="pc-detail" cx="10.9" cy="22.4" r="0.85"/>' +
    '<path class="pc-stroke" d="M 26.9,12.8 C 29.4,16.4 30.8,20.9 30.8,25.7 ' +
    'M 23.9,11.6 C 26.9,14.5 28.6,18.6 28.9,23"/>' +
    PC_BASE,

  /* ---------------------------------------------------------------- bishop */
  B:
    '<path class="pc-body" d="M 22.5,11 C 26.4,13.8 29.6,18 29.6,22 ' +
    'C 29.6,25.2 27.8,27.7 25.4,29.1 L 19.6,29.1 C 17.2,27.7 15.4,25.2 15.4,22 ' +
    'C 15.4,18 18.6,13.8 22.5,11 Z"/>' +
    '<circle class="pc-body" cx="22.5" cy="8.4" r="2.6"/>' +
    '<path class="pc-stroke" d="M 20,17.6 L 26.6,24.2"/>' +
    '<path class="pc-body" d="M 19.6,29.1 L 25.4,29.1 L 25.4,31.5 ' +
    'C 28.4,32.1 30.8,32.9 32.4,34 L 12.6,34 C 14.2,32.9 16.6,32.1 19.6,31.5 Z"/>' +
    PC_BASE,

  /* ----------------------------------------------------------------- queen */
  Q:
    '<path class="pc-body" d="M 8,13.6 L 14.2,25.2 L 14.5,11.8 L 19.6,24.8 ' +
    'L 22.5,10.8 L 25.4,24.8 L 30.5,11.8 L 30.8,25.2 L 37,13.6 L 35.6,26.9 ' +
    'C 35.6,29.7 34.9,31.9 33.8,34 L 11.2,34 C 10.1,31.9 9.4,29.7 9.4,26.9 Z"/>' +
    '<circle class="pc-body" cx="8" cy="11.3" r="2.35"/>' +
    '<circle class="pc-body" cx="14.5" cy="9.5" r="2.35"/>' +
    '<circle class="pc-body" cx="22.5" cy="8.3" r="2.6"/>' +
    '<circle class="pc-body" cx="30.5" cy="9.5" r="2.35"/>' +
    '<circle class="pc-body" cx="37" cy="11.3" r="2.35"/>' +
    '<path class="pc-stroke" d="M 9.7,26.9 C 15.5,25.6 29.5,25.6 35.3,26.9 ' +
    'M 10.6,30.6 C 16,29.5 29,29.5 34.4,30.6"/>' +
    PC_BASE,

  /* ------------------------------------------------------------------ king */
  K:
    '<path class="pc-detail" d="M 21.1,4.4 h 2.8 v 3.2 h 3.2 v 2.8 h -3.2 v 3.5 ' +
    'h -2.8 v -3.5 h -3.2 v -2.8 h 3.2 z"/>' +
    '<path class="pc-body" d="M 22.5,13.6 C 21.8,15.9 20.6,17.7 19,19.3 ' +
    'C 17,16.7 13.9,15.9 11.4,17.2 C 8.6,18.7 7.9,22.3 9.6,25.3 ' +
    'C 11.1,28 12.9,30.3 13.5,32 L 13.5,34 L 31.5,34 L 31.5,32 ' +
    'C 32.1,30.3 33.9,28 35.4,25.3 C 37.1,22.3 36.4,18.7 33.6,17.2 ' +
    'C 31.1,15.9 28,16.7 26,19.3 C 24.4,17.7 23.2,15.9 22.5,13.6 Z"/>' +
    '<path class="pc-stroke" d="M 13.5,31.8 C 18,30 27,30 31.5,31.8 ' +
    'M 22.5,19.6 L 22.5,31.4"/>' +
    PC_BASE
};

const PIECE_NAMES = {
  P: 'pawn', N: 'knight', B: 'bishop', R: 'rook', Q: 'queen', K: 'king'
};

/**
 * Build the markup for one piece.
 * @param {string} code two characters: colour ('w'|'b') + type ('K','Q',...)
 * @returns {string} an <svg> element as markup
 */
function pieceSVG(code) {
  const color = code[0];
  const type = code[1].toUpperCase();
  const shape = PC_SHAPES[type] || '';
  const label = (color === 'w' ? 'white ' : 'black ') + (PIECE_NAMES[type] || 'piece');
  return '<svg class="pc pc-' + color + '" viewBox="0 0 45 45" ' +
    'role="img" aria-label="' + label + '" focusable="false">' + shape + '</svg>';
}

/** Human-readable name, used for accessible labels. */
function pieceName(code) {
  return (code[0] === 'w' ? 'White ' : 'Black ') +
    (PIECE_NAMES[code[1].toUpperCase()] || 'piece');
}

/** A tiny inline glyph (used in the capture trays and the report legends). */
function pieceGlyph(code, cls) {
  return '<span class="glyph ' + (cls || '') + '">' + pieceSVG(code) + '</span>';
}

export { pieceSVG, pieceName, pieceGlyph, PIECE_NAMES };
