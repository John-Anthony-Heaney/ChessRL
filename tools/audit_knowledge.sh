#!/bin/sh
# tools/audit_knowledge.sh -- fail the build if hand-coded chess knowledge has
# crept into the learning or play path.
#
# ===========================================================================
# WHAT THIS ENFORCES
# ===========================================================================
# docs/FROM_SCRATCH.md is the contract.  It allows exactly two things that are
# not learned: the RULES of chess (src/chess.c) and a DOMAIN-INDEPENDENT search
# algorithm.  Everything else -- piece values, piece-square tables, mobility,
# king safety, pawn structure, MVV-LVA, static exchange evaluation, killers,
# history heuristics, opening books, tablebases, and reward shaping derived from
# the position rather than the result -- is forbidden.
#
# This script greps for those things.  It is deliberately a little paranoid:
# a false positive costs one explicit allow-list comment, a false negative costs
# the entire claim that the agent learned chess from self-play.
#
# ===========================================================================
# WHAT IT SCANS
# ===========================================================================
# By default: src/*.c except src/chess.c (the rules), and except any file that
# has opted out (see AUDIT-EXEMPT below).  Pass file names as arguments to scan
# something else.  Headers are not scanned: they declare, they do not decide.
#
# Comments and the contents of string literals are blanked before matching.
# That is not leniency -- neither can evaluate a position -- and without it
# every file that DOCUMENTS what it must not contain (src/mcts.c and src/search.c
# both do, at length) would fail the audit for saying so.
#
# ===========================================================================
# ESCAPE HATCHES -- both leave a permanent, greppable mark in the source
# ===========================================================================
#   AUDIT-OK                   on a line, suppresses findings on that line.
#   AUDIT-OK-BEGIN / -END      suppresses findings in the region between them.
#       Use for genuine false positives, and say why in the same comment.  The
#       canonical example is api.c's piece-value table, which exists solely so
#       the UI can draw a captured-pieces tray; nothing in the play path reads
#       it.  Run with -v to list every suppression in force.
#
#   AUDIT-EXEMPT               in the first 80 lines of a file, skips the whole
#       file.  Reserved for a file that is compiled ONLY as a measurement
#       baseline and is unreachable from the default play path.  Nothing uses
#       it today: src/search.c is such a baseline and is still scanned in full,
#       because a baseline that quietly re-grew a material term would poison
#       every measurement made with it.
#
# ===========================================================================
# EXIT STATUS
# ===========================================================================
#   0  clean (silent, unless -v)
#   1  forbidden knowledge found; every offending file:line is printed
#   2  the script was used wrongly (no files, unreadable file, ...)

set -u

PROG=$(basename "$0")
VERBOSE=0

usage() {
    cat <<USAGE
usage: $PROG [-v] [file.c ...]
  -v   also report allow-listed suppressions and exempt files
  With no files, scans src/*.c except src/chess.c.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        -v|--verbose) VERBOSE=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; break ;;
        -*)           echo "$PROG: unknown option '$1'" >&2; usage >&2; exit 2 ;;
        *)            break ;;
    esac
done

# ------------------------------------------------------------------- targets

ROOT=$(cd "$(dirname "$0")/.." 2>/dev/null && pwd) || ROOT=.

if [ $# -gt 0 ]; then
    FILES="$*"
else
    FILES=""
    for f in "$ROOT"/src/*.c; do
        [ -e "$f" ] || continue
        case "$f" in
            */chess.c) continue ;;      # the rules of the game, explicitly allowed
        esac
        FILES="$FILES $f"
    done
fi

[ -n "${FILES# }" ] || { echo "$PROG: nothing to scan" >&2; exit 2; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/audit_knowledge.XXXXXX") || exit 2
trap 'rm -rf "$TMP"' EXIT INT TERM HUP

HITS="$TMP/hits"
SUPP="$TMP/supp"
: > "$HITS"
: > "$SUPP"

# ---------------------------------------------------------------- constants
#
# Identifiers that are RULES, not evaluation, and are therefore always allowed
# wherever they appear.  chess.h defines insufficient_material() as the FIDE
# dead-position test -- "neither side CAN mate" -- which a random mover needs
# just as much as a trained one.  It is blanked before matching so that the
# `material` pattern does not trip over it.
RULES_OK='insufficient_material'

# Numbers that look like a piece value.  Three or more of them on one line (or
# across three consecutive lines) is a piece-value table however it is spelled:
# pawns-as-1 or pawns-as-100, with or without the king.
PIECE_NUMS='1 3 5 9 100 300 305 310 315 320 325 330 335 350 500 510 525 900 950 975 1000 1025 20000'

# ------------------------------------------------------------------ scanning

report() { printf '%s:%s: %s\n' "$1" "$2" "$3" >> "$HITS"; }

for f in $FILES; do
    [ -r "$f" ] || { echo "$PROG: cannot read '$f'" >&2; exit 2; }
    rel=${f#"$ROOT"/}

    if head -n 80 "$f" | grep -q 'AUDIT-EXEMPT'; then
        [ "$VERBOSE" -eq 1 ] && echo "$PROG: skipping $rel (AUDIT-EXEMPT)"
        continue
    fi

    # -- 1. line-preserving strip of comments and string/char literal contents.
    #       Every removed byte becomes a space, so line and column numbers in a
    #       finding still point at the real source.
    awk '
    BEGIN { inc = 0 }
    {
        line = $0; n = length(line); out = ""; i = 1; ins = 0; inch = 0
        while (i <= n) {
            c = substr(line, i, 1); d = substr(line, i, 2)
            if (inc) {
                if (d == "*/") { inc = 0; out = out "  "; i += 2 }
                else           { out = out " ";           i += 1 }
            } else if (ins) {
                if (c == "\\") { out = out "  "; i += 2 }
                else if (c == "\"") { ins = 0; out = out "\""; i += 1 }
                else { out = out " "; i += 1 }
            } else if (inch) {
                if (c == "\\") { out = out "  "; i += 2 }
                else if (c == "'"'"'") { inch = 0; out = out "'"'"'"; i += 1 }
                else { out = out " "; i += 1 }
            } else if (d == "/*") { inc = 1;  out = out "  "; i += 2 }
            else if (d == "//")   { while (i <= n) { out = out " "; i++ } }
            else if (c == "\"")   { ins = 1;  out = out c; i += 1 }
            else if (c == "'"'"'"){ inch = 1; out = out c; i += 1 }
            else                  { out = out c; i += 1 }
        }
        print out
    }' "$f" | sed "s/$RULES_OK/                     /g" > "$TMP/code"

    # -- 2. lines the author has explicitly allow-listed.  Read from the
    #       ORIGINAL file, because the markers live in comments.
    awk '
    /AUDIT-OK-BEGIN/ { on = 1 }
    { if (on || /AUDIT-OK/) print NR }
    /AUDIT-OK-END/   { on = 0 }
    ' "$f" > "$TMP/allow"

    allowed() { grep -qx "$1" "$TMP/allow"; }

    emit() {   # emit <line> <text> <what>
        if allowed "$1"; then
            printf '%s:%s: [allow-listed] %s\n' "$rel" "$1" "$3" >> "$SUPP"
        else
            report "$rel" "$1" "$3 -- $2"
        fi
    }

    # -- 3. forbidden identifiers ------------------------------------------
    #    Anchored so they match code, not English: `material` matches
    #    material_cp / white_material / MAT_CP, `see` only as a call or as the
    #    uppercase acronym, `center`/`centre` only when joined to a scoring
    #    word.  Each pattern names what it is looking for.
    scan() {   # scan <extended-regex> <description>
        grep -nE "$1" "$TMP/code" 2>/dev/null | while IFS=: read -r ln rest; do
            emit "$ln" "$(printf '%s' "$rest" | sed 's/^[ \t]*//' | cut -c1-90)" "$2"
        done
    }

    scan '[A-Za-z_]*[Mm][Aa][Tt][Ee][Rr][Ii][Aa][Ll][A-Za-z_]*' 'material term'
    scan '\b[A-Za-z_]*(MAT_CP|mat_cp|MATERIAL|matbal|mat_balance)[A-Za-z_]*' 'material term'
    scan '[Mm][Vv][Vv]|[Ll][Vv][Aa]'                             'MVV-LVA move ordering'
    scan '\bSEE\b|static_exchange|\b[A-Za-z_]*see(_[A-Za-z_]+)?[ \t]*\(' 'static exchange evaluation'
    scan '\b[A-Za-z_]*(psqt|PSQT)[A-Za-z_]*|\b(pst|PST)\b|piece_square|PIECE_SQUARE' 'piece-square table'
    scan '[Mm][Oo][Bb][Ii][Ll][Ii][Tt][Yy]'                      'mobility term'
    scan '[Kk][Ii][Nn][Gg][ _]?[Ss][Aa][Ff][Ee][Tt][Yy]|king_shield|pawn_shield' 'king safety term'
    scan '[Pp][Aa][Ww][Nn][ _]?[Ss][Tt][Rr][Uu][Cc][Tt]|passed[ _]?pawn|doubled[ _]?pawn|isolated[ _]?pawn|backward[ _]?pawn' 'pawn structure term'
    scan '\b[A-Za-z_]*[Tt][Ee][Mm][Pp][Oo][A-Za-z_]*'            'tempo term'
    scan '(centre|center|CENTRE|CENTER)[ _]?(bonus|control|weight|score|table|tab|BONUS|CONTROL)|\b(centre|center)_[A-Za-z_]+' 'centre-control bonus'
    scan '\b[A-Za-z_]*(piece_value|PIECE_VALUE|piece_val|PIECE_VAL|PIECE_CP|piece_cp)[A-Za-z_]*' 'piece-value table'
    scan '\b[A-Za-z_]*killer[A-Za-z_]*|\bKILLER'                 'killer moves'
    scan 'history_heuristic|hist_add|butterfly|\bHIST_MAX\b'     'history heuristic'
    scan 'bishop_pair|BISHOP_PAIR|outpost|OUTPOST|space_bonus|rook_open|open_file' 'positional bonus'
    scan 'opening_book|book_move|polyglot|tablebase|syzygy|\bEGTB\b|\begtb\b' 'opening book / tablebase'
    scan '\b[A-Za-z_]*shaping[A-Za-z_]*|\bSHAPING\b|potential_phi|\bphi\b'   'reward shaping'

    # -- 4. piece values as bare numbers ------------------------------------
    #    The signature of a piece-value table is a COMMA-SEPARATED LIST of
    #    numeric literals, three or more of which are piece values, in at least
    #    three distinct denominations.  Written out that carefully because the
    #    naive "three piece-value-looking numbers nearby" test flags every
    #    xoshiro RNG in the tree (`rotl(s[1] * 5, 7) * 9`).
    #
    #    Hexadecimal literals and integer/float suffixes are removed first, and
    #    the window spans three lines because a real table is wrapped:
    #
    #        static const int V[6] = {
    #            100, 320, 330,
    #            500, 900, 0 };
    awk -v nums="$PIECE_NUMS" '
    function isval(x,   i) { for (i in want) if (x + 0 == i + 0) return 1; return 0 }
    {
        buf[NR] = $0
        lo = (NR > 2) ? NR - 2 : 1
        s = ""
        for (i = lo; i <= NR; i++) s = s " " buf[i]

        # Hex literals are never piece values.  (awk gsub has no capture
        # groups, so numeric suffixes are stripped per token below, not here.)
        gsub(/0[xX][0-9A-Fa-f]+[uUlL]*/, " ", s)

        while (match(s, /[0-9][0-9.]*[uUlLfFdD]*([ \t]*,[ \t]*[0-9][0-9.]*[uUlLfFdD]*){2,}/)) {
            run = substr(s, RSTART, RLENGTH)
            s   = substr(s, RSTART + RLENGTH)
            n = split(run, t, /[ \t]*,[ \t]*/)
            hits = 0; delete seen
            for (i = 1; i <= n; i++) {
                sub(/[uUlLfFdD]+$/, "", t[i])              # 12ull / 1.5f -> 12 / 1.5
                if (isval(t[i]) && !((t[i] + 0) in seen)) { seen[t[i] + 0] = 1; hits++ }
            }
            if (hits >= 3) { print NR; break }
        }
    }
    BEGIN { split(nums, a, " "); for (i in a) want[a[i]] = 1 }
    ' "$TMP/code" | awk 'NR == 1 || $1 > prev + 2 { print } { prev = $1 }' | while read -r ln; do
        # One wrapped table lights up the whole three-line window; report only
        # the first line of each run so a finding means one table, not three.
        txt=$(sed -n "${ln}p" "$TMP/code" | sed 's/^[ \t]*//' | cut -c1-90)
        emit "$ln" "$txt" "piece values as literal numbers (3+ distinct of {1,3,5,9 / 100,320,330,500,900,...} in one list)"
    done

    # -- 5. a table of 64 constants -----------------------------------------
    #    Any brace initialiser holding 48 or more numeric literals, and any
    #    const array dimensioned [64] (or [N][64]) with an initialiser.  48 is
    #    below 64 on purpose: half a piece-square table is still a piece-square
    #    table, and a 32-entry table indexed by a mirrored square is the usual
    #    way of hiding one.
    awk '
    {
        if (start == 0 && $0 ~ /=[ \t]*\{/) { start = NR; depth = 0; cnt = 0; buf = "" }
        if (start != 0) {
            buf = buf " " $0
            o = gsub(/\{/, "{"); c = gsub(/\}/, "}")
            depth += o - c
            if (depth <= 0) {
                s = buf; gsub(/[^0-9]+/, " ", s)
                n = split(s, t, " "); cnt = 0
                for (i = 1; i <= n; i++) if (t[i] != "") cnt++
                if (cnt >= 48) print start "\t" cnt
                start = 0
            }
        }
    }' "$TMP/code" | while IFS="$(printf '\t')" read -r ln cnt; do
        emit "$ln" "$(sed -n "${ln}p" "$TMP/code" | sed 's/^[ \t]*//' | cut -c1-90)" \
             "table of $cnt constants (a 64-entry table is a piece-square table)"
    done

    scan '\[[ \t]*64[ \t]*\][ \t]*(\[[^]]*\][ \t]*)*=[ \t]*\{|\[[^]]*\][ \t]*\[[ \t]*64[ \t]*\][ \t]*=[ \t]*\{' \
         'initialised 64-entry table (piece-square table shape)'
done

# ------------------------------------------------------------------- verdict

if [ "$VERBOSE" -eq 1 ] && [ -s "$SUPP" ]; then
    echo "$PROG: allow-listed (AUDIT-OK) findings:"
    sort -u "$SUPP" | sed 's/^/    /'
fi

if [ -s "$HITS" ]; then
    echo "$PROG: FORBIDDEN CHESS KNOWLEDGE in the learning/play path" >&2
    echo "  (docs/FROM_SCRATCH.md: the only evaluation is the network;" >&2
    echo "   the only reward is the game result)" >&2
    echo >&2
    sort -u -t: -k1,1 -k2,2n "$HITS" | sed 's/^/  /' >&2
    echo >&2
    n=$(sort -u "$HITS" | wc -l | tr -d ' ')
    echo "$PROG: $n finding(s).  Remove it, or -- if it genuinely cannot" >&2
    echo "  evaluate a position -- mark it AUDIT-OK with a reason." >&2
    exit 1
fi

[ "$VERBOSE" -eq 1 ] && echo "$PROG: clean"
exit 0
