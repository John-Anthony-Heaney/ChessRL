#!/bin/sh
# Build libchessrl and start the ChessRL web server.
#
#   ./run.sh                                  # newest runs/*/best.crl, port 8000
#   ./run.sh --open                           # ... and open a browser
#   ./run.sh --model runs/foo/best.crl --port 9000
#
# Every argument is forwarded verbatim to py/server.py.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

PYTHON=${PYTHON:-python3}
MAKE=${MAKE:-make}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "run.sh: $PYTHON not found; install Python 3 or set PYTHON=..." >&2
    exit 1
fi

echo "run.sh: building libchessrl ..." >&2
"$MAKE" -j lib

exec "$PYTHON" py/server.py "$@"
