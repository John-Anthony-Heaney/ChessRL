# HTTP API contract (py/server.py  <->  web/)

Static files are served from `web/` at `/`. All API routes are under `/api/`,
return `application/json`, and never cache.

### `GET /api/new?fen=<optional FEN>&human=white|black`
Creates a game. -> `{"gid": 3, "state": <State>}`

### `GET /api/state?gid=N`  -> `<State>`

### `POST /api/move`  body `{"gid":N,"move":"e2e4"}` (promotions: `"e7e8q"`)
Rejects illegal moves with HTTP 400 `{"error":"illegal move"}`.
-> `{"state": <State>}`

### `POST /api/ai`  body `{"gid":N,"depth":4,"movetime":800}`
Makes the engine play one move for the side to move.
-> `{"move":"g1f3","san":"Nf3","score":31,"depth":5,"nodes":184203,"ms":712,
     "value":0.08,"top":[{"move":"g1f3","san":"Nf3","prob":0.31}, ...],
     "state": <State>}`

### `POST /api/undo` body `{"gid":N,"plies":2}` -> `{"state": <State>}`

### `GET /api/hint?gid=N` -> same shape as `/api/ai` but does **not** play the move.

### `GET /api/watch?a=<agent>&b=<agent>&plies=400`
Plays a complete agent-vs-agent game server-side and returns it for playback.
-> `{"result":1,"reason":"checkmate","moves":["e2e4",...],"sans":["e4",...],
     "fens":[...],"evals":[0.02,...],"white":"agent 12","black":"agent 7"}`

### `GET /api/model` -> `{"path":..., "generation":312, "n_agents":256,
      "agents":[{"i":0,"elo":1873.2,"rank":1,"temperature":0.61,...}, ...]}`

### `GET /api/report` -> the strategy analysis JSON produced by `py/report.py`.

### `<State>` object
```json
{
  "gid": 3,
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "turn": "white",
  "ply": 0,
  "result": 0,                  // 0 ongoing, 1 white win, 2 black win, 3 draw
  "reason": "",                 // "checkmate" | "stalemate" | "fifty-move" |
                                // "threefold repetition" | "insufficient material" |
                                // "move limit"
  "check": false,
  "legal": ["a2a3","a2a4", ...],// UCI, exactly the moves the UI may allow
  "san":   ["a3","a4", ...],    // parallel to `legal`
  "moves": ["e2e4","e7e5"],     // full move history in UCI
  "history_san": ["e4","e5"],
  "last": {"from":"e2","to":"e4"} | null,
  "material": {"white": 39, "black": 39},
  "captured": {"white": ["p"], "black": []}
}
```

The UI must derive **all** legality from `legal`; it must never allow a move
outside that list, and clicking a piece highlights exactly the `to` squares of
the `legal` entries whose `from` matches.
