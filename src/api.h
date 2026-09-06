/* api.h -- flat C ABI consumed by Python via ctypes (libchessrl.dylib).
 * Every function is thread-safe with respect to distinct handles. */
#ifndef API_H
#define API_H

#ifdef __cplusplus
extern "C" {
#endif

/* Call once. Returns 1. */
int  api_init(void);

/* ---- games ---- */
int  api_game_new(const char *fen);            /* NULL/"" => start position; -> gid, -1 on error */
void api_game_free(int gid);
/* Space-separated UCI strings of every legal move.  Returns bytes written. */
int  api_game_legal(int gid, char *buf, int buflen);
/* 1 on success, 0 if the move is not legal in this position. */
int  api_game_move(int gid, const char *uci);
int  api_game_undo(int gid);
/* JSON: fen, turn, result, reason, check, ply, moves[], san[], legal[], last, material */
int  api_game_state(int gid, char *buf, int buflen);

/* ---- engines ---- */
/* agent_index < 0 selects the best agent stored in the model. -> eid, -1 on error */
int  api_engine_load(const char *model_path, int agent_index);
void api_engine_free(int eid);
/* Picks a move for the current position of `gid` but does NOT play it.
 * Writes JSON {move, san, score, depth, nodes, ms, value, top[]} into buf. */
int  api_engine_move(int eid, int gid, int depth, int movetime_ms, char *buf, int buflen);
/* Raw policy for the current position: JSON [{move,san,prob,logit}...] */
int  api_engine_policy(int eid, int gid, char *buf, int buflen);
/* Number of agents in a model file, and JSON metadata about them. */
int  api_model_info(const char *model_path, char *buf, int buflen);

#ifdef __cplusplus
}
#endif
#endif /* API_H */
