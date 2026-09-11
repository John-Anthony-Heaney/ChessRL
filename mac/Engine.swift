//
//  Engine.swift -- the only file that talks to the C engine.
//
//  Everything below obeys the api.c truncation contract:
//
//      ret >= 0   ->  bytes written, buffer is NUL terminated
//      ret == -1  ->  hard error (bad handle / bad argument)
//      ret <  -1  ->  -(bytes required), retry with a buffer of exactly -ret
//
//  Nothing here assumes a fixed buffer is big enough: `CBuf.text` always
//  retries on a negative return.  api_game_legal in a 218-move position is the
//  case that punishes anyone who guesses.
//

import Foundation

// MARK: - Buffer protocol

enum CBuf {

    /// Calls a buffer-filling C entry point and returns its payload, growing the
    /// buffer exactly once if the first attempt was too small.
    static func text(_ initial: Int = 512,
                     _ body: (UnsafeMutablePointer<CChar>?, Int32) -> Int32) -> String? {
        var cap = Int32(max(16, min(initial, 1 << 20)))
        // The contract makes one retry sufficient; the loop is belt and braces.
        for _ in 0 ..< 4 {
            var buf = [CChar](repeating: 0, count: Int(cap))
            let ret: Int32 = buf.withUnsafeMutableBufferPointer { body($0.baseAddress, cap) }
            if ret >= 0 {
                let n = min(Int(ret), buf.count)
                var bytes = [UInt8]()
                bytes.reserveCapacity(n)
                for i in 0 ..< n { bytes.append(UInt8(bitPattern: buf[i])) }
                return String(decoding: bytes, as: UTF8.self)
            }
            if ret == -1 { return nil }              // genuine failure, never a size
            let needed = Int(-ret)
            if needed <= Int(cap) { return nil }     // would loop forever; refuse
            cap = Int32(needed)
        }
        return nil
    }
}

// MARK: - Decoded payloads

struct LastMove: Codable {
    let from: String
    let to: String
    let san: String
}

struct Material: Codable {
    let white: Int
    let black: Int
}

struct CapturedPieces: Codable {
    let white: [String]
    let black: [String]
}

/// The State object documented in docs/API.md.
struct GameState: Codable {
    let gid: Int
    let fen: String
    let turn: String
    let ply: Int
    let result: Int              // 0 ongoing, 1 white win, 2 black win, 3 draw
    let reason: String
    let check: Bool
    let legal: [String]
    let san: [String]
    let moves: [String]
    let historySAN: [String]
    let last: LastMove?
    let material: Material
    let captured: CapturedPieces

    enum CodingKeys: String, CodingKey {
        case gid, fen, turn, ply, result, reason, check, legal, san, moves
        case historySAN = "history_san"
        case last, material, captured
    }

    var whiteToMove: Bool { turn == "white" }
    var isOver: Bool { result != 0 }

    /// The `to` squares of every legal move starting on `from`.
    func destinations(from square: String) -> [String] {
        legal.filter { $0.hasPrefix(square) }.map { String($0.dropFirst(2).prefix(2)) }
    }

    /// The promotion suffixes available for this from/to pair, in Q R B N order.
    func promotions(from: String, to: String) -> [Character] {
        let stem = from + to
        let found = legal.filter { $0.count == 5 && $0.hasPrefix(stem) }
                         .compactMap { $0.last }
        return ["q", "r", "b", "n"].filter { found.contains($0) }
    }

    /// True when `from`+`to` is playable with no promotion suffix.
    func hasPlainMove(from: String, to: String) -> Bool {
        legal.contains(from + to)
    }
}

struct TopMove: Codable {
    let move: String
    let san: String
    let prob: Double
    let logit: Double
}

/// The object returned by api_engine_move.
struct EngineReport: Codable {
    let move: String
    let san: String
    let score: Int
    let depth: Int
    let nodes: Int
    let ms: Int
    let value: Double            // always from WHITE's point of view
    let top: [TopMove]

    var knps: Double { ms > 0 ? Double(nodes) / Double(ms) : 0 }
}

struct AgentInfo: Codable {
    let i: Int
    let elo: Double
    let rank: Int
    let temperature: Double
    let entropy_coef: Double
    let lr_scale: Double
    let shaping: Double
}

struct ModelInfo: Codable {
    let generation: Int
    let n_agents: Int
    let agents: [AgentInfo]

    /// api_model_info sorts by Elo, so rank 1 is the agent api_engine_load(-1) picks.
    var champion: AgentInfo? { agents.first }
}

// MARK: - Handles

/// A game handle.  Not thread safe: exactly one thread may own an instance.
final class GameRef {

    let gid: Int32
    private var freed = false

    init?(fen: String? = nil) {
        _ = api_init()
        let g: Int32
        if let fen = fen, !fen.isEmpty {
            g = fen.withCString { api_game_new($0) }
        } else {
            g = api_game_new(nil)
        }
        if g < 0 { return nil }
        gid = g
    }

    deinit { free() }

    func free() {
        if !freed {
            freed = true
            api_game_free(gid)
        }
    }

    func state() -> GameState? {
        guard let json = CBuf.text(4096, { api_game_state(gid, $0, $1) }),
              let data = json.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(GameState.self, from: data)
    }

    /// Space-separated UCI, straight from api_game_legal.
    func legalMoves() -> [String] {
        guard let s = CBuf.text(64, { api_game_legal(gid, $0, $1) }) else { return [] }
        return s.split(separator: " ").map(String.init)
    }

    @discardableResult
    func play(_ uci: String) -> Bool {
        uci.withCString { api_game_move(gid, $0) } == 1
    }

    @discardableResult
    func undo() -> Bool {
        api_game_undo(gid) == 1
    }

    /// Replays a UCI history onto a fresh game so a background thread can search
    /// a faithful copy -- repetition history and all -- without ever touching the
    /// handle the UI is showing.
    static func replay(_ moves: [String]) -> GameRef? {
        guard let g = GameRef() else { return nil }
        for m in moves {
            if !g.play(m) { g.free(); return nil }
        }
        return g
    }
}

/// A loaded network.  Only the engine queue may touch one of these.
final class EngineRef {

    let eid: Int32
    let path: String
    private var freed = false

    init?(path: String, agent: Int32 = -1) {
        _ = api_init()
        let e = path.withCString { api_engine_load($0, agent) }
        if e < 0 { return nil }
        eid = e
        self.path = path
    }

    deinit { free() }

    func free() {
        if !freed {
            freed = true
            api_engine_free(eid)
        }
    }

    func bestMove(in game: GameRef, depth: Int32, movetime: Int32) -> EngineReport? {
        guard let json = CBuf.text(1024, { api_engine_move(eid, game.gid, depth, movetime, $0, $1) }),
              let data = json.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(EngineReport.self, from: data)
    }

    static func info(path: String) -> ModelInfo? {
        _ = api_init()
        guard let json = CBuf.text(8192, { b, l in path.withCString { api_model_info($0, b, l) } }),
              let data = json.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(ModelInfo.self, from: data)
    }
}

// MARK: - Finding a model on disk

enum ModelLocator {

    /// Newest runs/<name>/best.crl found by walking up from the executable and
    /// from the working directory.  Returns nil when the app has no brain.
    ///
    /// This touches the file system, which on macOS can mean a TCC prompt when
    /// the repository lives somewhere protected such as ~/Desktop, and that
    /// prompt blocks the caller until the user answers it.  So: never call this
    /// on the main thread, and never walk past the user's home directory.
    static func find(explicit: String?) -> String? {
        let fm = FileManager.default
        if let explicit = explicit {
            return fm.fileExists(atPath: explicit) ? explicit : nil
        }

        let home = URL(fileURLWithPath: NSHomeDirectory()).resolvingSymlinksInPath().path
        var roots: [URL] = [URL(fileURLWithPath: fm.currentDirectoryPath)]
        var up = Bundle.main.bundleURL.resolvingSymlinksInPath()
        // build/ChessRL.app -> build -> the repository root is three of these;
        // a fourth is slack.  Stop before wandering into the home folder.
        for _ in 0 ..< 4 {
            if up.path == "/" || up.path == home { break }
            roots.append(up)
            up = up.deletingLastPathComponent()
        }

        var best: (path: String, date: Date)?
        var seen = Set<String>()
        for root in roots {
            let runs = root.appendingPathComponent("runs")
            guard seen.insert(runs.path).inserted,
                  let kids = try? fm.contentsOfDirectory(at: runs,
                                                         includingPropertiesForKeys: [.contentModificationDateKey],
                                                         options: [.skipsHiddenFiles])
            else { continue }
            for kid in kids {
                let candidate = kid.appendingPathComponent("best.crl")
                guard fm.fileExists(atPath: candidate.path) else { continue }
                let date = (try? candidate.resourceValues(forKeys: [.contentModificationDateKey]))?
                    .contentModificationDate ?? .distantPast
                if best == nil || date > best!.date { best = (candidate.path, date) }
            }
        }
        return best?.path
    }
}
