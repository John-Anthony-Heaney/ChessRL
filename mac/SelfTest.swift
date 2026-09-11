//
//  SelfTest.swift -- headless checks for the parts that are easy to get wrong.
//
//  Run with:  ChessRL --selftest
//  Opens no window, touches no AppKit UI, exits non-zero on the first failure.
//

import AppKit

enum SelfTest {

    /// The classic 218-legal-move position: anything that guesses a buffer size
    /// falls over here.
    static let maxMovesFEN = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1"

    /// Filled in by the whole-game check so the summary says what actually ran.
    private static var gameSummary = "no game played"

    static func run(modelPath: String?) -> Bool {
        _ = api_init()

        var checks = 0
        var failures: [String] = []

        func check(_ ok: Bool, _ what: String) {
            checks += 1
            if !ok {
                failures.append(what)
                FileHandle.standardError.write("FAIL: \(what)\n".data(using: .utf8)!)
            }
        }

        bufferProtocol(check)
        coordinateRoundTrip(check)
        clicksOnlyProduceLegalMoves(check)
        promotionVariants(check)
        undoRestoresExactly(check)
        playToTerminalResult(check, modelPath: modelPath)

        let passed = checks - failures.count
        print("selftest: model=\(modelPath ?? "none")  \(gameSummary)")
        print("selftest: \(passed)/\(checks) checks passed" +
              (failures.isEmpty ? "" : "  -- failed: \(failures.joined(separator: ", "))"))
        return failures.isEmpty
    }

    // MARK: - 1. The negative-return buffer retry protocol

    private static func bufferProtocol(_ check: (Bool, String) -> Void) {
        guard let game = GameRef(fen: maxMovesFEN) else {
            check(false, "buffer: could not create the 218-move position")
            return
        }
        defer { game.free() }

        // A deliberately hopeless buffer must report the exact size required.
        var tiny = [CChar](repeating: 0, count: 4)
        let short = tiny.withUnsafeMutableBufferPointer { api_game_legal(game.gid, $0.baseAddress, 4) }
        check(short < -1, "buffer: a too-small buffer must return a negative size, got \(short)")

        let needed = Int(-short)
        var exact = [CChar](repeating: 0, count: needed)
        let wrote = exact.withUnsafeMutableBufferPointer {
            api_game_legal(game.gid, $0.baseAddress, Int32(needed))
        }
        check(wrote >= 0, "buffer: retrying with exactly -ret bytes must succeed, got \(wrote)")
        check(wrote == needed - 1,
              "buffer: -ret is payload+1, expected \(needed - 1) bytes written, got \(wrote)")

        // One byte short of the requirement must still be refused.
        if needed > 1 {
            var almost = [CChar](repeating: 0, count: needed - 1)
            let r = almost.withUnsafeMutableBufferPointer {
                api_game_legal(game.gid, $0.baseAddress, Int32(needed - 1))
            }
            check(r < -1, "buffer: one byte short must still be refused, got \(r)")
        }

        // The wrapper must do all of that for us.
        let moves = game.legalMoves()
        check(moves.count > 200, "buffer: expected >200 legal moves, got \(moves.count)")
        check(moves.count == Set(moves).count, "buffer: legal move list contains duplicates")

        // api_game_state on the same position is far bigger than any sane guess.
        let json = CBuf.text(16, { api_game_state(game.gid, $0, $1) })
        check(json != nil && json!.count > 1000, "buffer: state JSON retry from a 16-byte buffer")
        if let json = json, let s = game.state() {
            check(json.hasPrefix("{") && json.hasSuffix("}"), "buffer: state JSON is complete")
            check(s.legal.count == moves.count, "buffer: state legal[] matches api_game_legal")
        }
    }

    // MARK: - 2. Square <-> coordinate mapping, both orientations

    private static func coordinateRoundTrip(_ check: (Bool, String) -> Void) {
        for flipped in [false, true] {
            let g = BoardGeometry(origin: CGPoint(x: 17, y: 23), squareSize: 64, flipped: flipped)
            var seen = Set<Int>()
            var cells = Set<Int>()
            for sq in Square.all {
                let centre = g.center(of: sq)
                guard let back = g.square(at: centre) else {
                    check(false, "geometry: centre of \(sq.name) fell off the board (flip \(flipped))")
                    continue
                }
                check(back == sq, "geometry: \(sq.name) round trip (flip \(flipped)) gave \(back.name)")
                seen.insert(back.index)
                cells.insert(g.row(of: sq) * 8 + g.column(of: sq))

                // Every corner of the square must map back to the same square.
                let r = g.rect(of: sq)
                let inset = r.insetBy(dx: 1, dy: 1)
                for p in [CGPoint(x: inset.minX, y: inset.minY), CGPoint(x: inset.maxX, y: inset.minY),
                          CGPoint(x: inset.minX, y: inset.maxY), CGPoint(x: inset.maxX, y: inset.maxY)] {
                    check(g.square(at: p) == sq,
                          "geometry: corner of \(sq.name) mis-mapped (flip \(flipped))")
                }
                // Name round trip.
                check(Square(sq.name) == sq, "geometry: name round trip for \(sq.name)")
                check(Square(index: sq.index) == sq, "geometry: index round trip for \(sq.name)")
            }
            check(seen.count == 64, "geometry: 64 distinct squares (flip \(flipped)), got \(seen.count)")
            check(cells.count == 64, "geometry: 64 distinct cells (flip \(flipped)), got \(cells.count)")

            // The two orientations must genuinely differ.
            let a1 = g.rect(of: Square("a1")!)
            check(flipped ? a1.minX > g.origin.x : a1.minX == g.origin.x,
                  "geometry: flip must move a1 (flip \(flipped))")
            // Off-board points.
            check(g.square(at: CGPoint(x: g.origin.x - 1, y: g.origin.y + 1)) == nil,
                  "geometry: point left of the board is off-board")
            check(g.square(at: CGPoint(x: g.origin.x + 8 * 64, y: g.origin.y)) == nil,
                  "geometry: point right of the board is off-board")
        }
    }

    // MARK: - 3. A click can only ever produce a move that is in legal[]

    private static let probeFENs = [
        nil,                                                                     // start position
        "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",  // castling both ways
        "8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",                                     // promotions everywhere
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",          // en passant
        maxMovesFEN
    ]

    private static func clicksOnlyProduceLegalMoves(_ check: (Bool, String) -> Void) {
        for fen in probeFENs {
            guard let game = GameRef(fen: fen), let state = game.state() else {
                check(false, "clicks: could not set up \(fen ?? "start")")
                continue
            }
            defer { game.free() }
            let legal = Set(state.legal)
            check(!legal.isEmpty, "clicks: position has legal moves")

            for flipped in [false, true] {
                let g = BoardGeometry(origin: .zero, squareSize: 48, flipped: flipped)

                // (a) The square under the pixel we would click for each legal
                //     move is exactly that move's destination.
                for uci in state.legal {
                    guard let from = Square(String(uci.prefix(2))),
                          let to = Square(String(uci.dropFirst(2).prefix(2))) else {
                        check(false, "clicks: unparsable UCI \(uci)")
                        continue
                    }
                    check(g.square(at: g.center(of: from)) == from, "clicks: from \(uci)")
                    check(g.square(at: g.center(of: to)) == to, "clicks: to \(uci)")
                }

                // (b) Whatever from/to pair the view considers playable -- the
                //     4-character prefix test it uses -- resolves to a move that
                //     really is in legal[], with or without a promotion suffix.
                for from in Square.all {
                    for to in Square.all where to != from {
                        let stem = from.name + to.name
                        let matches = state.legal.filter { $0.hasPrefix(stem) }
                        guard !matches.isEmpty else { continue }
                        let promos = state.promotions(from: from.name, to: to.name)
                        let played = promos.isEmpty ? [stem] : promos.map { stem + String($0) }
                        for uci in played {
                            check(legal.contains(uci), "clicks: \(uci) offered but not legal")
                        }
                        check(played.count == matches.count,
                              "clicks: \(stem) offers \(played.count) of \(matches.count) variants")
                    }
                }
            }

            // (c) The engine refuses anything outside legal[].
            for bogus in ["a1a1", "e4e5", "h1h8", "zz99", "", "e2e9"] where !legal.contains(bogus) {
                check(!game.play(bogus), "clicks: engine accepted the illegal move '\(bogus)'")
            }
        }
    }

    // MARK: - 4. Promotion variants

    private static func promotionVariants(_ check: (Bool, String) -> Void) {
        // White pawn on a7, black knight on b8: quiet and capturing promotions.
        guard let game = GameRef(fen: "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
              let s = game.state() else {
            check(false, "promotion: could not set up the position")
            return
        }
        defer { game.free() }

        check(s.promotions(from: "a7", to: "a8").count == 4, "promotion: 4 quiet variants on a8")
        check(s.promotions(from: "a7", to: "b8").count == 4, "promotion: 4 capture variants on b8")
        check(s.promotions(from: "a7", to: "a8") == ["q", "r", "b", "n"],
              "promotion: variants come back in Q R B N order")
        check(!s.hasPlainMove(from: "a7", to: "a8"),
              "promotion: the bare from/to must NOT be playable")
        check(s.legal.contains("a7a8q") && s.legal.contains("a7a8n"),
              "promotion: legal[] carries the suffixed moves")
        check(s.promotions(from: "e1", to: "e2").isEmpty,
              "promotion: a king move has no variants")
        check(s.promotions(from: "a7", to: "c8").isEmpty,
              "promotion: an impossible destination has no variants")

        // Each variant is individually playable and produces the right piece.
        for (suffix, kind) in [("q", PieceKind.queen), ("r", .rook), ("b", .bishop), ("n", .knight)] {
            guard let g2 = GameRef(fen: "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1") else { continue }
            defer { g2.free() }
            check(g2.play("a7a8" + suffix), "promotion: a7a8\(suffix) is playable")
            if let after = g2.state() {
                let board = FEN.board(after.fen)
                let piece = board[Square("a8")!.index]
                check(piece?.kind == kind && piece?.isWhite == true,
                      "promotion: a7a8\(suffix) produced the wrong piece")
            }
        }

        // Black promotes too, and the board parser agrees.
        guard let g3 = GameRef(fen: "4k3/8/8/8/8/8/6p1/4K2N b - - 0 1") else { return }
        defer { g3.free() }
        if let s3 = g3.state() {
            check(s3.promotions(from: "g2", to: "h1").count == 4,
                  "promotion: black capture-promotions on h1")
            check(s3.promotions(from: "g2", to: "g1").count == 4,
                  "promotion: black quiet promotions on g1")
        }
    }

    // MARK: - 5. Undo restores the previous state exactly

    private static func undoRestoresExactly(_ check: (Bool, String) -> Void) {
        guard let game = GameRef() else {
            check(false, "undo: could not create a game")
            return
        }
        defer { game.free() }

        // A line with a capture, a castle and a check in it.
        let line = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "f6e4",
                    "f1e1", "e4d6", "f3e5", "c6e5", "e1e5", "f8e7"]
        for (i, move) in line.enumerated() {
            guard let before = CBuf.text(4096, { api_game_state(game.gid, $0, $1) }) else {
                check(false, "undo: could not snapshot before ply \(i)")
                return
            }
            check(game.play(move), "undo: \(move) is playable at ply \(i)")
            check(game.undo(), "undo: api_game_undo reported success at ply \(i)")
            guard let after = CBuf.text(4096, { api_game_state(game.gid, $0, $1) }) else {
                check(false, "undo: could not snapshot after ply \(i)")
                return
            }
            check(before == after, "undo: state after undo differs at ply \(i)")
            check(game.play(move), "undo: \(move) still playable after the undo/redo at ply \(i)")
        }

        // Undo all the way back to the start position.
        guard let start = GameRef(), let startState = start.state() else { return }
        defer { start.free() }
        for _ in line { _ = game.undo() }
        if let back = game.state() {
            check(back.fen == startState.fen, "undo: unwinding the whole line restores the start FEN")
            check(back.ply == 0, "undo: ply is back to zero")
            check(Set(back.legal) == Set(startState.legal), "undo: legal[] matches the start position")
        }
        check(!game.undo(), "undo: undoing an empty game reports no-op")
    }

    // MARK: - 6. A whole game, played out to a terminal result

    private static func playToTerminalResult(_ check: (Bool, String) -> Void, modelPath: String?) {
        guard let game = GameRef() else {
            check(false, "game: could not create a game")
            return
        }
        defer { game.free() }

        let engine = modelPath.flatMap { EngineRef(path: $0) }
        var rng = SystemRandomNumberGenerator()
        var plies = 0
        var last: GameState? = game.state()

        while let s = last, !s.isOver, plies < 600 {
            var move: String?
            if let engine = engine,
               let report = engine.bestMove(in: game, depth: 2, movetime: 60),
               !report.move.isEmpty {
                move = report.move
                check(s.legal.contains(report.move),
                      "game: the engine proposed \(report.move), which is not legal")
            }
            if move == nil { move = s.legal.randomElement(using: &rng) }
            guard let chosen = move, game.play(chosen) else {
                check(false, "game: could not play at ply \(plies)")
                break
            }
            plies += 1
            last = game.state()
        }
        engine?.free()

        if let s = last {
            gameSummary = "\(engine == nil ? "random-vs-random" : "engine-vs-engine") game: "
                + "\(plies) plies, result \(s.result) (\(s.reason.isEmpty ? "?" : s.reason))"
            check(s.isOver, "game: reached a terminal result (result \(s.result) after \(plies) plies)")
            check(!s.reason.isEmpty, "game: the terminal position names a reason")
            check(s.legal.isEmpty || s.result == 3,
                  "game: a decisive result must have no legal moves")
            check(s.moves.count == plies, "game: move history length matches the plies played")
            check(s.historySAN.count == plies, "game: SAN history length matches the plies played")
        } else {
            check(false, "game: lost the state")
        }
    }
}
