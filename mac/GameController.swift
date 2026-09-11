//
//  GameController.swift -- glue between the board, the panel and the engine.
//
//  Threading rules, which the whole design hangs off:
//
//    * The live game handle is created, mutated, read and freed ONLY on the main
//      thread.  Every one of those calls is microseconds.
//    * A search never touches that handle.  Before dispatching, the controller
//      hands the background queue a plain Swift array of UCI moves; the queue
//      replays them into a private game handle it exclusively owns, searches it
//      and frees it.  Two threads therefore never share a handle.
//    * Every search is stamped with a generation counter.  New Game, Undo,
//      scrubbing and quitting all bump it, so a result that arrives late is
//      dropped instead of corrupting a position it no longer describes.
//

import AppKit

final class GameController: NSViewController, BoardViewDelegate, SidePanelDelegate,
                           NSMenuItemValidation {

    // MARK: - Views

    private let boardView = BoardView()
    private let panel = SidePanel()

    // MARK: - Game state (main thread only)

    private var game: GameRef?
    private var state: GameState?
    private var reviewState: GameState?
    private var reviewPly: Int?
    private var humanIsWhite = true
    private var resigned = false
    private var analysis: Analysis?
    private var hintFrom: Square?
    private var hintTo: Square?

    // MARK: - Engine

    private let engineQueue = DispatchQueue(label: "com.chessrl.engine", qos: .userInitiated)
    private var engine: EngineRef?          // engineQueue only
    private let requestedModel: String?     // --model PATH, or nil to go looking
    private var modelPath: String?
    private var modelInfo: ModelInfo?
    private var hasEngine = false           // main-thread mirror
    private var searchGeneration = 0
    private var thinking = false
    private var terminating = false

    /// True when there is no engine: the human then drives both armies.
    private var bothSides: Bool { !hasEngine }

    // MARK: - Init

    init(requestedModel: String?) {
        self.requestedModel = requestedModel
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) { nil }

    // MARK: - Layout

    override func loadView() {
        let root = NSView(frame: NSRect(x: 0, y: 0, width: 1120, height: 760))

        let sidebar = NSVisualEffectView()
        sidebar.material = .sidebar
        sidebar.blendingMode = .behindWindow
        sidebar.state = .followsWindowActiveState
        sidebar.translatesAutoresizingMaskIntoConstraints = false

        let boardArea = NSView()
        boardArea.translatesAutoresizingMaskIntoConstraints = false
        boardView.translatesAutoresizingMaskIntoConstraints = false
        boardView.delegate = self
        boardArea.addSubview(boardView)

        let divider = NSBox()
        divider.boxType = .separator
        divider.translatesAutoresizingMaskIntoConstraints = false

        panel.delegate = self
        sidebar.addSubview(panel)

        root.addSubview(boardArea)
        root.addSubview(divider)
        root.addSubview(sidebar)

        let grow = boardView.widthAnchor.constraint(equalTo: boardArea.widthAnchor, constant: -28)
        grow.priority = .defaultLow
        let growH = boardView.heightAnchor.constraint(equalTo: boardArea.heightAnchor, constant: -28)
        growH.priority = .defaultLow

        NSLayoutConstraint.activate([
            boardArea.leadingAnchor.constraint(equalTo: root.leadingAnchor),
            boardArea.topAnchor.constraint(equalTo: root.topAnchor),
            boardArea.bottomAnchor.constraint(equalTo: root.bottomAnchor),
            boardArea.trailingAnchor.constraint(equalTo: divider.leadingAnchor),

            divider.topAnchor.constraint(equalTo: root.topAnchor),
            divider.bottomAnchor.constraint(equalTo: root.bottomAnchor),
            divider.widthAnchor.constraint(equalToConstant: 1),
            divider.trailingAnchor.constraint(equalTo: sidebar.leadingAnchor),

            sidebar.trailingAnchor.constraint(equalTo: root.trailingAnchor),
            sidebar.topAnchor.constraint(equalTo: root.topAnchor),
            sidebar.bottomAnchor.constraint(equalTo: root.bottomAnchor),
            sidebar.widthAnchor.constraint(equalToConstant: 340),

            panel.leadingAnchor.constraint(equalTo: sidebar.leadingAnchor),
            panel.trailingAnchor.constraint(equalTo: sidebar.trailingAnchor),
            panel.topAnchor.constraint(equalTo: sidebar.topAnchor),
            panel.bottomAnchor.constraint(equalTo: sidebar.bottomAnchor),

            boardView.centerXAnchor.constraint(equalTo: boardArea.centerXAnchor),
            boardView.centerYAnchor.constraint(equalTo: boardArea.centerYAnchor),
            boardView.widthAnchor.constraint(equalTo: boardView.heightAnchor),
            boardView.widthAnchor.constraint(lessThanOrEqualTo: boardArea.widthAnchor, constant: -28),
            boardView.heightAnchor.constraint(lessThanOrEqualTo: boardArea.heightAnchor, constant: -28),
            grow, growH
        ])

        view = root
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        panel.showSearchingForModel()
        panel.showAnalysis(nil)
        loadEngine()
        startNewGame()
    }

    override func viewDidAppear() {
        super.viewDidAppear()
        view.window?.makeFirstResponder(boardView)
    }

    // MARK: - Engine lifecycle

    /// Finding the model means touching the file system, which can block on a
    /// TCC prompt, so it happens on the engine queue with the window already up
    /// rather than on the way to opening it.
    private func loadEngine() {
        let requested = requestedModel
        engineQueue.async { [weak self] in
            let path = ModelLocator.find(explicit: requested)
            let engine = path.flatMap { EngineRef(path: $0) }
            let info = path.flatMap { EngineRef.info(path: $0) }
            DispatchQueue.main.async {
                guard let self = self, !self.terminating else { return }
                self.engine = engine
                self.hasEngine = (engine != nil)
                self.modelPath = engine == nil ? nil : path
                self.modelInfo = info
                self.panel.showModel(path: self.modelPath, info: info)
                self.refresh()
                self.thinkIfEnginesTurn()
            }
        }
    }

    /// Abandons any search in flight: its result will be dropped on arrival.
    private func abandonSearch() {
        searchGeneration &+= 1
        thinking = false
    }

    func prepareToTerminate() {
        terminating = true
        abandonSearch()
        boardView.stopAnimations()
    }

    // MARK: - Game lifecycle

    func startNewGame() {
        abandonSearch()
        boardView.stopAnimations()
        clearHint()
        reviewPly = nil
        reviewState = nil
        resigned = false
        analysis = nil

        game?.free()
        game = GameRef()
        state = game?.state()

        switch panel.chosenColour {
        case 1:  humanIsWhite = false
        case 2:  humanIsWhite = Bool.random()
        default: humanIsWhite = true
        }
        boardView.boardFlipped = !humanIsWhite

        panel.showAnalysis(nil)
        refresh()
        thinkIfEnginesTurn()
    }

    private func thinkIfEnginesTurn() {
        guard !terminating, hasEngine, reviewPly == nil, !resigned,
              let s = state, !s.isOver, s.whiteToMove != humanIsWhite
        else { return }
        startSearch(playIt: true)
    }

    // MARK: - Searching

    private func startSearch(playIt: Bool) {
        guard hasEngine, let s = state, !s.isOver else { return }
        let level = SidePanel.levels[min(panel.chosenLevel, SidePanel.levels.count - 1)]
        let moves = s.moves
        let whiteToMove = s.whiteToMove
        searchGeneration &+= 1
        let generation = searchGeneration
        thinking = true
        panel.showThinking(true)
        refresh()

        engineQueue.async { [weak self] in
            guard let self = self else { return }
            var report: EngineReport?
            // A private game handle, owned start to finish by this queue.
            if let engine = self.engine, let scratch = GameRef.replay(moves) {
                report = engine.bestMove(in: scratch, depth: level.depth, movetime: level.movetime)
                scratch.free()
            }
            DispatchQueue.main.async {
                guard !self.terminating, generation == self.searchGeneration else { return }
                self.thinking = false
                self.panel.showThinking(false)
                self.finishSearch(report, whiteToMove: whiteToMove, playIt: playIt)
            }
        }
    }

    private func finishSearch(_ report: EngineReport?, whiteToMove: Bool, playIt: Bool) {
        guard let report = report, !report.move.isEmpty else {
            refresh()
            return
        }
        analysis = Analysis(report: report, whiteToMove: whiteToMove,
                            caption: playIt ? "engine move \(report.san)"
                                            : "hint: \(report.san)")
        panel.showAnalysis(analysis)

        if playIt {
            clearHint()
            apply(uci: report.move, animated: true)
        } else {
            hintFrom = Square(String(report.move.prefix(2)))
            hintTo = Square(String(report.move.dropFirst(2).prefix(2)))
            refresh()
        }
    }

    // MARK: - Moves

    private func apply(uci: String, animated: Bool) {
        guard let game = game, let before = state else { return }
        guard before.legal.contains(uci) else { return }   // belt and braces

        let from = Square(String(uci.prefix(2)))
        let piece = from.flatMap { FEN.board(before.fen)[$0.index] }
        let to = Square(String(uci.dropFirst(2).prefix(2)))

        guard game.play(uci) else { return }
        state = game.state()
        reviewPly = nil
        reviewState = nil
        refresh()

        if animated, let f = from, let t = to, let p = piece {
            boardView.animate(piece: p, from: f, to: t)
        }
    }

    // MARK: - BoardViewDelegate

    func boardView(_ view: BoardView, didPlay uci: String) {
        guard !thinking, reviewPly == nil, let s = state, !s.isOver, !resigned else { return }
        clearHint()
        apply(uci: uci, animated: false)
        thinkIfEnginesTurn()
    }

    // MARK: - SidePanelDelegate

    func sidePanelNewGame(_ panel: SidePanel) { startNewGame() }
    func sidePanelFlip(_ panel: SidePanel) { flipBoard() }
    func sidePanelUndo(_ panel: SidePanel) { undoMove() }
    func sidePanelHint(_ panel: SidePanel) { requestHint() }
    func sidePanelResign(_ panel: SidePanel) { resignGame() }

    func sidePanel(_ panel: SidePanel, scrubTo ply: Int) {
        guard let live = state else { return }
        let clamped = max(0, min(ply, live.ply))
        if clamped == live.ply {
            sidePanelReturnToLive(panel)
            return
        }
        abandonSearch()
        panel.showThinking(false)
        boardView.stopAnimations()
        clearHint()
        reviewPly = clamped
        reviewState = replayState(plies: clamped)
        refresh()
    }

    func sidePanelReturnToLive(_ panel: SidePanel) {
        guard reviewPly != nil else { return }
        reviewPly = nil
        reviewState = nil
        boardView.stopAnimations()
        refresh()
        thinkIfEnginesTurn()
    }

    /// A throwaway handle replayed on the main thread, used only for scrubbing.
    private func replayState(plies: Int) -> GameState? {
        guard let live = state else { return nil }
        guard let scratch = GameRef.replay(Array(live.moves.prefix(plies))) else { return nil }
        defer { scratch.free() }
        return scratch.state()
    }

    // MARK: - Commands

    func flipBoard() {
        boardView.boardFlipped.toggle()
        refresh()
    }

    func undoMove() {
        guard let game = game, let s = state, s.ply > 0 else { return }
        abandonSearch()
        panel.showThinking(false)
        boardView.stopAnimations()
        clearHint()
        reviewPly = nil
        reviewState = nil
        resigned = false
        analysis = nil
        panel.showAnalysis(nil)

        game.undo()
        if !bothSides, let after = game.state(), after.ply > 0, after.whiteToMove != humanIsWhite {
            game.undo()                    // take back the pair, so it is our move again
        }
        state = game.state()
        refresh()
        thinkIfEnginesTurn()
    }

    func requestHint() {
        guard hasEngine, !thinking, reviewPly == nil, !resigned,
              let s = state, !s.isOver else { return }
        startSearch(playIt: false)
    }

    func resignGame() {
        guard let s = state, !s.isOver, !resigned, !bothSides else { return }
        abandonSearch()
        panel.showThinking(false)
        resigned = true
        refresh()
    }

    private func clearHint() {
        hintFrom = nil
        hintTo = nil
    }

    // MARK: - Refresh

    private var displayState: GameState? { reviewState ?? state }

    private var interactiveNow: Bool {
        guard !terminating, reviewPly == nil, !thinking, !resigned,
              let s = state, !s.isOver else { return false }
        return bothSides || s.whiteToMove == humanIsWhite
    }

    private func refresh() {
        var m = BoardModel()
        if let s = displayState {
            m.board = FEN.board(s.fen)
            m.whiteToMove = s.whiteToMove
            m.legal = interactiveNow ? s.legal : []
            m.checkedKing = s.check ? FEN.king(of: s.whiteToMove, on: m.board) : nil
            m.lastFrom = s.last.flatMap { Square($0.from) }
            m.lastTo = s.last.flatMap { Square($0.to) }
            m.interactive = interactiveNow
            m.hintFrom = hintFrom
            m.hintTo = hintTo
            m.banner = bannerText()
            if let ply = reviewPly, let live = state {
                m.reviewNote = "Viewing position after \(ply) of \(live.ply) plies"
            }
        }
        boardView.model = m

        let shown = reviewPly ?? (state?.ply ?? 0)
        panel.showState(state, shownPly: shown, reviewing: reviewPly != nil, status: statusText())
        panel.setButtons(canUndo: (state?.ply ?? 0) > 0,
                         canHint: hasEngine && interactiveNow,
                         canResign: !bothSides && !resigned && !(state?.isOver ?? true),
                         canStart: true)
    }

    private func statusText() -> String {
        if reviewPly != nil { return "Reviewing an earlier position" }
        if let banner = bannerText() { return banner }
        if thinking { return "Engine is thinking..." }
        guard let s = state else { return "" }
        let side = s.whiteToMove ? "White" : "Black"
        if bothSides { return "\(side) to move" }
        return s.whiteToMove == humanIsWhite ? "Your move (\(side))" : "\(side) to move"
    }

    private func bannerText() -> String? {
        if reviewPly != nil { return nil }
        if resigned {
            return humanIsWhite ? "You resigned - Black wins" : "You resigned - White wins"
        }
        guard let s = state, s.isOver else { return nil }
        let reason = s.reason.isEmpty ? "game over" : s.reason
        if s.result == 3 { return "Draw by \(reason)" }
        let whiteWon = (s.result == 1)
        let who: String
        if bothSides {
            who = whiteWon ? "White wins" : "Black wins"
        } else {
            who = (whiteWon == humanIsWhite) ? "you win" : "engine wins"
        }
        if reason == "checkmate" { return "Checkmate - \(who)" }
        return "\(reason.prefix(1).uppercased())\(reason.dropFirst()) - \(who)"
    }

    // MARK: - Menu

    @objc func menuNewGame(_ sender: Any?) { startNewGame() }
    @objc func menuUndo(_ sender: Any?) { undoMove() }
    @objc func menuFlip(_ sender: Any?) { flipBoard() }
    @objc func menuHint(_ sender: Any?) { requestHint() }
    @objc func menuResign(_ sender: Any?) { resignGame() }

    func validateMenuItem(_ item: NSMenuItem) -> Bool {
        switch item.action {
        case #selector(menuUndo(_:)):   return (state?.ply ?? 0) > 0
        case #selector(menuHint(_:)):   return hasEngine && interactiveNow
        case #selector(menuResign(_:)): return !bothSides && !resigned && !(state?.isOver ?? true)
        default: return true
        }
    }
}
