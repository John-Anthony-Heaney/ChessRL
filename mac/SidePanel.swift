//
//  SidePanel.swift -- everything to the right of the board.
//

import AppKit

protocol SidePanelDelegate: AnyObject {
    func sidePanelNewGame(_ panel: SidePanel)
    func sidePanelUndo(_ panel: SidePanel)
    func sidePanelFlip(_ panel: SidePanel)
    func sidePanelHint(_ panel: SidePanel)
    func sidePanelResign(_ panel: SidePanel)
    func sidePanel(_ panel: SidePanel, scrubTo ply: Int)
    func sidePanelReturnToLive(_ panel: SidePanel)
}

/// The engine's last analysis, plus whose move it was, so the score can be put
/// into White's frame of reference for display.
struct Analysis {
    let report: EngineReport
    let whiteToMove: Bool
    let caption: String

    var scoreForWhite: Int { whiteToMove ? report.score : -report.score }
}

final class SidePanel: NSView {

    weak var delegate: SidePanelDelegate?

    // Levels 1...8 -> search depth and a movetime that keeps the top level snappy.
    static let levels: [(name: String, depth: Int32, movetime: Int32)] = [
        ("1 - Beginner",     1,  200),
        ("2 - Casual",       2,  300),
        ("3 - Novice",       3,  450),
        ("4 - Club",         4,  700),
        ("5 - Strong",       5, 1000),
        ("6 - Expert",       6, 1500),
        ("7 - Master",       7, 2200),
        ("8 - Maximum",      8, 3500)
    ]

    // Controls the controller reads.
    private let colourControl = NSSegmentedControl(labels: ["White", "Black", "Random"],
                                                   trackingMode: .selectOne, target: nil, action: nil)
    private let levelPopUp = NSPopUpButton(frame: .zero, pullsDown: false)

    var chosenColour: Int { colourControl.selectedSegment }          // 0 W, 1 B, 2 random
    var chosenLevel: Int { max(0, levelPopUp.indexOfSelectedItem) }  // index into `levels`

    // Readouts.
    private let modelTitle = NSTextField(labelWithString: "")
    private let modelDetail = NSTextField(labelWithString: "")
    private let evalBar = EvalBarView()
    private let scoreLabel = NSTextField(labelWithString: "0.00")
    private let statsLabel = NSTextField(labelWithString: "")
    private let engineCaption = NSTextField(labelWithString: "")
    private let spinner = NSProgressIndicator()
    private let topMoves = TopMovesView()
    private let whiteTray = CapturedTrayView()
    private let blackTray = CapturedTrayView()
    private let materialLabel = NSTextField(labelWithString: "")
    private let moveList = MoveListView()
    private let moveScroll = NSScrollView()
    private let liveButton = NSButton(title: "Return to live position", target: nil, action: nil)
    private let statusLabel = NSTextField(labelWithString: "")

    private let undoButton = NSButton(title: "Undo", target: nil, action: nil)
    private let hintButton = NSButton(title: "Hint", target: nil, action: nil)
    private let resignButton = NSButton(title: "Resign", target: nil, action: nil)
    private let newGameButton = NSButton(title: "New Game", target: nil, action: nil)
    private let flipButton = NSButton(title: "Flip", target: nil, action: nil)

    // MARK: - Build

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        translatesAutoresizingMaskIntoConstraints = false
        build()
    }

    required init?(coder: NSCoder) { nil }

    private static func mono(_ size: CGFloat, _ weight: NSFont.Weight = .regular) -> NSFont {
        NSFont.monospacedDigitSystemFont(ofSize: size, weight: weight)
    }

    private func build() {
        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.distribution = .fill
        stack.spacing = 10
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 14),
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -14),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -14)
        ])

        stack.addArrangedSubview(buildModelSection())
        stack.addArrangedSubview(buildNewGameSection())
        stack.addArrangedSubview(buildEngineSection())
        stack.addArrangedSubview(buildTopMovesSection())
        stack.addArrangedSubview(buildMaterialSection())
        let moves = buildMovesSection()
        stack.addArrangedSubview(moves)
        stack.addArrangedSubview(buildButtonRow())

        for view in stack.arrangedSubviews {
            view.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        }
        // Only the move list absorbs slack.
        moves.setContentHuggingPriority(.init(1), for: .vertical)
        moves.setContentCompressionResistancePriority(.init(300), for: .vertical)
    }

    private func buildModelSection() -> PanelSection {
        let box = PanelSection(title: "Champion")
        modelTitle.font = NSFont.systemFont(ofSize: 12, weight: .semibold)
        modelTitle.lineBreakMode = .byTruncatingMiddle
        modelDetail.font = SidePanel.mono(11)
        modelDetail.textColor = .secondaryLabelColor
        modelDetail.lineBreakMode = .byTruncatingTail

        let v = NSStackView(views: [modelTitle, modelDetail])
        v.orientation = .vertical
        v.alignment = .leading
        v.spacing = 2
        v.translatesAutoresizingMaskIntoConstraints = false
        box.content.addSubview(v)
        pin(v, to: box.content)
        return box
    }

    private func buildNewGameSection() -> PanelSection {
        let box = PanelSection(title: "New game")
        colourControl.selectedSegment = 0
        colourControl.segmentDistribution = .fillEqually
        colourControl.translatesAutoresizingMaskIntoConstraints = false

        for level in SidePanel.levels { levelPopUp.addItem(withTitle: level.name) }
        levelPopUp.selectItem(at: 3)
        levelPopUp.translatesAutoresizingMaskIntoConstraints = false

        let colourRow = labelled("You play", colourControl)
        let levelRow = labelled("Strength", levelPopUp)
        let v = NSStackView(views: [colourRow, levelRow])
        v.orientation = .vertical
        v.alignment = .leading
        v.spacing = 6
        v.translatesAutoresizingMaskIntoConstraints = false
        box.content.addSubview(v)
        pin(v, to: box.content)
        colourRow.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        levelRow.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        return box
    }

    private func buildEngineSection() -> PanelSection {
        let box = PanelSection(title: "Engine")
        scoreLabel.font = SidePanel.mono(20, .semibold)
        statsLabel.font = SidePanel.mono(10.5)
        statsLabel.textColor = .secondaryLabelColor
        engineCaption.font = NSFont.systemFont(ofSize: 10.5)
        engineCaption.textColor = .tertiaryLabelColor
        engineCaption.lineBreakMode = .byTruncatingTail

        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.isDisplayedWhenStopped = false
        spinner.translatesAutoresizingMaskIntoConstraints = false

        let spacer = NSView()
        spacer.translatesAutoresizingMaskIntoConstraints = false
        spacer.setContentHuggingPriority(.init(1), for: .horizontal)
        spacer.setContentCompressionResistancePriority(.init(1), for: .horizontal)
        let head = NSStackView(views: [scoreLabel, spacer, spinner])
        head.orientation = .horizontal
        head.alignment = .centerY
        head.distribution = .fill
        head.spacing = 6

        evalBar.translatesAutoresizingMaskIntoConstraints = false
        let v = NSStackView(views: [head, evalBar, statsLabel, engineCaption])
        v.orientation = .vertical
        v.alignment = .leading
        v.spacing = 5
        v.translatesAutoresizingMaskIntoConstraints = false
        box.content.addSubview(v)
        pin(v, to: box.content)
        head.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        evalBar.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        evalBar.heightAnchor.constraint(equalToConstant: 20).isActive = true
        statsLabel.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        engineCaption.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        return box
    }

    private func buildTopMovesSection() -> PanelSection {
        let box = PanelSection(title: "Policy - top moves")
        topMoves.translatesAutoresizingMaskIntoConstraints = false
        box.content.addSubview(topMoves)
        pin(topMoves, to: box.content)
        topMoves.heightAnchor.constraint(equalToConstant: 54).isActive = true
        return box
    }

    private func buildMaterialSection() -> PanelSection {
        let box = PanelSection(title: "Material")
        whiteTray.caption = "White"
        blackTray.caption = "Black"
        materialLabel.font = SidePanel.mono(11, .medium)
        materialLabel.textColor = .secondaryLabelColor

        whiteTray.translatesAutoresizingMaskIntoConstraints = false
        blackTray.translatesAutoresizingMaskIntoConstraints = false
        let v = NSStackView(views: [whiteTray, blackTray, materialLabel])
        v.orientation = .vertical
        v.alignment = .leading
        v.spacing = 3
        v.translatesAutoresizingMaskIntoConstraints = false
        box.content.addSubview(v)
        pin(v, to: box.content)
        whiteTray.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        blackTray.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        whiteTray.heightAnchor.constraint(equalToConstant: 22).isActive = true
        blackTray.heightAnchor.constraint(equalToConstant: 22).isActive = true
        return box
    }

    private func buildMovesSection() -> PanelSection {
        let box = PanelSection(title: "Moves")
        moveScroll.hasVerticalScroller = true
        moveScroll.autohidesScrollers = true
        moveScroll.drawsBackground = false
        moveScroll.borderType = .noBorder
        moveScroll.translatesAutoresizingMaskIntoConstraints = false
        moveList.translatesAutoresizingMaskIntoConstraints = false
        moveScroll.documentView = moveList
        NSLayoutConstraint.activate([
            moveList.leadingAnchor.constraint(equalTo: moveScroll.contentView.leadingAnchor),
            moveList.trailingAnchor.constraint(equalTo: moveScroll.contentView.trailingAnchor),
            moveList.topAnchor.constraint(equalTo: moveScroll.contentView.topAnchor)
        ])
        moveList.onSelect = { [weak self] ply in
            guard let self = self else { return }
            self.delegate?.sidePanel(self, scrubTo: ply)
        }

        liveButton.bezelStyle = .rounded
        liveButton.controlSize = .small
        liveButton.font = NSFont.systemFont(ofSize: 11)
        liveButton.target = self
        liveButton.action = #selector(returnToLive)
        liveButton.isHidden = true
        liveButton.translatesAutoresizingMaskIntoConstraints = false

        statusLabel.font = NSFont.systemFont(ofSize: 11, weight: .medium)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.lineBreakMode = .byTruncatingTail

        let v = NSStackView(views: [moveScroll, liveButton, statusLabel])
        v.orientation = .vertical
        v.alignment = .leading
        v.spacing = 5
        v.translatesAutoresizingMaskIntoConstraints = false
        box.content.addSubview(v)
        pin(v, to: box.content)
        moveScroll.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        moveScroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 64).isActive = true
        liveButton.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        statusLabel.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        return box
    }

    private func buildButtonRow() -> NSView {
        let buttons = [newGameButton, undoButton, flipButton, hintButton, resignButton]
        let actions: [Selector] = [#selector(newGame), #selector(undo), #selector(flip),
                                   #selector(hint), #selector(resign)]
        for (b, a) in zip(buttons, actions) {
            b.bezelStyle = .rounded
            b.target = self
            b.action = a
            b.translatesAutoresizingMaskIntoConstraints = false
        }
        let row1 = NSStackView(views: [newGameButton, undoButton, flipButton])
        let row2 = NSStackView(views: [hintButton, resignButton])
        for row in [row1, row2] {
            row.orientation = .horizontal
            row.distribution = .fillEqually
            row.spacing = 6
        }
        let v = NSStackView(views: [row1, row2])
        v.orientation = .vertical
        v.alignment = .leading
        v.spacing = 6
        v.translatesAutoresizingMaskIntoConstraints = false
        row1.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        row2.widthAnchor.constraint(equalTo: v.widthAnchor).isActive = true
        return v
    }

    private func labelled(_ text: String, _ control: NSView) -> NSView {
        let label = NSTextField(labelWithString: text)
        label.font = NSFont.systemFont(ofSize: 11)
        label.textColor = .secondaryLabelColor
        label.setContentHuggingPriority(.required, for: .horizontal)
        let row = NSStackView(views: [label, control])
        row.orientation = .horizontal
        row.alignment = .centerY
        row.spacing = 8
        label.widthAnchor.constraint(equalToConstant: 58).isActive = true
        return row
    }

    private func pin(_ view: NSView, to container: NSView) {
        NSLayoutConstraint.activate([
            view.topAnchor.constraint(equalTo: container.topAnchor),
            view.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            view.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            view.bottomAnchor.constraint(equalTo: container.bottomAnchor)
        ])
    }

    // MARK: - Actions

    @objc private func newGame() { delegate?.sidePanelNewGame(self) }
    @objc private func undo() { delegate?.sidePanelUndo(self) }
    @objc private func flip() { delegate?.sidePanelFlip(self) }
    @objc private func hint() { delegate?.sidePanelHint(self) }
    @objc private func resign() { delegate?.sidePanelResign(self) }
    @objc private func returnToLive() { delegate?.sidePanelReturnToLive(self) }

    // MARK: - Updates

    func showSearchingForModel() {
        modelTitle.stringValue = "Looking for a model..."
        modelDetail.stringValue = "newest runs/*/best.crl"
        evalBar.enabled = false
    }

    func showModel(path: String?, info: ModelInfo?) {
        guard let path = path else {
            modelTitle.stringValue = "No model found"
            modelDetail.stringValue = "Playing without an engine - you may move both sides."
            modelDetail.toolTip = nil
            evalBar.enabled = false
            return
        }
        modelTitle.stringValue = (path as NSString).lastPathComponent
        modelTitle.toolTip = path
        if let info = info, let champ = info.champion {
            modelDetail.stringValue = String(format: "gen %d   agent %d of %d   Elo %.0f",
                                             info.generation, champ.i, info.n_agents, champ.elo)
        } else {
            modelDetail.stringValue = "loaded"
        }
        modelDetail.toolTip = path
        evalBar.enabled = true
    }

    func showThinking(_ on: Bool) {
        if on { spinner.startAnimation(nil) } else { spinner.stopAnimation(nil) }
    }

    func showAnalysis(_ analysis: Analysis?) {
        guard let a = analysis else {
            scoreLabel.stringValue = "0.00"
            scoreLabel.textColor = .tertiaryLabelColor
            statsLabel.stringValue = "depth -   nodes -   time -"
            engineCaption.stringValue = ""
            evalBar.value = 0
            topMoves.moves = []
            return
        }
        let pawns = Double(a.scoreForWhite) / 100.0
        scoreLabel.stringValue = String(format: "%+.2f", pawns)
        scoreLabel.textColor = abs(pawns) < 0.2 ? .secondaryLabelColor : .labelColor
        statsLabel.stringValue = String(format: "depth %d   %@ nodes   %d ms   %.0f knps",
                                        a.report.depth, grouped(a.report.nodes),
                                        a.report.ms, a.report.knps)
        engineCaption.stringValue = a.caption
        evalBar.value = a.report.value
        topMoves.moves = a.report.top
    }

    private func grouped(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    func showState(_ state: GameState?, shownPly: Int, reviewing: Bool, status: String) {
        guard let state = state else { return }
        moveList.sans = state.historySAN
        moveList.shownPly = shownPly
        if !reviewing { moveList.revealCurrent() }
        liveButton.isHidden = !reviewing
        statusLabel.stringValue = status

        // A tray shows what that side has captured: its opponent's losses.
        whiteTray.pieces = state.captured.black.compactMap { letter in
            letter.first.flatMap { Piece(fen: $0) }
        }
        blackTray.pieces = state.captured.white.compactMap { letter in
            letter.first.flatMap { Piece(fen: Character($0.uppercased())) }
        }
        let diff = state.material.white - state.material.black
        if diff == 0 {
            materialLabel.stringValue = "material level"
        } else {
            materialLabel.stringValue = String(format: "%@ +%d",
                                               diff > 0 ? "White" : "Black", abs(diff))
        }
    }

    func setButtons(canUndo: Bool, canHint: Bool, canResign: Bool, canStart: Bool) {
        undoButton.isEnabled = canUndo
        hintButton.isEnabled = canHint
        resignButton.isEnabled = canResign
        newGameButton.isEnabled = canStart
    }
}
