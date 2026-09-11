//
//  Widgets.swift -- the custom-drawn pieces of the side panel.
//

import AppKit

// MARK: - Section container

/// A titled, rounded box so the panel reads as a shipped app rather than a
/// stack of loose controls.
final class PanelSection: NSView {

    let content = NSView()
    private let titleLabel = NSTextField(labelWithString: "")

    init(title: String) {
        super.init(frame: .zero)
        wantsLayer = true
        layer?.cornerRadius = 8
        layer?.borderWidth = 1
        translatesAutoresizingMaskIntoConstraints = false

        titleLabel.stringValue = title.uppercased()
        titleLabel.font = NSFont.systemFont(ofSize: 9.5, weight: .semibold)
        titleLabel.textColor = .tertiaryLabelColor
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        content.translatesAutoresizingMaskIntoConstraints = false
        addSubview(titleLabel)
        addSubview(content)

        NSLayoutConstraint.activate([
            titleLabel.topAnchor.constraint(equalTo: topAnchor, constant: 7),
            titleLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            titleLabel.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor, constant: -10),
            content.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 6),
            content.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            content.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10),
            content.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -9)
        ])
        applyColours()
    }

    required init?(coder: NSCoder) { nil }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        applyColours()
    }

    private func applyColours() {
        effectiveAppearance.performAsCurrentDrawingAppearance {
            layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
            layer?.borderColor = NSColor.separatorColor.withAlphaComponent(0.6).cgColor
        }
    }
}

// MARK: - Evaluation bar

/// White's advantage as a horizontal bar.  `value` is the engine's `value`
/// field, already in White's frame of reference, in [-1, +1].
final class EvalBarView: NSView {

    var value: Double = 0 { didSet { needsDisplay = true } }
    var enabled = true { didSet { needsDisplay = true } }

    override var isFlipped: Bool { true }
    override var intrinsicContentSize: NSSize { NSSize(width: NSView.noIntrinsicMetric, height: 20) }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let r = bounds.insetBy(dx: 0, dy: 2)
        let radius = r.height / 2
        let clip = NSBezierPath(roundedRect: r, xRadius: radius, yRadius: radius)

        ctx.saveGState()
        clip.addClip()

        // Black's half is the background, White's half is drawn over it.
        NSColor(srgbRed: 0.19, green: 0.19, blue: 0.21, alpha: enabled ? 1 : 0.35).setFill()
        r.fill()

        let clamped = max(-1, min(1, value))
        let w = r.width * CGFloat((clamped + 1) / 2)
        NSColor(srgbRed: 0.95, green: 0.95, blue: 0.94, alpha: enabled ? 1 : 0.35).setFill()
        NSRect(x: r.minX, y: r.minY, width: w, height: r.height).fill()
        ctx.restoreGState()

        // Centre tick.
        NSColor.systemGray.withAlphaComponent(0.7).setFill()
        NSRect(x: r.midX - 0.5, y: r.minY, width: 1, height: r.height).fill()

        NSColor.separatorColor.setStroke()
        clip.lineWidth = 1
        clip.stroke()
    }
}

// MARK: - Top policy moves

final class TopMovesView: NSView {

    var moves: [TopMove] = [] { didSet { needsDisplay = true } }

    override var isFlipped: Bool { true }
    override var intrinsicContentSize: NSSize { NSSize(width: NSView.noIntrinsicMetric, height: 54) }

    override func draw(_ dirtyRect: NSRect) {
        let rowH: CGFloat = 18
        let font = NSFont.monospacedSystemFont(ofSize: 11, weight: .medium)
        let small = NSFont.monospacedDigitSystemFont(ofSize: 10, weight: .regular)

        if moves.isEmpty {
            NSAttributedString(string: "no analysis yet",
                               attributes: [.font: NSFont.systemFont(ofSize: 11),
                                            .foregroundColor: NSColor.tertiaryLabelColor])
                .draw(at: CGPoint(x: 0, y: 2))
            return
        }

        for (i, m) in moves.prefix(3).enumerated() {
            let y = CGFloat(i) * rowH
            NSAttributedString(string: m.san,
                               attributes: [.font: font, .foregroundColor: NSColor.labelColor])
                .draw(at: CGPoint(x: 0, y: y + 2))

            let pct = String(format: "%3.0f%%", m.prob * 100)
            let pctStr = NSAttributedString(string: pct,
                                            attributes: [.font: small,
                                                         .foregroundColor: NSColor.secondaryLabelColor])
            let pctW = pctStr.size().width
            pctStr.draw(at: CGPoint(x: bounds.maxX - pctW, y: y + 3))

            let barX: CGFloat = 52
            let barW = max(10, bounds.maxX - pctW - 8 - barX)
            let track = NSRect(x: barX, y: y + 5, width: barW, height: 8)
            NSColor.quaternaryLabelColor.setFill()
            NSBezierPath(roundedRect: track, xRadius: 4, yRadius: 4).fill()
            let fillW = max(2, barW * CGFloat(max(0, min(1, m.prob))))
            NSColor.controlAccentColor.setFill()
            NSBezierPath(roundedRect: NSRect(x: barX, y: y + 5, width: fillW, height: 8),
                         xRadius: 4, yRadius: 4).fill()
        }
    }
}

// MARK: - Captured pieces tray

final class CapturedTrayView: NSView {

    /// The pieces this side has captured, drawn in the opponent's colour.
    var pieces: [Piece] = [] { didSet { needsDisplay = true } }
    var caption: String = "" { didSet { needsDisplay = true } }

    override var isFlipped: Bool { false }
    override var intrinsicContentSize: NSSize { NSSize(width: NSView.noIntrinsicMetric, height: 22) }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let label = NSAttributedString(string: caption,
                                       attributes: [.font: NSFont.systemFont(ofSize: 10, weight: .medium),
                                                    .foregroundColor: NSColor.secondaryLabelColor])
        label.draw(at: CGPoint(x: 0, y: bounds.midY - label.size().height / 2))

        let size = min(bounds.height, 20)
        var x: CGFloat = 44
        let step = size * 0.62
        for piece in pieces {
            if x + size > bounds.maxX { break }
            PieceRenderer.draw(piece, in: CGRect(x: x, y: bounds.midY - size / 2,
                                                 width: size, height: size),
                               context: ctx, shadow: false)
            x += step
        }
    }
}

// MARK: - Move list

/// Two columns of SAN with the current ply highlighted.  Clicking a move asks
/// the controller to show the position after it.
final class MoveListView: NSView {

    /// Full SAN history, one entry per ply.
    var sans: [String] = [] { didSet { relayout() } }
    /// Number of plies currently shown on the board (0 == start position).
    var shownPly: Int = 0 { didSet { needsDisplay = true } }
    /// Called with a ply count (0 == start position).
    var onSelect: ((Int) -> Void)?

    private let rowHeight: CGFloat = 19
    private let numberWidth: CGFloat = 34

    override var isFlipped: Bool { true }

    override func draw(_ dirtyRect: NSRect) {
        let font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        let bold = NSFont.monospacedSystemFont(ofSize: 12, weight: .bold)
        let numFont = NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .regular)

        if sans.isEmpty {
            NSAttributedString(string: "  no moves yet",
                               attributes: [.font: NSFont.systemFont(ofSize: 11),
                                            .foregroundColor: NSColor.tertiaryLabelColor])
                .draw(at: CGPoint(x: 4, y: 6))
            return
        }

        let colWidth = (bounds.width - numberWidth - 8) / 2
        let rows = (sans.count + 1) / 2
        for row in 0 ..< rows {
            let y = CGFloat(row) * rowHeight
            guard y + rowHeight >= dirtyRect.minY, y <= dirtyRect.maxY else { continue }

            if row % 2 == 1 {
                NSColor.quaternaryLabelColor.withAlphaComponent(0.35).setFill()
                NSRect(x: 0, y: y, width: bounds.width, height: rowHeight).fill()
            }
            NSAttributedString(string: String(format: "%3d.", row + 1),
                               attributes: [.font: numFont,
                                            .foregroundColor: NSColor.tertiaryLabelColor])
                .draw(at: CGPoint(x: 2, y: y + 3))

            for half in 0 ..< 2 {
                let ply = row * 2 + half
                guard ply < sans.count else { break }
                let x = numberWidth + CGFloat(half) * colWidth
                let cell = NSRect(x: x, y: y + 1, width: colWidth - 2, height: rowHeight - 2)
                let current = (ply + 1) == shownPly
                if current {
                    NSColor.controlAccentColor.setFill()
                    NSBezierPath(roundedRect: cell, xRadius: 4, yRadius: 4).fill()
                }
                NSAttributedString(string: sans[ply],
                                   attributes: [.font: current ? bold : font,
                                                .foregroundColor: current ? NSColor.white
                                                                          : NSColor.labelColor])
                    .draw(at: CGPoint(x: x + 5, y: y + 3))
            }
        }
    }

    /// The document view is pinned to the clip view's width by Auto Layout, so
    /// only the height has to follow the move count.
    override var intrinsicContentSize: NSSize {
        let rows = max(1, (sans.count + 1) / 2)
        return NSSize(width: NSView.noIntrinsicMetric, height: CGFloat(rows) * rowHeight + 4)
    }

    private func relayout() {
        invalidateIntrinsicContentSize()
        needsDisplay = true
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        needsDisplay = true
    }

    override func mouseDown(with event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        let row = Int(p.y / rowHeight)
        guard row >= 0 else { return }
        let colWidth = (bounds.width - numberWidth - 8) / 2
        guard p.x >= numberWidth, colWidth > 0 else { return }
        let half = min(1, Int((p.x - numberWidth) / colWidth))
        let ply = row * 2 + half
        guard ply < sans.count else { return }
        onSelect?(ply + 1)
    }

    /// Scrolls so the highlighted move is visible.
    func revealCurrent() {
        guard shownPly > 0 else { return }
        let row = (shownPly - 1) / 2
        let r = NSRect(x: 0, y: CGFloat(row) * rowHeight - rowHeight,
                       width: bounds.width, height: rowHeight * 3)
        scrollToVisible(r)
    }
}
