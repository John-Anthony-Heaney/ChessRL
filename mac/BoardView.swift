//
//  BoardView.swift -- the chessboard, drawn with Core Graphics.
//
//  Every playable move comes from `model.legal`, which is exactly the list the C
//  engine returned for the current position.  No legality is computed here.
//

import AppKit

// MARK: - What the board is showing

struct BoardModel {
    var board: [Piece?] = [Piece?](repeating: nil, count: 64)
    var legal: [String] = []
    var whiteToMove = true
    var checkedKing: Square?
    var lastFrom: Square?
    var lastTo: Square?
    var hintFrom: Square?
    var hintTo: Square?
    var interactive = false          // may the human touch the pieces right now
    var banner: String?              // game over text
    var reviewNote: String?          // "Viewing move 12 of 40"
}

protocol BoardViewDelegate: AnyObject {
    /// The user completed a legal move.  `uci` is guaranteed to be a member of
    /// the `legal` list the controller last handed to the view.
    func boardView(_ view: BoardView, didPlay uci: String)
}

// MARK: - The view

final class BoardView: NSView {

    weak var delegate: BoardViewDelegate?

    var model = BoardModel() {
        didSet {
            selected = nil
            promotion = nil
            dragging = nil
            needsDisplay = true
        }
    }

    /// Set without clearing the interaction state (used for cheap repaints).
    func refreshOverlays() { needsDisplay = true }

    var boardFlipped = false {
        didSet {
            selected = nil
            promotion = nil
            dragging = nil
            needsDisplay = true
        }
    }

    // ------------------------------------------------------------ interaction

    private var selected: Square?

    private struct Drag {
        let origin: Square
        let piece: Piece
        var point: CGPoint
        var live: Bool           // has it moved far enough to count as a drag
        let downPoint: CGPoint
        let wasSelected: Bool    // clicking a selected piece again deselects it
    }
    private var dragging: Drag?

    private struct Promotion {
        let from: Square
        let to: Square
        let options: [Character]
    }
    private var promotion: Promotion?

    private struct Sliding {
        let piece: Piece
        let hide: Square         // square whose piece is suppressed while sliding
        let from: CGPoint
        let to: CGPoint
        let start: CFTimeInterval
        let duration: CFTimeInterval
    }
    private var sliding: Sliding?
    private var slideTimer: Timer?

    private var trackingArea: NSTrackingArea?

    // ------------------------------------------------------------------ setup

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
    }

    required init?(coder: NSCoder) { nil }

    override var isFlipped: Bool { false }          // y up, like the ranks
    override var isOpaque: Bool { true }
    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let old = trackingArea { removeTrackingArea(old) }
        let area = NSTrackingArea(rect: bounds,
                                  options: [.mouseMoved, .mouseEnteredAndExited,
                                            .activeInKeyWindow, .inVisibleRect],
                                  owner: self, userInfo: nil)
        addTrackingArea(area)
        trackingArea = area
    }

    // -------------------------------------------------------------- geometry

    /// Board geometry for the current bounds: a centred square, whole pixels.
    var geometry: BoardGeometry {
        let pad: CGFloat = 6
        let side = max(64, min(bounds.width, bounds.height) - pad * 2)
        let sq = (side / 8).rounded(.down)
        let board = sq * 8
        let ox = (bounds.width - board).rounded(.down) / 2
        let oy = (bounds.height - board).rounded(.down) / 2
        return BoardGeometry(origin: CGPoint(x: ox, y: oy), squareSize: sq, flipped: boardFlipped)
    }

    // --------------------------------------------------------------- palette

    private static let lightSquare = NSColor(name: nil) { appearance in
        appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            ? NSColor(srgbRed: 0.729, green: 0.706, blue: 0.647, alpha: 1)
            : NSColor(srgbRed: 0.929, green: 0.914, blue: 0.855, alpha: 1)
    }
    private static let darkSquare = NSColor(name: nil) { appearance in
        appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            ? NSColor(srgbRed: 0.353, green: 0.443, blue: 0.353, alpha: 1)
            : NSColor(srgbRed: 0.463, green: 0.588, blue: 0.337, alpha: 1)
    }
    private static let lastMoveTint = NSColor(srgbRed: 0.98, green: 0.83, blue: 0.24, alpha: 0.42)
    private static let selectTint    = NSColor(srgbRed: 0.30, green: 0.66, blue: 0.98, alpha: 0.42)
    private static let hintTint      = NSColor(srgbRed: 0.45, green: 0.85, blue: 0.55, alpha: 0.55)

    // ---------------------------------------------------------------- drawing

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let g = geometry

        NSColor.underPageBackgroundColor.setFill()
        bounds.fill()

        let boardRect = CGRect(x: g.origin.x, y: g.origin.y,
                               width: g.squareSize * 8, height: g.squareSize * 8)

        // Frame and drop shadow under the board.
        ctx.saveGState()
        ctx.setShadow(offset: CGSize(width: 0, height: -2), blur: 10,
                      color: NSColor.black.withAlphaComponent(0.28).cgColor)
        NSColor.black.withAlphaComponent(0.85).setFill()
        ctx.fill(boardRect.insetBy(dx: -1.5, dy: -1.5))
        ctx.restoreGState()

        drawSquares(ctx, g)
        drawSquareTints(ctx, g)
        drawQuietTargets(ctx, g)
        drawPieces(ctx, g)
        drawCaptureTargets(ctx, g)
        drawCoordinates(ctx, g)
        drawSlidingPiece(ctx, g)
        drawDraggedPiece(ctx, g)
        drawPromotionOverlay(ctx, g)
        drawReviewNote(ctx, g, boardRect)
        drawBanner(ctx, boardRect)
    }

    private func drawSquares(_ ctx: CGContext, _ g: BoardGeometry) {
        for sq in Square.all {
            (sq.isLight ? BoardView.lightSquare : BoardView.darkSquare).setFill()
            ctx.fill(g.rect(of: sq))
        }
    }

    private func drawSquareTints(_ ctx: CGContext, _ g: BoardGeometry) {
        for sq in [model.lastFrom, model.lastTo].compactMap({ $0 }) {
            BoardView.lastMoveTint.setFill()
            ctx.fill(g.rect(of: sq))
        }
        if let king = model.checkedKing {
            let r = g.rect(of: king)
            ctx.saveGState()
            let colors = [NSColor(srgbRed: 1, green: 0.20, blue: 0.16, alpha: 0.92).cgColor,
                          NSColor(srgbRed: 0.85, green: 0.0, blue: 0.0, alpha: 0.0).cgColor]
            if let grad = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(),
                                     colors: colors as CFArray, locations: [0, 1]) {
                ctx.addRect(r)
                ctx.clip()
                ctx.drawRadialGradient(grad,
                                       startCenter: CGPoint(x: r.midX, y: r.midY), startRadius: 0,
                                       endCenter: CGPoint(x: r.midX, y: r.midY),
                                       endRadius: r.width * 0.62,
                                       options: [])
            }
            ctx.restoreGState()
        }
        if let sel = selected {
            BoardView.selectTint.setFill()
            ctx.fill(g.rect(of: sel))
        }
        if let from = model.hintFrom, let to = model.hintTo {
            ctx.setStrokeColor(BoardView.hintTint.cgColor)
            ctx.setLineWidth(max(3, g.squareSize * 0.06))
            for sq in [from, to] {
                ctx.stroke(g.rect(of: sq).insetBy(dx: g.squareSize * 0.05,
                                                  dy: g.squareSize * 0.05))
            }
            // A line from origin to destination so the suggestion reads at a glance.
            ctx.setLineCap(.round)
            ctx.setLineWidth(max(3, g.squareSize * 0.075))
            ctx.setStrokeColor(BoardView.hintTint.withAlphaComponent(0.75).cgColor)
            ctx.move(to: g.center(of: from))
            ctx.addLine(to: g.center(of: to))
            ctx.strokePath()
        }
    }

    /// The `to` squares of every legal move from the selected square.
    private var activeTargets: [Square] {
        guard let sel = selected, model.interactive else { return [] }
        var out: [Square] = []
        var seen = Set<Int>()
        for uci in model.legal where uci.hasPrefix(sel.name) {
            let to = String(uci.dropFirst(2).prefix(2))
            if let sq = Square(to), seen.insert(sq.index).inserted { out.append(sq) }
        }
        return out
    }

    private func drawQuietTargets(_ ctx: CGContext, _ g: BoardGeometry) {
        ctx.setFillColor(NSColor.black.withAlphaComponent(0.20).cgColor)
        for sq in activeTargets where model.board[sq.index] == nil {
            let r = g.rect(of: sq)
            let d = g.squareSize * 0.28
            ctx.fillEllipse(in: CGRect(x: r.midX - d / 2, y: r.midY - d / 2, width: d, height: d))
        }
    }

    private func drawCaptureTargets(_ ctx: CGContext, _ g: BoardGeometry) {
        ctx.setStrokeColor(NSColor.black.withAlphaComponent(0.22).cgColor)
        let w = g.squareSize * 0.09
        ctx.setLineWidth(w)
        for sq in activeTargets where model.board[sq.index] != nil {
            let r = g.rect(of: sq).insetBy(dx: w / 2 + 1, dy: w / 2 + 1)
            ctx.strokeEllipse(in: r)
        }
    }

    private func drawPieces(_ ctx: CGContext, _ g: BoardGeometry) {
        for sq in Square.all {
            guard let piece = model.board[sq.index] else { continue }
            if let d = dragging, d.live, d.origin == sq { continue }
            if let s = sliding, s.hide == sq { continue }
            PieceRenderer.draw(piece, in: g.rect(of: sq), context: ctx)
        }
        // The origin of a live drag is dimmed rather than empty.
        if let d = dragging, d.live {
            ctx.saveGState()
            ctx.setAlpha(0.24)
            PieceRenderer.draw(d.piece, in: g.rect(of: d.origin), context: ctx, shadow: false)
            ctx.restoreGState()
        }
    }

    private func drawSlidingPiece(_ ctx: CGContext, _ g: BoardGeometry) {
        guard let s = sliding else { return }
        let t = min(1, max(0, (CACurrentMediaTime() - s.start) / s.duration))
        let e = 1 - pow(1 - t, 3)                       // ease-out cubic
        let p = CGPoint(x: s.from.x + (s.to.x - s.from.x) * e,
                        y: s.from.y + (s.to.y - s.from.y) * e)
        let r = CGRect(x: p.x - g.squareSize / 2, y: p.y - g.squareSize / 2,
                       width: g.squareSize, height: g.squareSize)
        PieceRenderer.draw(s.piece, in: r, context: ctx)
    }

    private func drawDraggedPiece(_ ctx: CGContext, _ g: BoardGeometry) {
        guard let d = dragging, d.live else { return }
        let size = g.squareSize * 1.06
        let r = CGRect(x: d.point.x - size / 2, y: d.point.y - size / 2,
                       width: size, height: size)
        PieceRenderer.draw(d.piece, in: r, context: ctx)
    }

    private func drawCoordinates(_ ctx: CGContext, _ g: BoardGeometry) {
        let size = max(8, g.squareSize * 0.20)
        let font = NSFont.systemFont(ofSize: size, weight: .semibold)
        for i in 0 ..< 8 {
            // Rank numbers run up the left-hand column.
            let rankSquare = g.square(at: CGPoint(x: g.origin.x + g.squareSize * 0.5,
                                                  y: g.origin.y + (CGFloat(i) + 0.5) * g.squareSize))
            if let sq = rankSquare {
                let colour = sq.isLight ? BoardView.darkSquare : BoardView.lightSquare
                let s = NSAttributedString(string: "\(sq.rank + 1)",
                                           attributes: [.font: font, .foregroundColor: colour])
                let r = g.rect(of: sq)
                s.draw(at: CGPoint(x: r.minX + size * 0.30, y: r.maxY - size * 1.35))
            }
            // File letters run along the bottom row.
            let fileSquare = g.square(at: CGPoint(x: g.origin.x + (CGFloat(i) + 0.5) * g.squareSize,
                                                  y: g.origin.y + g.squareSize * 0.5))
            if let sq = fileSquare {
                let colour = sq.isLight ? BoardView.darkSquare : BoardView.lightSquare
                let letter = String(Character(UnicodeScalar(UInt8(0x61 + sq.file))))
                let s = NSAttributedString(string: letter,
                                           attributes: [.font: font, .foregroundColor: colour])
                let r = g.rect(of: sq)
                let w = s.size().width
                s.draw(at: CGPoint(x: r.maxX - w - size * 0.30, y: r.minY + size * 0.22))
            }
        }
    }

    private func drawReviewNote(_ ctx: CGContext, _ g: BoardGeometry, _ boardRect: CGRect) {
        guard let note = model.reviewNote else { return }
        ctx.setStrokeColor(NSColor.systemOrange.cgColor)
        ctx.setLineWidth(3)
        ctx.stroke(boardRect.insetBy(dx: 1.5, dy: 1.5))

        // Flush with the top edge so it hides as little of rank 8 as possible.
        let font = NSFont.systemFont(ofSize: 11, weight: .semibold)
        let text = NSAttributedString(string: note,
                                      attributes: [.font: font,
                                                   .foregroundColor: NSColor.white])
        let size = text.size()
        let pill = CGRect(x: boardRect.midX - size.width / 2 - 9,
                          y: boardRect.maxY - size.height - 8,
                          width: size.width + 18, height: size.height + 4)
        NSColor.systemOrange.withAlphaComponent(0.95).setFill()
        NSBezierPath(roundedRect: pill, xRadius: pill.height / 2, yRadius: pill.height / 2).fill()
        text.draw(at: CGPoint(x: pill.minX + 9, y: pill.minY + 2))
    }

    private func drawBanner(_ ctx: CGContext, _ boardRect: CGRect) {
        guard let banner = model.banner else { return }
        let font = NSFont.systemFont(ofSize: max(15, boardRect.width * 0.042), weight: .semibold)
        let text = NSAttributedString(string: banner,
                                      attributes: [.font: font,
                                                   .foregroundColor: NSColor.white])
        let size = text.size()
        let box = CGRect(x: boardRect.midX - size.width / 2 - 22,
                         y: boardRect.midY - size.height / 2 - 14,
                         width: size.width + 44, height: size.height + 28)
        ctx.saveGState()
        ctx.setShadow(offset: CGSize(width: 0, height: -3), blur: 14,
                      color: NSColor.black.withAlphaComponent(0.5).cgColor)
        NSColor(srgbRed: 0.09, green: 0.09, blue: 0.11, alpha: 0.92).setFill()
        NSBezierPath(roundedRect: box, xRadius: 12, yRadius: 12).fill()
        ctx.restoreGState()
        text.draw(at: CGPoint(x: box.minX + 22, y: box.minY + 14))
    }

    // ----------------------------------------------------------- promotion UI

    /// The clickable squares of the promotion picker, in Q R B N order.
    private func promotionRects(_ g: BoardGeometry, _ promo: Promotion) -> [(Character, CGRect)] {
        let baseRow = g.row(of: promo.to)
        let downward = baseRow >= 4          // grow toward the middle of the board
        let col = g.column(of: promo.to)
        var out: [(Character, CGRect)] = []
        for (i, ch) in promo.options.enumerated() {
            let row = downward ? baseRow - i : baseRow + i
            guard row >= 0, row < 8 else { continue }
            out.append((ch, CGRect(x: g.origin.x + CGFloat(col) * g.squareSize,
                                   y: g.origin.y + CGFloat(row) * g.squareSize,
                                   width: g.squareSize, height: g.squareSize)))
        }
        return out
    }

    private func drawPromotionOverlay(_ ctx: CGContext, _ g: BoardGeometry) {
        guard let promo = promotion else { return }
        let rects = promotionRects(g, promo)
        guard let first = rects.first else { return }

        // Dim everything else.
        NSColor.black.withAlphaComponent(0.45).setFill()
        ctx.fill(bounds)

        var union = first.1
        for r in rects { union = union.union(r.1) }
        let panel = union.insetBy(dx: -6, dy: -6)

        ctx.saveGState()
        ctx.setShadow(offset: CGSize(width: 0, height: -3), blur: 12,
                      color: NSColor.black.withAlphaComponent(0.6).cgColor)
        NSColor.windowBackgroundColor.setFill()
        NSBezierPath(roundedRect: panel, xRadius: 10, yRadius: 10).fill()
        ctx.restoreGState()

        let white = model.whiteToMove
        for (ch, rect) in rects {
            NSColor.controlBackgroundColor.setFill()
            NSBezierPath(roundedRect: rect.insetBy(dx: 3, dy: 3), xRadius: 7, yRadius: 7).fill()
            if let kind = PromotionPicker.kind(for: ch) {
                PieceRenderer.draw(Piece(kind: kind, isWhite: white), in: rect, context: ctx)
            }
        }
    }

    enum PromotionPicker {
        static func kind(for ch: Character) -> PieceKind? {
            switch ch {
            case "q": return .queen
            case "r": return .rook
            case "b": return .bishop
            case "n": return .knight
            default:  return nil
            }
        }
    }

    // ------------------------------------------------------------------ input

    private func location(_ event: NSEvent) -> CGPoint {
        convert(event.locationInWindow, from: nil)
    }

    private func myPiece(at sq: Square) -> Piece? {
        guard let piece = model.board[sq.index], piece.isWhite == model.whiteToMove else { return nil }
        return piece
    }

    /// The one and only legality question the view ever asks, and it asks it of
    /// the engine's own list.
    private func canMove(from: Square, to: Square) -> Bool {
        let stem = from.name + to.name
        return model.legal.contains { $0.hasPrefix(stem) }
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        let p = location(event)
        let g = geometry

        // 1. The promotion picker swallows every click while it is up.
        if let promo = promotion {
            for (ch, rect) in promotionRects(g, promo) where rect.contains(p) {
                let uci = promo.from.name + promo.to.name + String(ch)
                promotion = nil
                selected = nil
                needsDisplay = true
                if model.legal.contains(uci) { delegate?.boardView(self, didPlay: uci) }
                return
            }
            promotion = nil                       // clicking away cancels the move
            selected = nil
            needsDisplay = true
            return
        }

        guard model.interactive, let sq = g.square(at: p) else {
            selected = nil
            needsDisplay = true
            return
        }

        // 2. Completing a click-to-move (quiet move or capture).
        if let sel = selected, sel != sq, myPiece(at: sq) == nil, canMove(from: sel, to: sq) {
            attempt(from: sel, to: sq)
            return
        }

        // 3. Picking up one of our own pieces (also arms a drag).
        if let piece = myPiece(at: sq) {
            let again = (selected == sq)
            selected = sq
            dragging = Drag(origin: sq, piece: piece, point: p, live: false,
                            downPoint: p, wasSelected: again)
            needsDisplay = true
            return
        }

        selected = nil
        needsDisplay = true
    }

    override func mouseDragged(with event: NSEvent) {
        guard var d = dragging else { return }
        let p = location(event)
        d.point = p
        if !d.live {
            let dx = p.x - d.downPoint.x, dy = p.y - d.downPoint.y
            if dx * dx + dy * dy > 16 { d.live = true }
        }
        dragging = d
        if d.live { needsDisplay = true }
    }

    override func mouseUp(with event: NSEvent) {
        guard let d = dragging else { return }
        dragging = nil
        guard d.live else {
            // A click, not a drag: clicking the same piece twice puts it down.
            if d.wasSelected { selected = nil }
            needsDisplay = true
            return
        }

        let g = geometry
        let p = location(event)
        if let sq = g.square(at: p), sq != d.origin, canMove(from: d.origin, to: sq) {
            attempt(from: d.origin, to: sq)
            return
        }
        // Illegal drop: snap home.
        snapBack(piece: d.piece, to: d.origin, from: p)
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 {                    // Escape
            if promotion != nil || selected != nil {
                promotion = nil
                selected = nil
                needsDisplay = true
                return
            }
        }
        super.keyDown(with: event)
    }

    override func mouseMoved(with event: NSEvent) {
        let g = geometry
        var hand = false
        if promotion != nil {
            hand = true
        } else if model.interactive, let sq = g.square(at: location(event)) {
            hand = myPiece(at: sq) != nil || (selected.map { canMove(from: $0, to: sq) } ?? false)
        }
        (hand ? NSCursor.pointingHand : NSCursor.arrow).set()
    }

    override func mouseExited(with event: NSEvent) {
        NSCursor.arrow.set()
    }

    /// Plays from/to, opening the promotion picker when the pair has variants.
    private func attempt(from: Square, to: Square) {
        let promos = ["q", "r", "b", "n"].compactMap { suffix -> Character? in
            model.legal.contains(from.name + to.name + suffix) ? Character(suffix) : nil
        }
        if !promos.isEmpty {
            promotion = Promotion(from: from, to: to, options: promos)
            selected = nil
            needsDisplay = true
            return
        }
        let uci = from.name + to.name
        guard model.legal.contains(uci) else {          // cannot happen; refuse anyway
            selected = nil
            needsDisplay = true
            return
        }
        selected = nil
        needsDisplay = true
        delegate?.boardView(self, didPlay: uci)
    }

    // ------------------------------------------------------------- animation

    /// Slides a piece from one square to another; used for the engine's replies.
    func animate(piece: Piece, from: Square, to: Square, duration: CFTimeInterval = 0.12) {
        let g = geometry
        startSlide(Sliding(piece: piece, hide: to,
                           from: g.center(of: from), to: g.center(of: to),
                           start: CACurrentMediaTime(), duration: duration))
    }

    private func snapBack(piece: Piece, to square: Square, from p: CGPoint) {
        let g = geometry
        startSlide(Sliding(piece: piece, hide: square, from: p, to: g.center(of: square),
                           start: CACurrentMediaTime(), duration: 0.13))
    }

    private func startSlide(_ s: Sliding) {
        slideTimer?.invalidate()
        sliding = s
        needsDisplay = true
        let timer = Timer(timeInterval: 1.0 / 120.0, repeats: true) { [weak self] t in
            guard let self = self, let cur = self.sliding else { t.invalidate(); return }
            if CACurrentMediaTime() - cur.start >= cur.duration {
                self.sliding = nil
                t.invalidate()
                self.slideTimer = nil
            }
            self.needsDisplay = true
        }
        RunLoop.main.add(timer, forMode: .common)
        slideTimer = timer
    }

    /// Cancels any in-flight animation, e.g. because a new game started.
    func stopAnimations() {
        slideTimer?.invalidate()
        slideTimer = nil
        sliding = nil
        needsDisplay = true
    }
}
