//
//  Pieces.swift -- vector chess pieces and the board coordinate system.
//
//  The artwork is pure Core Graphics: every piece is a closed CGPath authored in
//  a 100x100 design box with y pointing UP, so it stays crisp at any size and
//  does not depend on a font being installed.
//

import AppKit

// MARK: - Squares

/// A board square.  file 0 == 'a', rank 0 == '1'.
struct Square: Hashable {

    let file: Int
    let rank: Int

    init(file: Int, rank: Int) {
        self.file = file
        self.rank = rank
    }

    init(index: Int) {
        self.file = index % 8
        self.rank = index / 8
    }

    init?(_ name: String) {
        let s = Array(name.utf8)
        guard s.count == 2,
              s[0] >= 0x61, s[0] <= 0x68,     // a...h
              s[1] >= 0x31, s[1] <= 0x38      // 1...8
        else { return nil }
        file = Int(s[0]) - 0x61
        rank = Int(s[1]) - 0x31
    }

    var index: Int { rank * 8 + file }

    var name: String {
        let f = Character(UnicodeScalar(UInt8(0x61 + file)))
        let r = Character(UnicodeScalar(UInt8(0x31 + rank)))
        return "\(f)\(r)"
    }

    var isLight: Bool { (file + rank) % 2 == 1 }

    static let all: [Square] = (0 ..< 64).map { Square(index: $0) }
}

// MARK: - Geometry

/// Maps squares to pixels and back.  The single place the flip is applied, so
/// hit testing, highlights, drags and the promotion overlay can never disagree.
struct BoardGeometry {

    let origin: CGPoint          // bottom-left corner of the board, view coords (y up)
    let squareSize: CGFloat
    let flipped: Bool            // true == Black at the bottom

    /// Column drawn left-to-right for this square (0 == leftmost).
    func column(of sq: Square) -> Int { flipped ? 7 - sq.file : sq.file }
    /// Row drawn bottom-to-top for this square (0 == bottom).
    func row(of sq: Square) -> Int { flipped ? 7 - sq.rank : sq.rank }

    func rect(of sq: Square) -> CGRect {
        CGRect(x: origin.x + CGFloat(column(of: sq)) * squareSize,
               y: origin.y + CGFloat(row(of: sq)) * squareSize,
               width: squareSize, height: squareSize)
    }

    func center(of sq: Square) -> CGPoint {
        let r = rect(of: sq)
        return CGPoint(x: r.midX, y: r.midY)
    }

    /// Inverse of `rect`.  nil when the point is off the board.
    func square(at p: CGPoint) -> Square? {
        guard squareSize > 0 else { return nil }
        let cx = (p.x - origin.x) / squareSize
        let cy = (p.y - origin.y) / squareSize
        guard cx >= 0, cx < 8, cy >= 0, cy < 8 else { return nil }
        let col = min(7, max(0, Int(cx)))
        let row = min(7, max(0, Int(cy)))
        return Square(file: flipped ? 7 - col : col,
                      rank: flipped ? 7 - row : row)
    }
}

// MARK: - Pieces

enum PieceKind: Int {
    case pawn, knight, bishop, rook, queen, king
}

struct Piece: Equatable {

    let kind: PieceKind
    let isWhite: Bool

    init(kind: PieceKind, isWhite: Bool) {
        self.kind = kind
        self.isWhite = isWhite
    }

    init?(fen c: Character) {
        let white = c.isUppercase
        switch Character(c.lowercased()) {
        case "p": kind = .pawn
        case "n": kind = .knight
        case "b": kind = .bishop
        case "r": kind = .rook
        case "q": kind = .queen
        case "k": kind = .king
        default: return nil
        }
        isWhite = white
    }

    var letter: String {
        let base: String
        switch kind {
        case .pawn:   base = "p"
        case .knight: base = "n"
        case .bishop: base = "b"
        case .rook:   base = "r"
        case .queen:  base = "q"
        case .king:   base = "k"
        }
        return isWhite ? base.uppercased() : base
    }

    /// Rough centipawn-free material value, used for the tray readout.
    var value: Int {
        switch kind {
        case .pawn: return 1
        case .knight, .bishop: return 3
        case .rook: return 5
        case .queen: return 9
        case .king: return 0
        }
    }
}

// MARK: - FEN

enum FEN {

    /// Piece placement only.  Index 0 is a1, index 63 is h8.
    static func board(_ fen: String) -> [Piece?] {
        var out = [Piece?](repeating: nil, count: 64)
        guard let placement = fen.split(separator: " ").first else { return out }
        var rank = 7
        var file = 0
        for ch in placement {
            if ch == "/" {
                rank -= 1
                file = 0
                if rank < 0 { break }
            } else if let skip = ch.wholeNumberValue, skip >= 1, skip <= 8 {
                file += skip
            } else if let piece = Piece(fen: ch) {
                if file < 8 && rank >= 0 { out[rank * 8 + file] = piece }
                file += 1
            }
        }
        return out
    }

    /// The square of the given side's king, if it is on the board.
    static func king(of white: Bool, on board: [Piece?]) -> Square? {
        for i in 0 ..< 64 {
            if let p = board[i], p.kind == .king, p.isWhite == white { return Square(index: i) }
        }
        return nil
    }
}

// MARK: - Artwork

/// One piece drawn as vectors in a 100x100 box, y up.
struct PieceArt {

    let outline: CGPath          // filled with the piece colour, then stroked
    let marks: CGPath?           // stroked only, in the outline colour
    let dots: CGPath?            // filled with the outline colour (the knight's eye)

    private static var cache: [Int: PieceArt] = [:]

    static func art(for kind: PieceKind) -> PieceArt {
        if let hit = cache[kind.rawValue] { return hit }
        let made: PieceArt
        switch kind {
        case .pawn:   made = pawn()
        case .knight: made = knight()
        case .bishop: made = bishop()
        case .rook:   made = rook()
        case .queen:  made = queen()
        case .king:   made = king()
        }
        cache[kind.rawValue] = made
        return made
    }

    // ---------------------------------------------------------------- helpers

    private static func p() -> CGMutablePath { CGMutablePath() }

    // ------------------------------------------------------------------ pawn

    private static func pawn() -> PieceArt {
        let path = p()
        let headC = CGPoint(x: 50, y: 71)
        let headR: CGFloat = 15.5
        // Circle meets the collar top (y = 58) at x = 50 +/- 8.5
        let dx = sqrt(headR * headR - 13 * 13)

        path.move(to: CGPoint(x: 23, y: 8))
        path.addLine(to: CGPoint(x: 23, y: 15))
        path.addCurve(to: CGPoint(x: 38, y: 24),
                      control1: CGPoint(x: 35, y: 15.5), control2: CGPoint(x: 39, y: 19))
        path.addCurve(to: CGPoint(x: 42.5, y: 52),
                      control1: CGPoint(x: 40, y: 34), control2: CGPoint(x: 42, y: 44))
        path.addLine(to: CGPoint(x: 35, y: 52))
        path.addLine(to: CGPoint(x: 35, y: 58))
        path.addLine(to: CGPoint(x: 50 - dx, y: 58))
        path.addArc(center: headC, radius: headR,
                    startAngle: atan2(58 - headC.y, -dx),
                    endAngle: atan2(58 - headC.y, dx),
                    clockwise: true)
        path.addLine(to: CGPoint(x: 65, y: 58))
        path.addLine(to: CGPoint(x: 65, y: 52))
        path.addLine(to: CGPoint(x: 57.5, y: 52))
        path.addCurve(to: CGPoint(x: 62, y: 24),
                      control1: CGPoint(x: 58, y: 44), control2: CGPoint(x: 60, y: 34))
        path.addCurve(to: CGPoint(x: 77, y: 15),
                      control1: CGPoint(x: 61, y: 19), control2: CGPoint(x: 65, y: 15.5))
        path.addLine(to: CGPoint(x: 77, y: 8))
        path.closeSubpath()
        return PieceArt(outline: path, marks: nil, dots: nil)
    }

    // ------------------------------------------------------------------ rook

    private static func rook() -> PieceArt {
        let path = p()
        path.move(to: CGPoint(x: 19, y: 8))
        path.addLine(to: CGPoint(x: 19, y: 20))
        path.addLine(to: CGPoint(x: 27, y: 20))
        path.addLine(to: CGPoint(x: 30, y: 27))
        path.addCurve(to: CGPoint(x: 33, y: 61),
                      control1: CGPoint(x: 32, y: 38), control2: CGPoint(x: 33, y: 50))
        path.addLine(to: CGPoint(x: 22, y: 69))
        path.addLine(to: CGPoint(x: 22, y: 91))
        path.addLine(to: CGPoint(x: 36, y: 91))
        path.addLine(to: CGPoint(x: 36, y: 81))
        path.addLine(to: CGPoint(x: 43, y: 81))
        path.addLine(to: CGPoint(x: 43, y: 91))
        path.addLine(to: CGPoint(x: 57, y: 91))
        path.addLine(to: CGPoint(x: 57, y: 81))
        path.addLine(to: CGPoint(x: 64, y: 81))
        path.addLine(to: CGPoint(x: 64, y: 91))
        path.addLine(to: CGPoint(x: 78, y: 91))
        path.addLine(to: CGPoint(x: 78, y: 69))
        path.addLine(to: CGPoint(x: 67, y: 61))
        path.addCurve(to: CGPoint(x: 70, y: 27),
                      control1: CGPoint(x: 67, y: 50), control2: CGPoint(x: 68, y: 38))
        path.addLine(to: CGPoint(x: 73, y: 20))
        path.addLine(to: CGPoint(x: 81, y: 20))
        path.addLine(to: CGPoint(x: 81, y: 8))
        path.closeSubpath()

        let marks = p()
        marks.move(to: CGPoint(x: 22, y: 69))
        marks.addLine(to: CGPoint(x: 78, y: 69))
        marks.move(to: CGPoint(x: 27, y: 20))
        marks.addLine(to: CGPoint(x: 73, y: 20))
        return PieceArt(outline: path, marks: marks, dots: nil)
    }

    // ---------------------------------------------------------------- bishop

    private static func bishop() -> PieceArt {
        let path = p()
        path.move(to: CGPoint(x: 17, y: 8))
        path.addLine(to: CGPoint(x: 17, y: 18))
        path.addCurve(to: CGPoint(x: 33, y: 27),
                      control1: CGPoint(x: 25, y: 19), control2: CGPoint(x: 30, y: 22))
        path.addLine(to: CGPoint(x: 30, y: 34))
        path.addLine(to: CGPoint(x: 30, y: 41))
        path.addCurve(to: CGPoint(x: 35, y: 53),
                      control1: CGPoint(x: 30, y: 46), control2: CGPoint(x: 32, y: 50))
        path.addCurve(to: CGPoint(x: 50, y: 82),
                      control1: CGPoint(x: 29, y: 66), control2: CGPoint(x: 39, y: 76))
        path.addCurve(to: CGPoint(x: 65, y: 53),
                      control1: CGPoint(x: 61, y: 76), control2: CGPoint(x: 71, y: 66))
        path.addCurve(to: CGPoint(x: 70, y: 41),
                      control1: CGPoint(x: 68, y: 50), control2: CGPoint(x: 70, y: 46))
        path.addLine(to: CGPoint(x: 70, y: 34))
        path.addLine(to: CGPoint(x: 67, y: 27))
        path.addCurve(to: CGPoint(x: 83, y: 18),
                      control1: CGPoint(x: 70, y: 22), control2: CGPoint(x: 75, y: 19))
        path.addLine(to: CGPoint(x: 83, y: 8))
        path.closeSubpath()
        // Finial.
        path.addEllipse(in: CGRect(x: 43.5, y: 78, width: 13, height: 13))

        let marks = p()
        marks.move(to: CGPoint(x: 30, y: 41))          // the mitre band
        marks.addLine(to: CGPoint(x: 70, y: 41))
        marks.move(to: CGPoint(x: 44, y: 58))          // the traditional slit
        marks.addLine(to: CGPoint(x: 56, y: 70))
        return PieceArt(outline: path, marks: marks, dots: nil)
    }

    // ---------------------------------------------------------------- knight

    private static func knight() -> PieceArt {
        let path = p()
        path.move(to: CGPoint(x: 16, y: 8))
        path.addLine(to: CGPoint(x: 84, y: 8))
        path.addLine(to: CGPoint(x: 84, y: 18))
        path.addCurve(to: CGPoint(x: 74, y: 27),
                      control1: CGPoint(x: 81, y: 20), control2: CGPoint(x: 77, y: 23))
        path.addCurve(to: CGPoint(x: 78, y: 56),
                      control1: CGPoint(x: 76, y: 37), control2: CGPoint(x: 79, y: 46))
        path.addCurve(to: CGPoint(x: 70, y: 77),
                      control1: CGPoint(x: 77, y: 66), control2: CGPoint(x: 74, y: 72))
        path.addLine(to: CGPoint(x: 77, y: 90))        // rear ear
        path.addLine(to: CGPoint(x: 66, y: 83))        // notch between the ears
        path.addLine(to: CGPoint(x: 61, y: 90))        // forward ear
        path.addLine(to: CGPoint(x: 52, y: 76))        // forehead
        path.addCurve(to: CGPoint(x: 30, y: 61),
                      control1: CGPoint(x: 44, y: 71), control2: CGPoint(x: 36, y: 66))
        path.addCurve(to: CGPoint(x: 12, y: 52),
                      control1: CGPoint(x: 24, y: 58), control2: CGPoint(x: 17, y: 54))
        path.addCurve(to: CGPoint(x: 21, y: 42),
                      control1: CGPoint(x: 10, y: 47), control2: CGPoint(x: 15, y: 43))
        path.addLine(to: CGPoint(x: 33, y: 46))
        path.addCurve(to: CGPoint(x: 36, y: 26),
                      control1: CGPoint(x: 37, y: 38), control2: CGPoint(x: 33, y: 31))
        path.addCurve(to: CGPoint(x: 16, y: 18),
                      control1: CGPoint(x: 32, y: 22), control2: CGPoint(x: 24, y: 19))
        path.closeSubpath()

        let marks = p()
        marks.move(to: CGPoint(x: 24, y: 47))          // the muzzle line
        marks.addLine(to: CGPoint(x: 33, y: 49))
        marks.move(to: CGPoint(x: 55, y: 72))          // mane
        marks.addCurve(to: CGPoint(x: 68, y: 55),
                       control1: CGPoint(x: 63, y: 68), control2: CGPoint(x: 67, y: 62))

        let dots = p()
        dots.addEllipse(in: CGRect(x: 41, y: 63, width: 7, height: 7))
        return PieceArt(outline: path, marks: marks, dots: dots)
    }

    // ----------------------------------------------------------------- queen

    private static func queen() -> PieceArt {
        let path = p()
        path.move(to: CGPoint(x: 17, y: 8))
        path.addLine(to: CGPoint(x: 17, y: 19))
        path.addCurve(to: CGPoint(x: 32, y: 28),
                      control1: CGPoint(x: 25, y: 20), control2: CGPoint(x: 30, y: 23))
        path.addCurve(to: CGPoint(x: 36, y: 54),
                      control1: CGPoint(x: 33, y: 38), control2: CGPoint(x: 36, y: 46))
        path.addLine(to: CGPoint(x: 26, y: 58))
        path.addLine(to: CGPoint(x: 22, y: 64))
        path.addLine(to: CGPoint(x: 20, y: 79))         // spike 1
        path.addLine(to: CGPoint(x: 27.5, y: 67))
        path.addLine(to: CGPoint(x: 35, y: 85))         // spike 2
        path.addLine(to: CGPoint(x: 42.5, y: 67))
        path.addLine(to: CGPoint(x: 50, y: 89))         // centre spike
        path.addLine(to: CGPoint(x: 57.5, y: 67))
        path.addLine(to: CGPoint(x: 65, y: 85))
        path.addLine(to: CGPoint(x: 72.5, y: 67))
        path.addLine(to: CGPoint(x: 80, y: 79))
        path.addLine(to: CGPoint(x: 78, y: 64))
        path.addLine(to: CGPoint(x: 74, y: 58))
        path.addLine(to: CGPoint(x: 64, y: 54))
        path.addCurve(to: CGPoint(x: 68, y: 28),
                      control1: CGPoint(x: 64, y: 46), control2: CGPoint(x: 67, y: 38))
        path.addCurve(to: CGPoint(x: 83, y: 19),
                      control1: CGPoint(x: 70, y: 23), control2: CGPoint(x: 75, y: 20))
        path.addLine(to: CGPoint(x: 83, y: 8))
        path.closeSubpath()
        // Pearls on the spikes.
        for tip in [CGPoint(x: 20, y: 80), CGPoint(x: 35, y: 86), CGPoint(x: 50, y: 90),
                    CGPoint(x: 65, y: 86), CGPoint(x: 80, y: 80)] {
            path.addEllipse(in: CGRect(x: tip.x - 5.5, y: tip.y - 5.5, width: 11, height: 11))
        }

        let marks = p()
        marks.move(to: CGPoint(x: 24, y: 61))
        marks.addLine(to: CGPoint(x: 76, y: 61))
        return PieceArt(outline: path, marks: marks, dots: nil)
    }

    // ------------------------------------------------------------------ king

    private static func king() -> PieceArt {
        let path = p()
        path.move(to: CGPoint(x: 17, y: 8))
        path.addLine(to: CGPoint(x: 17, y: 19))
        path.addCurve(to: CGPoint(x: 32, y: 28),
                      control1: CGPoint(x: 25, y: 20), control2: CGPoint(x: 30, y: 23))
        path.addCurve(to: CGPoint(x: 36, y: 52),
                      control1: CGPoint(x: 33, y: 38), control2: CGPoint(x: 36, y: 45))
        path.addLine(to: CGPoint(x: 26, y: 56))
        path.addLine(to: CGPoint(x: 22, y: 62))
        path.addCurve(to: CGPoint(x: 33, y: 75),
                      control1: CGPoint(x: 21, y: 70), control2: CGPoint(x: 26, y: 75))
        path.addCurve(to: CGPoint(x: 42, y: 69),
                      control1: CGPoint(x: 37, y: 75), control2: CGPoint(x: 40, y: 72))
        path.addLine(to: CGPoint(x: 45, y: 74))
        path.addLine(to: CGPoint(x: 55, y: 74))
        path.addLine(to: CGPoint(x: 58, y: 69))
        path.addCurve(to: CGPoint(x: 67, y: 75),
                      control1: CGPoint(x: 60, y: 72), control2: CGPoint(x: 63, y: 75))
        path.addCurve(to: CGPoint(x: 78, y: 62),
                      control1: CGPoint(x: 74, y: 75), control2: CGPoint(x: 79, y: 70))
        path.addLine(to: CGPoint(x: 74, y: 56))
        path.addLine(to: CGPoint(x: 64, y: 52))
        path.addCurve(to: CGPoint(x: 68, y: 28),
                      control1: CGPoint(x: 64, y: 45), control2: CGPoint(x: 67, y: 38))
        path.addCurve(to: CGPoint(x: 83, y: 19),
                      control1: CGPoint(x: 70, y: 23), control2: CGPoint(x: 75, y: 20))
        path.addLine(to: CGPoint(x: 83, y: 8))
        path.closeSubpath()
        // The cross.
        let cross = p()
        cross.move(to: CGPoint(x: 45, y: 71))
        cross.addLine(to: CGPoint(x: 45, y: 81))
        cross.addLine(to: CGPoint(x: 35, y: 81))
        cross.addLine(to: CGPoint(x: 35, y: 88))
        cross.addLine(to: CGPoint(x: 45, y: 88))
        cross.addLine(to: CGPoint(x: 45, y: 97))
        cross.addLine(to: CGPoint(x: 55, y: 97))
        cross.addLine(to: CGPoint(x: 55, y: 88))
        cross.addLine(to: CGPoint(x: 65, y: 88))
        cross.addLine(to: CGPoint(x: 65, y: 81))
        cross.addLine(to: CGPoint(x: 55, y: 81))
        cross.addLine(to: CGPoint(x: 55, y: 71))
        cross.closeSubpath()
        path.addPath(cross)

        let marks = p()
        marks.move(to: CGPoint(x: 24, y: 59))
        marks.addLine(to: CGPoint(x: 76, y: 59))
        return PieceArt(outline: path, marks: marks, dots: nil)
    }
}

// MARK: - Rendering

enum PieceRenderer {

    /// Draws `piece` to fill `rect`, vector all the way down.
    static func draw(_ piece: Piece, in rect: CGRect, context ctx: CGContext,
                     shadow: Bool = true) {
        let art = PieceArt.art(for: piece.kind)
        let side = min(rect.width, rect.height)
        let inset = side * 0.05
        let scale = (side - inset * 2) / 100.0
        guard scale > 0 else { return }

        let fill: CGColor = piece.isWhite
            ? CGColor(red: 0.98, green: 0.97, blue: 0.95, alpha: 1)
            : CGColor(red: 0.16, green: 0.16, blue: 0.18, alpha: 1)
        let line: CGColor = piece.isWhite
            ? CGColor(red: 0.20, green: 0.19, blue: 0.18, alpha: 1)
            : CGColor(red: 0.92, green: 0.91, blue: 0.89, alpha: 1)

        ctx.saveGState()
        ctx.translateBy(x: rect.midX - side / 2 + inset, y: rect.midY - side / 2 + inset)
        ctx.scaleBy(x: scale, y: scale)
        ctx.setLineJoin(.round)
        ctx.setLineCap(.round)

        if shadow {
            // The CTM is applied to shadow parameters, so undo the scale here.
            ctx.setShadow(offset: CGSize(width: 0, height: -1.5 / scale),
                          blur: 3.0 / scale,
                          color: CGColor(red: 0, green: 0, blue: 0, alpha: 0.30))
        }
        ctx.addPath(art.outline)
        ctx.setFillColor(fill)
        ctx.fillPath()
        ctx.setShadow(offset: .zero, blur: 0, color: nil)

        ctx.addPath(art.outline)
        ctx.setStrokeColor(line)
        ctx.setLineWidth(3.2)
        ctx.strokePath()

        if let marks = art.marks {
            ctx.addPath(marks)
            ctx.setStrokeColor(line)
            ctx.setLineWidth(2.6)
            ctx.strokePath()
        }
        if let dots = art.dots {
            ctx.addPath(dots)
            ctx.setFillColor(line)
            ctx.fillPath()
        }
        ctx.restoreGState()
    }
}
