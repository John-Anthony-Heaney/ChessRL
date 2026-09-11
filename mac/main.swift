//
//  main.swift -- entry point, menu bar and window.
//
//  ChessRL: a native AppKit front end for the C engine.  No browser, no web
//  view, no server, no Python -- the board is Core Graphics and the engine is
//  called in-process through src/api.h.
//

import AppKit

// MARK: - Command line

private func flagValue(_ name: String, in args: [String]) -> String? {
    if let i = args.firstIndex(of: name), i + 1 < args.count { return args[i + 1] }
    for a in args where a.hasPrefix(name + "=") {
        return String(a.dropFirst(name.count + 1))
    }
    return nil
}

let arguments = CommandLine.arguments

if arguments.contains("--help") || arguments.contains("-h") {
    print("""
    ChessRL -- play the trained champion.

      ChessRL                 open the app, newest runs/*/best.crl
      ChessRL --model PATH    open the app with a specific .crl model
      ChessRL --selftest      run the headless checks and exit
    """)
    exit(0)
}

let requestedModel = flagValue("--model", in: arguments)

if arguments.contains("--selftest") {
    // Command-line only, so resolving the model here cannot stall a window.
    exit(SelfTest.run(modelPath: ModelLocator.find(explicit: requestedModel)) ? 0 : 1)
}

// NOTE: the model is deliberately NOT looked up here.  Reading the repository
// can trigger a macOS privacy prompt, and a prompt raised before the process is
// a regular, activated application has nowhere to appear -- the app would hang
// with no window.  GameController does the lookup on its engine queue once the
// window is on screen.

// MARK: - Application delegate

final class AppDelegate: NSObject, NSApplicationDelegate {

    private let requestedModel: String?
    private var window: NSWindow?
    private var controller: GameController?

    init(requestedModel: String?) {
        self.requestedModel = requestedModel
        super.init()
    }

    func applicationWillFinishLaunching(_ notification: Notification) {
        NSApp.mainMenu = buildMenu()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let controller = GameController(requestedModel: requestedModel)
        self.controller = controller

        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1160, height: 800),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered,
                              defer: false)
        window.title = "ChessRL"
        window.contentViewController = controller
        window.contentMinSize = NSSize(width: 900, height: 700)
        window.minSize = NSSize(width: 900, height: 700)
        window.setContentSize(NSSize(width: 1160, height: 800))
        window.setFrameAutosaveName("ChessRLMainWindow")
        if !window.setFrameUsingName("ChessRLMainWindow") { window.center() }
        window.makeKeyAndOrderFront(nil)
        self.window = window

        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // A search may still be running on the engine queue; tell the controller
        // to disown its result so nothing touches the UI on the way out.
        controller?.prepareToTerminate()
        return .terminateNow
    }

    // MARK: Menu

    private func buildMenu() -> NSMenu {
        let name = "ChessRL"
        let main = NSMenu()

        // ---- Application menu
        let appItem = NSMenuItem()
        main.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "About \(name)",
                        action: #selector(showAbout(_:)), keyEquivalent: "").target = self
        appMenu.addItem(.separator())
        // Cmd-H belongs to Hint in this app, so Hide keeps only its menu entry.
        appMenu.addItem(withTitle: "Hide \(name)",
                        action: #selector(NSApplication.hide(_:)), keyEquivalent: "")
        let hideOthers = appMenu.addItem(withTitle: "Hide Others",
                                         action: #selector(NSApplication.hideOtherApplications(_:)),
                                         keyEquivalent: "h")
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        appMenu.addItem(withTitle: "Show All",
                        action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit \(name)",
                        action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu

        // ---- Game menu
        let gameItem = NSMenuItem()
        main.addItem(gameItem)
        let gameMenu = NSMenu(title: "Game")
        gameMenu.addItem(withTitle: "New Game",
                         action: #selector(GameController.menuNewGame(_:)), keyEquivalent: "n")
        gameMenu.addItem(.separator())
        gameMenu.addItem(withTitle: "Undo Move",
                         action: #selector(GameController.menuUndo(_:)), keyEquivalent: "z")
        gameMenu.addItem(withTitle: "Hint",
                         action: #selector(GameController.menuHint(_:)), keyEquivalent: "h")
        gameMenu.addItem(.separator())
        gameMenu.addItem(withTitle: "Flip Board",
                         action: #selector(GameController.menuFlip(_:)), keyEquivalent: "f")
        gameMenu.addItem(.separator())
        gameMenu.addItem(withTitle: "Resign",
                         action: #selector(GameController.menuResign(_:)), keyEquivalent: "")
        gameItem.submenu = gameMenu

        // ---- Window menu
        let windowItem = NSMenuItem()
        main.addItem(windowItem)
        let windowMenu = NSMenu(title: "Window")
        windowMenu.addItem(withTitle: "Minimize",
                           action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windowMenu.addItem(withTitle: "Zoom",
                           action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        windowMenu.addItem(.separator())
        windowMenu.addItem(withTitle: "Bring All to Front",
                           action: #selector(NSApplication.arrangeInFront(_:)), keyEquivalent: "")
        windowItem.submenu = windowMenu
        NSApp.windowsMenu = windowMenu

        return main
    }

    @objc private func showAbout(_ sender: Any?) {
        let credits = NSAttributedString(
            string: """
            A native macOS front end for the ChessRL engine.

            Bitboard chess core, population-based reinforcement learning and \
            alpha-beta search, all written from scratch in C.  This window is \
            AppKit and Core Graphics talking to that engine in-process.
            """,
            attributes: [.font: NSFont.systemFont(ofSize: 11),
                         .foregroundColor: NSColor.labelColor])
        NSApp.orderFrontStandardAboutPanel(options: [
            .applicationName: "ChessRL",
            .applicationVersion: "1.0",
            .credits: credits
        ])
    }
}

// MARK: - Run

let application = NSApplication.shared
application.setActivationPolicy(.regular)
let appDelegate = AppDelegate(requestedModel: requestedModel)
application.delegate = appDelegate
application.run()
