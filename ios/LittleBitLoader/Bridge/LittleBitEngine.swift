import Foundation
import Combine

enum AppEngineState: String {
    case unloaded = "غير محمل"
    case loading = "جاري التحميل..."
    case ready = "جاهز للاستدلال"
    case generating = "جاري التوليد..."
    case error = "خطأ"
}

struct ChatMessage: Identifiable, Codable {
    let id: UUID
    let sender: String // "user" or "assistant"
    var text: String
    let timestamp: Date

    init(sender: String, text: String) {
        self.id = UUID()
        self.sender = sender
        self.text = text
        self.timestamp = Date()
    }
}

@MainActor
class LittleBitEngine: ObservableObject {
    static let shared = LittleBitEngine()

    @Published var state: AppEngineState = .unloaded
    @Published var currentModelName: String = "لا يوجد نموذج محدد"
    @Published var currentModelPath: String? = nil
    @Published var lastErrorMessage: String? = nil

    @Published var messages: [ChatMessage] = []
    @Published var isGenerating: Bool = false

    // Performance metrics
    @Published var ramUsedMB: Int = 0
    @Published var loadTimeMs: Double = 0.0
    @Published var tokensPerSecond: Double = 0.0
    @Published var ttftMs: Double = 0.0
    @Published var kvCacheMB: Int = 0

    private var memoryTimer: Timer?
    private var securityScopedModelURL: URL?
    private var didStartModelAccess = false

    init() {
        startMemoryMonitoring()
    }

    private func startMemoryMonitoring() {
        memoryTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.ramUsedMB = LittleBitEngineBridge.sharedInstance().currentResidentMemoryMB()
            }
        }
    }

    func loadModel(from directoryURL: URL, maxSeqLen: Int = 1024) {
        let bridge = LittleBitEngineBridge.sharedInstance()
        self.lastErrorMessage = nil

        // Keep the previous security scope alive until the bridge confirms the
        // new load was accepted. Stopping it early can invalidate mmap-backed
        // pages if the new load is rejected while the old model is still live.
        let previousURL = securityScopedModelURL
        let previousDidStart = didStartModelAccess
        let newDidStart = directoryURL.startAccessingSecurityScopedResource()
        self.state = .loading

        bridge.loadModel(atPath: directoryURL.path, maxSeqLen: maxSeqLen) { [weak self] success, errorMsg in
            guard let self = self else {
                if newDidStart { directoryURL.stopAccessingSecurityScopedResource() }
                return
            }
            if success && bridge.state != .unloaded {
                if previousDidStart { previousURL?.stopAccessingSecurityScopedResource() }
                self.securityScopedModelURL = directoryURL
                self.didStartModelAccess = newDidStart
                self.state = .ready
                self.currentModelName = directoryURL.lastPathComponent
                self.currentModelPath = directoryURL.path
                return
            }

            // The bridge rejected or failed the load. Drop only the new scope;
            // the stored scope still describes whichever model remains mapped.
            if newDidStart { directoryURL.stopAccessingSecurityScopedResource() }
            guard bridge.state != .unloaded else { return } // deferred unload owns state/scope
            self.lastErrorMessage = errorMsg ?? "فشل تحميل النموذج."
            switch bridge.state {
            case .ready: self.state = .ready
            case .generating: self.state = .generating
            case .loading: self.state = .loading
            case .error: self.state = .error
            case .unloaded: self.state = .unloaded
            }
        }
    }

    private func stopModelAccess() {
        if didStartModelAccess {
            securityScopedModelURL?.stopAccessingSecurityScopedResource()
            didStartModelAccess = false
        }
        securityScopedModelURL = nil
    }

    func unloadModel(completion: (() -> Void)? = nil) {
        // The bridge calls back only after clear_cache() and model release have
        // finished on the main queue, even if generation is still in flight.
        LittleBitEngineBridge.sharedInstance().unloadModel(completion: { [weak self] in
            guard let self = self else {
                completion?()
                return
            }
            self.stopModelAccess()
            self.state = .unloaded
            self.currentModelName = "لا يوجد نموذج محدد"
            self.currentModelPath = nil
            self.lastErrorMessage = nil
            completion?()
        })
    }

    func sendUserMessage(_ prompt: String, temperature: Float = 0.6, topP: Float = 0.95, maxTokens: Int = 256) {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        // Append user message
        messages.append(ChatMessage(sender: "user", text: prompt))

        // Create empty assistant response message
        let assistantMessage = ChatMessage(sender: "assistant", text: "")
        let assistantID = assistantMessage.id
        messages.append(assistantMessage)
        // The bridge is the source of truth for the busy gate; do not publish
        // .generating here or Stop would let a second request be accepted.
        let bridge = LittleBitEngineBridge.sharedInstance()
        bridge.generatePrompt(prompt, temperature: temperature, topP: topP, maxNewTokens: maxTokens) { [weak self] tokenStr in
            guard let self = self else { return false }
            // Target by identity: Clear chat during generation must not append
            // late tokens from this request to a newer assistant bubble.
            if let index = self.messages.firstIndex(where: { $0.id == assistantID }) {
                self.messages[index].text += tokenStr
            }
            return true
        } completion: { [weak self] metrics in
            guard let self = self else { return }
            self.isGenerating = false
            if metrics.accepted && bridge.state != .unloaded {
                self.state = .ready
            }
            if !metrics.accepted {
                // Rejected request: remove the empty assistant bubble and keep
                // the state the bridge actually reported.
                self.messages.removeAll { $0.id == assistantID }
            }
            self.loadTimeMs = metrics.loadTimeMs
            self.ttftMs = metrics.ttftMs
            self.tokensPerSecond = metrics.tokensPerSecond
            self.kvCacheMB = metrics.kvCacheMB
            self.ramUsedMB = metrics.residentMemoryMB
        }
        // Acceptance is synchronous inside generatePrompt: a rejected request
        // leaves the bridge state unchanged and its completion already ran.
        if bridge.state == .generating {
            self.isGenerating = true
            self.state = .generating
        }
    }

    func stopGeneration() {
        // Only ask the bridge to cancel. isGenerating/.generating stay set until
        // the generation completion runs, so Send remains disabled while the
        // background pass is still winding down.
        LittleBitEngineBridge.sharedInstance().stopGeneration()
    }

    func clearChat() {
        messages.removeAll()
    }
}
