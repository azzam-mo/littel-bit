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
        self.state = .loading
        self.lastErrorMessage = nil

        // Files.app imports are security-scoped; keep the scope alive until mmap is ready.
        securityScopedModelURL?.stopAccessingSecurityScopedResource()
        securityScopedModelURL = directoryURL
        didStartModelAccess = directoryURL.startAccessingSecurityScopedResource()

        let bridge = LittleBitEngineBridge.sharedInstance()
        bridge.loadModel(atPath: directoryURL.path, maxSeqLen: maxSeqLen) { [weak self] success, errorMsg in
            guard let self = self else { return }
            guard bridge.state != .unloaded else { return } // deferred unload won teardown
            if success {
                self.state = .ready
                self.currentModelName = directoryURL.lastPathComponent
                self.currentModelPath = directoryURL.path
            } else {
                self.state = .error
                self.lastErrorMessage = errorMsg ?? "فشل تحميل النموذج."
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

    func unloadModel() {
        // The bridge calls back only after clear_cache() and model release have
        // finished on the main queue, even if generation is still in flight.
        LittleBitEngineBridge.sharedInstance().unloadModelWithCompletion { [weak self] in
            guard let self = self else { return }
            self.stopModelAccess()
            self.state = .unloaded
            self.currentModelName = "لا يوجد نموذج محدد"
            self.currentModelPath = nil
            self.lastErrorMessage = nil
        }
    }

    func sendUserMessage(_ prompt: String, temperature: Float = 0.6, topP: Float = 0.95, maxTokens: Int = 256) {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        // Append user message
        messages.append(ChatMessage(sender: "user", text: prompt))

        // Create empty assistant response message
        let assistantIndex = messages.count
        messages.append(ChatMessage(sender: "assistant", text: ""))
        self.isGenerating = true
        self.state = .generating

        let bridge = LittleBitEngineBridge.sharedInstance()
        bridge.generatePrompt(prompt, temperature: temperature, topP: topP, maxNewTokens: maxTokens) { [weak self] tokenStr in
            guard let self = self else { return false }
            if assistantIndex < self.messages.count {
                self.messages[assistantIndex].text += tokenStr
            }
            return true
        } completion: { [weak self] metrics in
            guard let self = self else { return }
            self.isGenerating = false
            if bridge.state != .unloaded {
                self.state = .ready
            }
            self.loadTimeMs = metrics.loadTimeMs
            self.ttftMs = metrics.ttftMs
            self.tokensPerSecond = metrics.tokensPerSecond
            self.kvCacheMB = metrics.kvCacheMB
            self.ramUsedMB = metrics.residentMemoryMB
        }
    }

    func stopGeneration() {
        let bridge = LittleBitEngineBridge.sharedInstance()
        bridge.stopGeneration()
        self.isGenerating = false
        if bridge.state != .unloaded {
            self.state = .ready
        }
    }

    func clearChat() {
        messages.removeAll()
    }
}
