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

        let bridge = LittleBitEngineBridge.sharedInstance()
        bridge.loadModel(atPath: directoryURL.path, maxSeqLen: maxSeqLen) { [weak self] success, errorMsg in
            guard let self = self else { return }
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

    func unloadModel() {
        LittleBitEngineBridge.sharedInstance().unloadModel()
        self.state = .unloaded
        self.currentModelName = "لا يوجد نموذج محدد"
        self.currentModelPath = nil
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
            self.state = .ready
            if let m = metrics {
                self.loadTimeMs = m.loadTimeMs
                self.ttftMs = m.ttftMs
                self.tokensPerSecond = m.tokensPerSecond
                self.kvCacheMB = m.kvCacheMB
                self.ramUsedMB = m.residentMemoryMB
            }
        }
    }

    func stopGeneration() {
        LittleBitEngineBridge.sharedInstance().stopGeneration()
        self.isGenerating = false
        self.state = .ready
    }

    func clearChat() {
        messages.removeAll()
    }
}
