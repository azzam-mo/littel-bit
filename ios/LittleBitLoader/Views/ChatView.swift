import SwiftUI

struct ChatView: View {
    @ObservedObject var engine = LittleBitEngine.shared
    @State private var inputText: String = ""
    @State private var temperature: Float = 0.6
    @State private var topP: Float = 0.95
    @State private var maxTokens: Int = 256
    @State private var showSettings: Bool = false

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Messages List
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 14) {
                            if engine.messages.isEmpty {
                                VStack(spacing: 12) {
                                    Image(systemName: "bubble.left.and.bubble.right")
                                        .font(.system(size: 40))
                                        .foregroundColor(.secondary)
                                    Text("ابدأ محادثة محلية مع نموذج LittleBit-Qwen3")
                                        .font(.subheadline)
                                        .foregroundColor(.secondary)
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.top, 60)
                            }

                            ForEach(engine.messages) { msg in
                                HStack {
                                    if msg.sender == "user" {
                                        Spacer()
                                        Text(msg.text)
                                            .padding(12)
                                            .background(Color.blue)
                                            .foregroundColor(.white)
                                            .cornerRadius(16)
                                            .frame(maxWidth: 280, alignment: .trailing)
                                    } else {
                                        VStack(alignment: .leading, spacing: 4) {
                                            Text(msg.text.isEmpty && engine.isGenerating ? "جاري التفكير..." : msg.text)
                                                .padding(12)
                                                .background(Color(.secondarySystemBackground))
                                                .foregroundColor(.primary)
                                                .cornerRadius(16)
                                                .frame(maxWidth: 280, alignment: .leading)
                                        }
                                        Spacer()
                                    }
                                }
                                .id(msg.id)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: engine.messages.last?.text) { _ in
                        if let lastId = engine.messages.last?.id {
                            proxy.scrollTo(lastId, anchor: .bottom)
                        }
                    }
                }

                Divider()

                // Input Bar
                HStack(spacing: 10) {
                    TextField("اكتب رسالتك هنا...", text: $inputText)
                        .padding(10)
                        .background(Color(.systemGray6))
                        .cornerRadius(20)
                        .disabled(engine.state != .ready && engine.state != .generating)

                    if engine.isGenerating {
                        Button(action: {
                            engine.stopGeneration()
                        }) {
                            Image(systemName: "stop.circle.fill")
                                .font(.system(size: 30))
                                .foregroundColor(.red)
                        }
                    } else {
                        Button(action: sendMessage) {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.system(size: 30))
                                .foregroundColor(canSend ? .blue : .gray)
                        }
                        .disabled(!canSend)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color(.systemBackground))
            }
            .navigationTitle("المحادثة المحلية")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: { engine.clearChat() }) {
                        Image(systemName: "trash")
                    }
                    .disabled(engine.messages.isEmpty)
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showSettings.toggle() }) {
                        Image(systemName: "slider.horizontal.3")
                    }
                }
            }
            .sheet(isPresented: $showSettings) {
                NavigationView {
                    Form {
                        Section(header: Text("إعدادات الاستدلال (Sampling)")) {
                            VStack(alignment: .leading) {
                                HStack {
                                    Text("درجة الحرارة (Temperature)")
                                    Spacer()
                                    Text(String(format: "%.2f", temperature))
                                }
                                Slider(value: $temperature, in: 0.1...1.5, step: 0.05)
                            }

                            VStack(alignment: .leading) {
                                HStack {
                                    Text("Top-P (Nucleus Sampling)")
                                    Spacer()
                                    Text(String(format: "%.2f", topP))
                                }
                                Slider(value: $topP, in: 0.1...1.0, step: 0.05)
                            }

                            VStack(alignment: .leading) {
                                HStack {
                                    Text("أقصى عدد توكنز جديدة")
                                    Spacer()
                                    Text("\(maxTokens)")
                                }
                                Stepper(value: $maxTokens, in: 32...1024, step: 32) {
                                    EmptyView()
                                }
                            }
                        }
                    }
                    .navigationTitle("إعدادات السياق")
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("إغلاق") { showSettings = false }
                        }
                    }
                }
            }
        }
    }

    private var canSend: Bool {
        return (engine.state == .ready) && !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        inputText = ""
        engine.sendUserMessage(text, temperature: temperature, topP: topP, maxTokens: maxTokens)
    }
}
