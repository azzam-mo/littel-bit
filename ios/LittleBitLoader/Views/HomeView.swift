import SwiftUI

struct HomeView: View {
    @ObservedObject var engine = LittleBitEngine.shared
    @Binding var selectedTab: Int

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header Card
                    VStack(spacing: 8) {
                        Image(systemName: "cpu.fill")
                            .font(.system(size: 54))
                            .foregroundColor(.blue)

                        Text("LittleBit Loader")
                            .font(.system(size: 28, weight: .bold, design: .rounded))

                        Text("Sub-1-Bit Qwen3-8B On-Device Engine")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 20)

                    // Status Card
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Text("حالة المحرك")
                                .font(.headline)
                            Spacer()
                            Text(engine.state.rawValue)
                                .font(.caption.bold())
                                .padding(.horizontal, 10)
                                .padding(.vertical, 4)
                                .background(statusColor.opacity(0.15))
                                .foregroundColor(statusColor)
                                .cornerRadius(8)
                        }

                        Divider()

                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("النموذج الحالي")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(engine.currentModelName)
                                    .font(.subheadline.bold())
                            }
                            Spacer()
                        }

                        if let err = engine.lastErrorMessage {
                            Text(err)
                                .font(.caption)
                                .foregroundColor(.red)
                        }
                    }
                    .padding(16)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(14)
                    .padding(.horizontal)

                    // Navigation Action Buttons
                    VStack(spacing: 14) {
                        Button(action: { selectedTab = 1 }) {
                            HStack {
                                Image(systemName: "folder.badge.gearshape")
                                Text("إدارة واستيراد النماذج (Model Manager)")
                                Spacer()
                                Image(systemName: "chevron.left")
                            }
                            .font(.headline)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }

                        Button(action: { selectedTab = 2 }) {
                            HStack {
                                Image(systemName: "message.fill")
                                Text("بدء المحادثة المحلية (Chat)")
                                Spacer()
                                Image(systemName: "chevron.left")
                            }
                            .font(.headline)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(engine.state == .ready ? Color.green : Color.gray.opacity(0.4))
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(engine.state != .ready)

                        Button(action: { selectedTab = 3 }) {
                            HStack {
                                Image(systemName: "speedometer")
                                Text("مراقبة الأداء والذاكرة (Performance)")
                                Spacer()
                                Image(systemName: "chevron.left")
                            }
                            .font(.headline)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color(.tertiarySystemBackground))
                            .foregroundColor(.primary)
                            .cornerRadius(12)
                        }
                    }
                    .padding(.horizontal)

                    // Target Device Info
                    VStack(alignment: .leading, spacing: 6) {
                        Text("مواصفات الجهاز المستهدف: iPhone 13 (A15 Bionic, 4GB RAM)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("النموذج المعتمد: littlebit-qwen3-8b-0.55bpw (LittleBit-2)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 10)
                }
            }
            .navigationTitle("الرئيسية")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var statusColor: Color {
        switch engine.state {
        case .unloaded: return .orange
        case .loading: return .blue
        case .ready: return .green
        case .generating: return .purple
        case .error: return .red
        }
    }
}
