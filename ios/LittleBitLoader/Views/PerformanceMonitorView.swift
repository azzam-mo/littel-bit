import SwiftUI

struct MetricRow: View {
    let title: String
    let value: String
    let icon: String
    var statusColor: Color = .blue

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 20))
                .foregroundColor(statusColor)
                .frame(width: 28)

            Text(title)
                .font(.subheadline)

            Spacer()

            Text(value)
                .font(.subheadline.bold())
                .foregroundColor(.primary)
        }
        .padding(.vertical, 4)
    }
}

struct PerformanceMonitorView: View {
    @ObservedObject var engine = LittleBitEngine.shared

    // iPhone 13 has 4GB physical RAM. Apps typically get Jetsam-killed if resident RAM exceeds ~2.8-3.0 GB.
    private let ramLimitMB: Double = 3000.0

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // RAM Usage Gauge Card
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("الذاكرة المستخدمة (Resident RAM)")
                                .font(.headline)
                            Spacer()
                            Text("\(engine.ramUsedMB) MB")
                                .font(.headline.bold())
                                .foregroundColor(ramGaugeColor)
                        }

                        // Progress Bar
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(Color(.systemGray5))
                                    .frame(height: 14)

                                RoundedRectangle(cornerRadius: 6)
                                    .fill(ramGaugeColor)
                                    .frame(width: min(geo.size.width * CGFloat(Double(engine.ramUsedMB) / ramLimitMB), geo.size.width), height: 14)
                            }
                        }
                        .frame(height: 14)

                        HStack {
                            Text("السقف الآمن: 2800 MB")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(ramStatusText)
                                .font(.caption2.bold())
                                .foregroundColor(ramGaugeColor)
                        }
                    }
                    .padding(16)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(14)
                    .padding(.horizontal)

                    // Real-Time Metrics Card
                    VStack(alignment: .leading, spacing: 14) {
                        Text("مؤشرات الاستدلال والسرعة")
                            .font(.headline)

                        Divider()

                        MetricRow(
                            title: "Tokens Per Second",
                            value: String(format: "%.2f tok/s", engine.tokensPerSecond),
                            icon: "speedometer",
                            statusColor: .green
                        )

                        MetricRow(
                            title: "زمن الاستجابة (TTFT)",
                            value: String(format: "%.1f ms", engine.ttftMs),
                            icon: "stopwatch.fill",
                            statusColor: .purple
                        )

                        MetricRow(
                            title: "زمن تحميل النموذج",
                            value: String(format: "%.2f s", engine.loadTimeMs / 1000.0),
                            icon: "hourglass",
                            statusColor: .orange
                        )

                        MetricRow(
                            title: "حجم ذاكرة KV Cache",
                            value: "\(engine.kvCacheMB) MB",
                            icon: "memorychip",
                            statusColor: .blue
                        )

                        MetricRow(
                            title: "حالة المحرك",
                            value: engine.state.rawValue,
                            icon: "gearshape.2.fill",
                            statusColor: .teal
                        )
                    }
                    .padding(16)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(14)
                    .padding(.horizontal)

                    // Architecture Specifications Card
                    VStack(alignment: .leading, spacing: 10) {
                        Text("مواصفات بيئة التشغيل")
                            .font(.headline)

                        Divider()

                        HStack {
                            Text("الجهاز")
                            Spacer()
                            Text("Apple iPhone 13")
                                .foregroundColor(.secondary)
                        }
                        .font(.caption)

                        HStack {
                            Text("المعالج")
                            Spacer()
                            Text("Apple A15 Bionic (6-Core CPU, 4-Core GPU, 16-Core NPU)")
                                .foregroundColor(.secondary)
                        }
                        .font(.caption)

                        HStack {
                            Text("الذاكرة الكلية")
                            Spacer()
                            Text("4.0 GB LPDDR4X")
                                .foregroundColor(.secondary)
                        }
                        .font(.caption)

                        HStack {
                            Text("معمارية التحميل")
                            Spacer()
                            Text("Zero-Copy mmap + On-The-Fly Binary GEMM")
                                .foregroundColor(.secondary)
                        }
                        .font(.caption)
                    }
                    .padding(16)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(14)
                    .padding(.horizontal)
                }
                .padding(.vertical)
            }
            .navigationTitle("مراقبة الأداء")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var ramGaugeColor: Color {
        let mb = Double(engine.ramUsedMB)
        if mb < 1800 { return .green }
        if mb < 2500 { return .orange }
        return .red
    }

    private var ramStatusText: String {
        let mb = Double(engine.ramUsedMB)
        if mb < 1800 { return "آمن ومستقر" }
        if mb < 2500 { return "استهلاك مرتفع" }
        return "تحذير: اقتراب من حد الذاكرة"
    }
}
