import SwiftUI
import UniformTypeIdentifiers

struct ModelFileItem: Identifiable {
    let id = UUID()
    let name: String
    let sizeMB: Double
    let isRequired: Bool
    let exists: Bool
}

struct ModelManagerView: View {
    @ObservedObject var engine = LittleBitEngine.shared
    @State private var isImporting = false
    @State private var importedModels: [URL] = []
    @State private var selectedModelURL: URL? = nil
    @State private var fileItems: [ModelFileItem] = []
    @State private var verificationMessage: String = ""
    @State private var isCompatible: Bool = false

    var body: some View {
        NavigationView {
            List {
                Section(header: Text("استيراد من تطبيق Files")) {
                    Button(action: { isImporting = true }) {
                        Label("اختيار مجلد النموذج من ملفات iPhone", systemImage: "square.and.arrow.down")
                    }
                }

                if !importedModels.isEmpty {
                    Section(header: Text("النماذج المتاحة محليًا")) {
                        ForEach(importedModels, id: \.self) { url in
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(url.lastPathComponent)
                                        .font(.headline)
                                    Text(url.path)
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                if engine.currentModelPath == url.path {
                                    Text("محمّل")
                                        .font(.caption.bold())
                                        .foregroundColor(.green)
                                }
                            }
                            .contentShape(Rectangle())
                            .onTapGesture {
                                selectModel(url)
                            }
                        }
                        .onDelete(perform: deleteModel)
                    }
                }

                if let model = selectedModelURL {
                    Section(header: Text("فحص توافق ملفات النموذج (\(model.lastPathComponent))")) {
                        ForEach(fileItems) { item in
                            HStack {
                                Image(systemName: item.exists ? "checkmark.circle.fill" : "xmark.circle.fill")
                                    .foregroundColor(item.exists ? .green : (item.isRequired ? .red : .orange))

                                VStack(alignment: .leading) {
                                    Text(item.name)
                                        .font(.subheadline)
                                    if item.exists {
                                        Text(String(format: "%.2f MB", item.sizeMB))
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                    }
                                }
                                Spacer()
                                if item.isRequired {
                                    Text("أساسي")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }

                        if !verificationMessage.isEmpty {
                            Text(verificationMessage)
                                .font(.caption)
                                .foregroundColor(isCompatible ? .green : .red)
                        }

                        if engine.currentModelPath != model.path {
                            Button(action: {
                                engine.loadModel(from: model)
                            }) {
                                HStack {
                                    Spacer()
                                    Text("تحميل النموذج في المحرك")
                                        .bold()
                                    Spacer()
                                }
                            }
                            .disabled(!isCompatible || engine.state == .loading)
                        } else {
                            Button(role: .destructive, action: {
                                engine.unloadModel()
                            }) {
                                HStack {
                                    Spacer()
                                    Text("إلغاء تحميل النموذج")
                                    Spacer()
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("إدارة النماذج")
            .onAppear(perform: scanDocumentsDirectory)
            .fileImporter(
                isPresented: $isImporting,
                allowedContentTypes: [.folder],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    if let first = urls.first {
                        _ = first.startAccessingSecurityScopedResource()
                        importModelFolder(from: first)
                        first.stopAccessingSecurityScopedResource()
                    }
                case .failure(let error):
                    print("Error importing: \(error.localizedDescription)")
                }
            }
        }
    }

    private func scanDocumentsDirectory() {
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        if let contents = try? fm.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil, options: .skipsHiddenFiles) {
            importedModels = contents.filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true }
            if selectedModelURL == nil, let first = importedModels.first {
                selectModel(first)
            }
        }
    }

    private func importModelFolder(from sourceURL: URL) {
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let destURL = docs.appendingPathComponent(sourceURL.lastPathComponent)
        // The file-importer security scope is stopped as soon as this method
        // returns, so copy to staging synchronously while it is still valid.
        let stagingURL = docs.appendingPathComponent(".import-\(UUID().uuidString)")

        do {
            try fm.copyItem(at: sourceURL, to: stagingURL)
        } catch {
            print("Failed to stage model import: \(error)")
            return
        }

        let finishImport = { [self] in
            try? fm.removeItem(at: destURL)
            do {
                try fm.moveItem(at: stagingURL, to: destURL)
                self.scanDocumentsDirectory()
                self.selectModel(destURL)
            } catch {
                print("Failed to install staged model: \(error)")
                try? fm.removeItem(at: stagingURL)
            }
        }

        // Always drain the bridge gate, not only when currentModelPath matches:
        // a load may still be in flight, in which case currentModelPath is nil
        // or still points at the previous model while C++ is reading this folder.
        engine.unloadModel(completion: finishImport)
    }

    /// Mirrors `Qwen3Model::load` and `verifyModelDirectory:error:` so the UI
    /// cannot report compatibility for a directory the engine will reject.
    static func isCompatibleModelDirectory(_ url: URL) -> Bool {
        do {
            try LittleBitEngineBridge.sharedInstance().verifyModelDirectory(url.path)
            return true
        } catch {
            return false
        }
    }

    private func selectModel(_ url: URL) {
        selectedModelURL = url
        let fm = FileManager.default

        // Engine contract (src/model.cpp:457-537): config.json,
        // littlebit_config.json and single-file model.safetensors are required.
        // Tokenizer is tokenizer.json OR vocab.json + merges.txt.
        let expectedFiles: [(String, Bool)] = [
            ("model.safetensors", true),
            ("config.json", true),
            ("littlebit_config.json", true),
            ("vocab.json", false),
            ("merges.txt", false),
            ("tokenizer.json", false),
            ("generation_config.json", false)
        ]

        var items: [ModelFileItem] = []

        for (name, req) in expectedFiles {
            let fileURL = url.appendingPathComponent(name)
            let exists = fm.fileExists(atPath: fileURL.path)
            var sizeMB: Double = 0.0
            if exists,
               let attr = try? fm.attributesOfItem(atPath: fileURL.path),
               let bytes = attr[.size] as? Int64 {
                sizeMB = Double(bytes) / (1024.0 * 1024.0)
            }
            items.append(ModelFileItem(name: name, sizeMB: sizeMB, isRequired: req, exists: exists))
        }

        let hasTokenizerJSON = fm.fileExists(atPath: url.appendingPathComponent("tokenizer.json").path)
        let hasVocabAndMerges = fm.fileExists(atPath: url.appendingPathComponent("vocab.json").path)
            && fm.fileExists(atPath: url.appendingPathComponent("merges.txt").path)
        let hasShardedIndex = fm.fileExists(atPath: url.appendingPathComponent("model.safetensors.index.json").path)
        self.fileItems = items
        self.isCompatible = Self.isCompatibleModelDirectory(url)
        self.verificationMessage = self.isCompatible ?
            "النموذج متوافق وكافة الملفات المطلوبة متوفرة." :
            hasShardedIndex ?
                "تنبيه: المحرك لا يدعم النماذج المجزأة (model.safetensors.index.json)." :
                "تنبيه: الملفات الأساسية أو مسار الـ tokenizer غير مكتمل."
    }

    private func deleteModel(at offsets: IndexSet) {
        let fm = FileManager.default
        let urls = offsets.map { importedModels[$0] }
        for url in urls {
            let remove = { [self] in
                try? fm.removeItem(at: url)
                self.scanDocumentsDirectory()
                if self.selectedModelURL?.path == url.path {
                    self.selectedModelURL = nil
                }
            }
            // Wait for the bridge gate even when this folder is not the
            // recorded current model: a load of it may still be in flight.
            engine.unloadModel(completion: remove)
        }
    }
}
