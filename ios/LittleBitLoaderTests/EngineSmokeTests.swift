import Foundation
import XCTest

/// Executable on a real iPhone (or simulator) via `xcodebuild test`.
/// The inference correctness itself is covered on Linux by `test_model_inference`;
/// these tests validate the Objective-C++ bridge and engine lifecycle on Apple platforms.
final class EngineSmokeTests: XCTestCase {

    func testInitialStateIsUnloaded() {
        XCTAssertEqual(LittleBitEngineBridge.sharedInstance().state, .unloaded)
    }

    func testMissingModelDirectoryIsRejected() throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("littlebit-missing-\(UUID().uuidString)")
        do {
            try LittleBitEngineBridge.sharedInstance().verifyModelDirectory(missing.path)
            XCTFail("Expected verifyModelDirectory to reject a missing directory")
        } catch {
            XCTAssertEqual((error as NSError).code, 101)
        }
    }

    func testResidentMemoryIsAvailable() {
        XCTAssertGreaterThanOrEqual(LittleBitEngineBridge.sharedInstance().currentResidentMemoryMB(), 0)
    }

    func testChatTemplateContainsQwenControlTokens() {
        // Mirrors Tokenizer::apply_chat_template in the C++ engine.
        let prompt = "<|im_start|>user\nمرحبا<|im_end|>\n<|im_start|>assistant\n"
        XCTAssertTrue(prompt.contains("<|im_start|>"))
        XCTAssertTrue(prompt.contains("<|im_end|>"))
    }

    func testGenerationRejectedBeforeModelLoad() {
        let expectation = expectation(description: "completion called")
        LittleBitEngineBridge.sharedInstance().generatePrompt(
            "test", temperature: 0, topP: 1, maxNewTokens: 1,
            onToken: { _ in true }
        ) { _ in expectation.fulfill() }
        wait(for: [expectation], timeout: 2.0)
    }

    // MARK: - Tiny-fixture lifecycle tests (A/B/C)

    /// CI generates tests/make_tiny_model.py into a runner-temp directory and
    /// passes it as TEST_RUNNER_LITTLEBIT_TINY_MODEL; xcodebuild strips the
    /// TEST_RUNNER_ prefix before launching the test runner.
    private func tinyModelDirectory() throws -> URL {
        guard let value = ProcessInfo.processInfo.environment["LITTLEBIT_TINY_MODEL"],
              !value.isEmpty else {
            throw XCTSkip("LITTLEBIT_TINY_MODEL is not set; skipping lifecycle test")
        }
        let url = URL(fileURLWithPath: value)
        for name in ["model.safetensors", "config.json", "vocab.json", "merges.txt"] {
            guard FileManager.default.fileExists(atPath: url.appendingPathComponent(name).path) else {
                throw XCTSkip("Tiny model fixture is missing \(name) at \(value)")
            }
        }
        return url
    }

    private func loadTinyModel(_ directory: URL) {
        let bridge = LittleBitEngineBridge.sharedInstance()
        let loaded = expectation(description: "tiny model loaded")
        bridge.loadModel(atPath: directory.path, maxSeqLen: 64) { success, message in
            XCTAssertTrue(success, message ?? "load failed without an error message")
            loaded.fulfill()
        }
        wait(for: [loaded], timeout: 10.0)
        XCTAssertEqual(bridge.state, .ready)
    }

    private func unloadTinyModel() {
        let bridge = LittleBitEngineBridge.sharedInstance()
        guard bridge.state != .unloaded else { return }
        let unloaded = expectation(description: "tiny model unloaded")
        bridge.unloadModelWithCompletion { unloaded.fulfill() }
        wait(for: [unloaded], timeout: 10.0)
    }

    /// A. Unload during generation: the bridge must keep the model mapped and
    /// the state generating until the in-flight pass has been cancelled.
    func testUnloadDuringGenerationDefersTeardown() throws {
        let directory = try tinyModelDirectory()
        unloadTinyModel()
        loadTinyModel(directory)

        let bridge = LittleBitEngineBridge.sharedInstance()
        let generationDone = expectation(description: "generation completion called")
        let unloadDone = expectation(description: "deferred unload completion called")
        var unloadRequested = false

        bridge.generatePrompt(
            "unload lifecycle probe",
            temperature: 0,
            topP: 1,
            maxNewTokens: 4096,
            onToken: { _ in
                if !unloadRequested {
                    unloadRequested = true
                    // onToken runs on the main queue while _busy is still true.
                    bridge.unloadModelWithCompletion { unloadDone.fulfill() }
                }
                return true
            },
            completion: { _ in
                XCTAssertEqual(bridge.state, .unloaded)
                generationDone.fulfill()
            }
        )
        XCTAssertEqual(bridge.state, .generating)
        XCTAssertNotNil(bridge.currentModelPath)

        wait(for: [unloadDone, generationDone], timeout: 10.0)
        XCTAssertTrue(unloadRequested)
        XCTAssertEqual(bridge.state, .unloaded)
        XCTAssertNil(bridge.currentModelPath)
    }

    /// B. Unload during load: the load completion must not leave the bridge
    /// ready, and the gate must be free again for a subsequent load.
    func testUnloadDuringLoadDefersTeardown() throws {
        let directory = try tinyModelDirectory()
        unloadTinyModel()

        let bridge = LittleBitEngineBridge.sharedInstance()
        let loadDone = expectation(description: "load completion called after deferred unload")
        let unloadDone = expectation(description: "deferred unload completion called")

        bridge.loadModel(atPath: directory.path, maxSeqLen: 64) { success, _ in
            XCTAssertFalse(success)
            XCTAssertEqual(bridge.state, .unloaded)
            loadDone.fulfill()
        }
        // Called synchronously from the main test thread: _busy is already true.
        bridge.unloadModelWithCompletion { unloadDone.fulfill() }

        XCTAssertEqual(bridge.state, .loading)
        wait(for: [unloadDone, loadDone], timeout: 10.0)
        XCTAssertEqual(bridge.state, .unloaded)
        XCTAssertNil(bridge.currentModelPath)

        // If _busy had been released too early this load would still succeed,
        // but the earlier teardown would race it; assert the gate is clean now.
        let reloadDone = expectation(description: "reload after deferred teardown")
        bridge.loadModel(atPath: directory.path, maxSeqLen: 64) { success, _ in
            XCTAssertTrue(success)
            reloadDone.fulfill()
        }
        wait(for: [reloadDone], timeout: 10.0)
        XCTAssertEqual(bridge.state, .ready)
        unloadTinyModel()
    }

    /// C. Completion ordering: unload teardown callback precedes the generation
    /// completion, so a UI model cannot be forced back to .ready afterwards.
    func testUnloadCompletionAndGenerationOrdering() throws {
        let directory = try tinyModelDirectory()
        unloadTinyModel()
        loadTinyModel(directory)

        let bridge = LittleBitEngineBridge.sharedInstance()
        let generationDone = expectation(description: "generation completion called")
        let unloadDone = expectation(description: "unload completion called")
        let lock = NSLock()
        var order: [String] = []
        var unloadRequested = false

        bridge.generatePrompt(
            "completion ordering probe",
            temperature: 0,
            topP: 1,
            maxNewTokens: 4096,
            onToken: { _ in
                if !unloadRequested {
                    unloadRequested = true
                    bridge.unloadModelWithCompletion {
                        lock.lock()
                        order.append("unload")
                        lock.unlock()
                        unloadDone.fulfill()
                    }
                }
                return true
            },
            completion: { _ in
                lock.lock()
                order.append("generation")
                lock.unlock()
                XCTAssertEqual(bridge.state, .unloaded)
                generationDone.fulfill()
            }
        )

        wait(for: [unloadDone, generationDone], timeout: 10.0)
        lock.lock()
        let observed = order
        lock.unlock()
        XCTAssertEqual(observed, ["unload", "generation"])
        XCTAssertEqual(bridge.state, .unloaded)
    }
}
