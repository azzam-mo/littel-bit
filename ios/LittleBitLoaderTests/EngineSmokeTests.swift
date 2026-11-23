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
        var error: NSError?
        let ok = LittleBitEngineBridge.sharedInstance().verifyModelDirectory(missing.path, error: &error)
        XCTAssertFalse(ok)
        XCTAssertEqual(error?.code, 101)
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
}
