#import "LittleBitEngineBridge.h"
#include "littlebit/model.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <dispatch/dispatch.h>
#include <sys/qos.h>

namespace {

// Byte-level BPE can emit a multi-byte UTF-8 sequence across several tokens.
// Buffer incomplete tails so SwiftUI never receives invalid standalone bytes.
std::string append_utf8_stream(std::string& pending, const std::string& bytes) {
    std::string buffer = pending + bytes;
    pending.clear();
    std::string out;
    size_t i = 0;
    while (i < buffer.size()) {
        unsigned char c = static_cast<unsigned char>(buffer[i]);
        size_t need = (c < 0x80) ? 1 : (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 0;
        if (need == 0 || i + need > buffer.size()) {
            if (need == 0) ++i;
            else break;
            continue;
        }
        out.append(buffer, i, need);
        i += need;
    }
    if (i < buffer.size()) pending = buffer.substr(i);
    return out;
}

} // namespace

@implementation EnginePerformanceMetrics
@end

@interface LittleBitEngineBridge () {
    std::unique_ptr<littlebit::Qwen3Model> _model;
    std::atomic<bool> _cancelGeneration;
    std::atomic<bool> _busy;
    std::atomic<bool> _pendingUnload;
    std::mutex _lifecycleMutex;
    std::string _pendingUTF8;
    NSMutableArray* _pendingUnloadCompletions;
}
@property (nonatomic, readwrite) EngineState state;
@property (nonatomic, readwrite, nullable) NSString* currentModelPath;
@property (nonatomic, readwrite, nullable) NSString* lastError;
- (void)teardownModelOnMainQueue;
- (void)drainPendingUnloadOnMainQueue;
- (void)finishBusyOperationOnMainQueue;
@end

@implementation LittleBitEngineBridge

+ (instancetype)sharedInstance {
    static LittleBitEngineBridge* instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[LittleBitEngineBridge alloc] init];
    });
    return instance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _state = EngineStateUnloaded;
        _currentModelPath = nil;
        _lastError = nil;
        _cancelGeneration = false;
        _busy = false;
        _pendingUnload = false;
        _pendingUnloadCompletions = [NSMutableArray array];
        _model = std::make_unique<littlebit::Qwen3Model>();
    }
    return self;
}

- (BOOL)verifyModelDirectory:(NSString*)modelDirPath error:(NSError**)error {
    NSFileManager* fm = [NSFileManager defaultManager];
    NSString* weightsPath = [modelDirPath stringByAppendingPathComponent:@"model.safetensors"];
    NSString* configPath = [modelDirPath stringByAppendingPathComponent:@"config.json"];

    if (![fm fileExistsAtPath:weightsPath]) {
        if (error) {
            *error = [NSError errorWithDomain:@"LittleBitError" 
                                         code:101 
                                     userInfo:@{NSLocalizedDescriptionKey: @"model.safetensors is missing."}];
        }
        return NO;
    }
    if (![fm fileExistsAtPath:configPath]) {
        if (error) {
            *error = [NSError errorWithDomain:@"LittleBitError" 
                                         code:102 
                                     userInfo:@{NSLocalizedDescriptionKey: @"config.json is missing."}];
        }
        return NO;
    }
    return YES;
}

// Main-queue only: clear_cache() and releasing the old Qwen3Model are what
// actually close the mmap-backed SafeTensorsFile.
- (void)teardownModelOnMainQueue {
    _model->clear_cache();
    _model = std::make_unique<littlebit::Qwen3Model>();
    _pendingUTF8.clear();
    self.state = EngineStateUnloaded;
    self.currentModelPath = nil;
    self.lastError = nil;
    _cancelGeneration = false;
}

// Drains every unload requested while an operation held _busy. The mutex keeps
// the pending flag and completion list consistent with unloadModelWithCompletion
// even if that call arrives while a main-queue completion is already running.
- (void)drainPendingUnloadOnMainQueue {
    while (true) {
        NSArray* completions = nil;
        {
            std::lock_guard<std::mutex> lock(_lifecycleMutex);
            if (!_pendingUnload.exchange(false)) {
                break;
            }
            completions = [_pendingUnloadCompletions copy];
            [_pendingUnloadCompletions removeAllObjects];
        }
        [self teardownModelOnMainQueue];
        for (dispatch_block_t block in completions) {
            if (block) block();
        }
    }
}

// Clears _busy only after all pending teardown and callbacks are done. The
// final pending check and gate release share one lock, so a late unload cannot
// slip between them.
- (void)finishBusyOperationOnMainQueue {
    while (true) {
        [self drainPendingUnloadOnMainQueue];
        BOOL pending = NO;
        {
            std::lock_guard<std::mutex> lock(_lifecycleMutex);
            pending = _pendingUnload.load();
            if (!pending) {
                _busy = false;
            }
        }
        if (!pending) {
            break;
        }
    }
}

- (void)loadModelAtPath:(NSString*)modelDirPath 
             maxSeqLen:(NSInteger)maxSeqLen 
            completion:(void (^)(BOOL success, NSString* _Nullable errorMsg))completion {
    BOOL rejected = NO;
    {
        std::lock_guard<std::mutex> lock(_lifecycleMutex);
        if (_pendingUnload.load() || _busy.exchange(true)) {
            rejected = YES;
        }
    }
    if (rejected) {
        if (completion) completion(NO, @"Another model operation is in progress.");
        return;
    }

    self.state = EngineStateLoading;
    self.lastError = nil;

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        std::string cppPath = [modelDirPath UTF8String];
        bool ok = self->_model->load(cppPath, static_cast<size_t>(maxSeqLen));

        dispatch_async(dispatch_get_main_queue(), ^{
            BOOL deferredUnload = NO;
            {
                std::lock_guard<std::mutex> lock(_lifecycleMutex);
                deferredUnload = _pendingUnload.load();
            }
            if (deferredUnload) {
                [self drainPendingUnloadOnMainQueue];
                if (completion) completion(NO, @"Model load was cancelled by unload.");
            } else if (ok) {
                self.state = EngineStateReady;
                self.currentModelPath = modelDirPath;
                if (completion) completion(YES, nil);
            } else {
                self.state = EngineStateError;
                self.lastError = @"Failed to load model weights or config files.";
                if (completion) completion(NO, self.lastError);
            }
            // Also drains an unload requested by the completion callback.
            [self finishBusyOperationOnMainQueue];
        });
    });
}

- (void)unloadModel {
    [self unloadModelWithCompletion:nil];
}

- (void)unloadModelWithCompletion:(void (^)(void))completion {
    BOOL wasBusy = NO;
    {
        std::lock_guard<std::mutex> lock(_lifecycleMutex);
        wasBusy = _busy.exchange(true);
        _cancelGeneration = true;
        _pendingUnload = true;
        if (completion) [_pendingUnloadCompletions addObject:[completion copy]];
    }
    if (wasBusy) {
        // Never free the model while a background forward pass or load is
        // using it. The owning operation drains this on the main queue.
        return;
    }

    dispatch_async(dispatch_get_main_queue(), ^{
        [self drainPendingUnloadOnMainQueue];
        [self finishBusyOperationOnMainQueue];
    });
}

- (void)generatePrompt:(NSString*)prompt
           temperature:(float)temperature
                  topP:(float)topP
          maxNewTokens:(NSInteger)maxNewTokens
               onToken:(BOOL (^)(NSString* tokenStr))onToken
            completion:(void (^)(EnginePerformanceMetrics* metrics))completion {
    BOOL rejected = NO;
    {
        std::lock_guard<std::mutex> lock(_lifecycleMutex);
        if (self.state != EngineStateReady || _pendingUnload.load() || _busy.exchange(true)) {
            rejected = YES;
        }
    }
    if (rejected) {
        if (completion) {
            EnginePerformanceMetrics* empty = [[EnginePerformanceMetrics alloc] init];
            completion(empty);
        }
        return;
    }

    self.state = EngineStateGenerating;
    _cancelGeneration = false;
    _pendingUTF8.clear();

    // Allocate in method scope so the main-queue block only captures a plain
    // Objective-C object pointer, never a local declared inside another block.
    EnginePerformanceMetrics* res = [[EnginePerformanceMetrics alloc] init];

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        std::string cppPrompt = [prompt UTF8String];

        auto cppCallback = [self, onToken](int32_t /*token_id*/, const std::string& token_str) -> bool {
            if (self->_cancelGeneration.load()) {
                return false;
            }
            __block BOOL keep_going = YES;
            std::string decoded = append_utf8_stream(_pendingUTF8, token_str);
            NSString* nsToken = [[NSString alloc] initWithBytes:decoded.data()
                                                         length:decoded.size()
                                                       encoding:NSUTF8StringEncoding];
            dispatch_sync(dispatch_get_main_queue(), ^{
                if (onToken) {
                    keep_going = onToken(nsToken ?: @"");
                }
            });
            return keep_going;
        };

        auto shouldCancel = [self]() -> bool {
            return self->_cancelGeneration.load();
        };

        littlebit::PerformanceMetrics m = self->_model->generate(
            cppPrompt,
            static_cast<size_t>(maxNewTokens),
            temperature,
            topP,
            cppCallback,
            shouldCancel
        );

        res.loadTimeMs = m.load_time_ms;
        res.ttftMs = m.ttft_ms;
        res.tokensPerSecond = m.tokens_per_second;
        res.tokensGenerated = m.tokens_generated;
        res.residentMemoryMB = m.resident_memory_mb;
        res.kvCacheMB = m.kv_cache_mb;

        _pendingUTF8.clear();
        dispatch_async(dispatch_get_main_queue(), ^{
            BOOL deferredUnload = NO;
            {
                std::lock_guard<std::mutex> lock(_lifecycleMutex);
                deferredUnload = _pendingUnload.load();
            }
            if (deferredUnload) {
                [self drainPendingUnloadOnMainQueue];
            } else {
                self.state = EngineStateReady;
            }
            // Swift checks bridge.state before forcing .ready in its UI model.
            if (completion) completion(res);
            // Gate must stay closed until after deferred teardown and callbacks.
            [self finishBusyOperationOnMainQueue];
        });
    });
}

- (void)stopGeneration {
    _cancelGeneration = true;
}

- (NSInteger)currentResidentMemoryMB {
    return static_cast<NSInteger>(littlebit::Qwen3Model::get_current_rss_mb());
}

@end
