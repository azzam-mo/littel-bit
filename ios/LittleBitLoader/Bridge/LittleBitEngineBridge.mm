#import "LittleBitEngineBridge.h"
#include "littlebit/model.h"

#include <atomic>
#include <memory>
#include <string>

@implementation EnginePerformanceMetrics
@end

@interface LittleBitEngineBridge () {
    std::unique_ptr<littlebit::Qwen3Model> _model;
    std::atomic<bool> _cancelGeneration;
}
@property (nonatomic, readwrite) EngineState state;
@property (nonatomic, readwrite, nullable) NSString* currentModelPath;
@property (nonatomic, readwrite, nullable) NSString* lastError;
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

- (void)loadModelAtPath:(NSString*)modelDirPath 
             maxSeqLen:(NSInteger)maxSeqLen 
            completion:(void (^)(BOOL success, NSString* _Nullable errorMsg))completion {
    self.state = EngineStateLoading;
    self.lastError = nil;

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^{
        std::string cppPath = [modelDirPath UTF8String];
        bool ok = self->_model->load(cppPath, static_cast<size_t>(maxSeqLen));

        dispatch_async(dispatch_get_main_queue(), ^{
            if (ok) {
                self.state = EngineStateReady;
                self.currentModelPath = modelDirPath;
                completion(YES, nil);
            } else {
                self.state = EngineStateError;
                self.lastError = @"Failed to load model weights or config files.";
                completion(NO, self.lastError);
            }
        });
    });
}

- (void)unloadModel {
    _model->clear_cache();
    _model = std::make_unique<littlebit::Qwen3Model>();
    self.state = EngineStateUnloaded;
    self.currentModelPath = nil;
    self.lastError = nil;
}

- (void)generatePrompt:(NSString*)prompt
           temperature:(float)temperature
                  topP:(float)topP
          maxNewTokens:(NSInteger)maxNewTokens
               onToken:(BOOL (^)(NSString* tokenStr))onToken
            completion:(void (^)(EnginePerformanceMetrics* metrics))completion {
    if (self.state != EngineStateReady) {
        return;
    }

    self.state = EngineStateGenerating;
    _cancelGeneration = false;

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_USER_INITIATED, 0), ^{
        std::string cppPrompt = [prompt UTF8String];

        auto cppCallback = [self, onToken](int32_t /*token_id*/, const std::string& token_str) -> bool {
            if (self->_cancelGeneration.load()) {
                return false;
            }
            __block BOOL keep_going = YES;
            dispatch_sync(dispatch_get_main_queue(), ^{
                NSString* nsToken = [NSString stringWithUTF8String:token_str.c_str()];
                if (onToken) {
                    keep_going = onToken(nsToken ?: @"");
                }
            });
            return keep_going;
        };

        littlebit::PerformanceMetrics m = self->_model->generate(
            cppPrompt,
            static_cast<size_t>(maxNewTokens),
            temperature,
            topP,
            cppCallback
        );

        dispatch_async(dispatch_get_main_queue(), ^{
            self.state = EngineStateReady;
            EnginePerformanceMetrics* res = [[EnginePerformanceMetrics alloc] init];
            res.loadTimeMs = m.load_time_ms;
            res.ttftMs = m.ttft_ms;
            res.tokensPerSecond = m.tokens_per_second;
            res.tokensGenerated = m.tokens_generated;
            res.residentMemoryMB = m.resident_memory_mb;
            res.kvCacheMB = m.kv_cache_mb;
            if (completion) completion(res);
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
