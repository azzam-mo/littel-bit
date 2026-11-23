#pragma once

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, EngineState) {
    EngineStateUnloaded,
    EngineStateLoading,
    EngineStateReady,
    EngineStateGenerating,
    EngineStateError
};

@interface EnginePerformanceMetrics : NSObject
@property (nonatomic, assign) double loadTimeMs;
@property (nonatomic, assign) double ttftMs;
@property (nonatomic, assign) double tokensPerSecond;
@property (nonatomic, assign) NSInteger tokensGenerated;
@property (nonatomic, assign) NSInteger residentMemoryMB;
@property (nonatomic, assign) NSInteger kvCacheMB;
@end

@interface LittleBitEngineBridge : NSObject

@property (nonatomic, readonly) EngineState state;
@property (nonatomic, readonly, nullable) NSString* currentModelPath;
@property (nonatomic, readonly, nullable) NSString* lastError;

+ (instancetype)sharedInstance;

- (BOOL)verifyModelDirectory:(NSString*)modelDirPath error:(NSError**)error;
- (void)loadModelAtPath:(NSString*)modelDirPath 
             maxSeqLen:(NSInteger)maxSeqLen 
            completion:(void (^)(BOOL success, NSString* _Nullable errorMsg))completion;

- (void)unloadModel;

- (void)generatePrompt:(NSString*)prompt
           temperature:(float)temperature
                  topP:(float)topP
          maxNewTokens:(NSInteger)maxNewTokens
               onToken:(BOOL (^)(NSString* tokenStr))onToken
            completion:(void (^)(EnginePerformanceMetrics* metrics))completion;

- (void)stopGeneration;

- (NSInteger)currentResidentMemoryMB;

@end

NS_ASSUME_NONNULL_END
