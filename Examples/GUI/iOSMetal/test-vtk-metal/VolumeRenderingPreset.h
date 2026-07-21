//
//  VolumeRenderingPreset.h
//  Eyesight
//
//  Created by lyrae on 06/10/23.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ColorTransferFunctionMember : NSObject

@property(nonatomic, readonly) double red;
@property(nonatomic, readonly) double green;
@property(nonatomic, readonly) double blue;

- (instancetype)initWithDictionary:(NSDictionary *)dictionary;

@end

@interface OpacityTransferFunctionMember : NSObject

@property(nonatomic, readonly) double x;
@property(nonatomic, readonly) double y;

- (instancetype)initWithDictionary:(NSDictionary *)dictionary;

@end

@interface VolumeRenderingPreset : NSObject

@property(nonatomic, copy, readonly) NSString *name;
@property(nonatomic, copy, readonly) NSArray<NSArray<ColorTransferFunctionMember *> *> *colorTransferFunctions;
@property(nonatomic, copy, readonly) NSArray<NSArray<OpacityTransferFunctionMember *> *> *opacityTransferFunctions;
@property(nonatomic, readonly) BOOL useShading;
@property(nonatomic, readonly) double ww;
@property(nonatomic, readonly) double wl;

- (nullable instancetype)initWithContentsOfURL:(NSURL *)url;
- (instancetype)initWithDictionary:(NSDictionary *)dictionary;

@end

NS_ASSUME_NONNULL_END
