//
//  VolumeRenderingPreset.m
//  Eyesight
//
//  Created by lyrae on 06/10/23.
//

#import "VolumeRenderingPreset.h"

@implementation ColorTransferFunctionMember

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    self = [super init];
    if (self) {
        _red = [dictionary[@"red"] doubleValue];
        _green = [dictionary[@"green"] doubleValue];
        _blue = [dictionary[@"blue"] doubleValue];
    }
    return self;
}

@end

@implementation OpacityTransferFunctionMember

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    self = [super init];
    if (self) {
        _x = [dictionary[@"x"] doubleValue];
        _y = [dictionary[@"y"] doubleValue];
    }
    return self;
}

@end

@implementation VolumeRenderingPreset

- (nullable instancetype)initWithContentsOfURL:(NSURL *)url {
    NSDictionary *dictionary = [NSDictionary dictionaryWithContentsOfURL:url];
    if (!dictionary) return nil;
    return [self initWithDictionary:dictionary];
}

- (instancetype)initWithDictionary:(NSDictionary *)dictionary {
    self = [super init];
    if (self) {
        if (!dictionary[@"16bitClutColors"] || !dictionary[@"16bitClutCurves"] || !dictionary[@"name"]) {
            return nil;
        }
        _name = [dictionary[@"name"] copy];
        _useShading = [dictionary[@"useShading"] boolValue];
        _ww = [dictionary[@"ww"] doubleValue];
        _wl = [dictionary[@"wl"] doubleValue];

        NSMutableArray *colorFunctions = [NSMutableArray array];
        for (NSArray *membersArray in dictionary[@"16bitClutColors"]) {
            NSMutableArray *members = [NSMutableArray array];
            for (NSDictionary *memberDict in membersArray) {
                [members addObject:[[ColorTransferFunctionMember alloc] initWithDictionary:memberDict]];
            }
            [colorFunctions addObject:members];
        }
        _colorTransferFunctions = [colorFunctions copy];

        NSMutableArray *opacityFunctions = [NSMutableArray array];
        for (NSArray *membersArray in dictionary[@"16bitClutCurves"]) {
            NSMutableArray *members = [NSMutableArray array];
            for (NSDictionary *memberDict in membersArray) {
                [members addObject:[[OpacityTransferFunctionMember alloc] initWithDictionary:memberDict]];
            }
            [opacityFunctions addObject:members];
        }
        _opacityTransferFunctions = [opacityFunctions copy];
    }
    return self;
}

@end
