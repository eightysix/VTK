//
//  VolumeRenderingPresetsManager.h
//  Eyesight
//
//  Created by Marco  Albera on 06/10/23.
//

#import "VolumeRenderingPreset.h"
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface VolumeRenderingPresetsManager : NSObject

@property(class, nonatomic, readonly, nullable) NSArray<NSURL *> *presetsURLs;
@property(class, nonatomic, readonly, nullable) NSArray<VolumeRenderingPreset *> *presets;
@property(class, nonatomic, readonly, nullable) VolumeRenderingPreset *firstVolumeRenderingPreset;

@end

NS_ASSUME_NONNULL_END
