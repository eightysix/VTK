//
//  VolumeRenderingPresetsManager.m
//  Eyesight
//
//  Created by Marco  Albera on 06/10/23.
//

#import "VolumeRenderingPresetsManager.h"

@implementation VolumeRenderingPresetsManager

+ (NSArray<NSURL *> *)presetsURLs {
    static NSArray<NSURL *> *_presetsURLs = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSURL *resourceDirectory = [[NSBundle mainBundle] resourceURL];
        if (!resourceDirectory) return;

        // Look in VRPresets subdirectory first, fall back to bundle root
        NSURL *vrPresetsDir = [resourceDirectory URLByAppendingPathComponent:@"VRPresets"];
        NSURL *searchDir = [[NSFileManager defaultManager] fileExistsAtPath:vrPresetsDir.path]
                               ? vrPresetsDir
                               : resourceDirectory;

        NSError *error = nil;
        NSArray<NSURL *> *directoryContents = [[NSFileManager defaultManager] contentsOfDirectoryAtURL:searchDir
                                                                            includingPropertiesForKeys:@[ NSURLIsRegularFileKey ]
                                                                                               options:0
                                                                                                 error:&error];
        if (error) return;

        NSPredicate *predicate = [NSPredicate predicateWithFormat:@"pathExtension == 'plist'"];
        _presetsURLs = [[directoryContents filteredArrayUsingPredicate:predicate] sortedArrayUsingComparator:^NSComparisonResult(NSURL *url1, NSURL *url2) {
            return [url1.lastPathComponent localizedStandardCompare:url2.lastPathComponent];
        }];
    });
    return _presetsURLs;
}

+ (NSArray<VolumeRenderingPreset *> *)presets {
    static NSArray<VolumeRenderingPreset *> *_presets = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSArray<NSURL *> *urls = [self presetsURLs];
        if (!urls) return;

        NSMutableArray *presets = [NSMutableArray array];
        for (NSURL *url in urls) {
            VolumeRenderingPreset *preset = [[VolumeRenderingPreset alloc] initWithContentsOfURL:url];
            if (preset) {
                [presets addObject:preset];
            }
        }
        _presets = [presets copy];
    });
    return _presets;
}

+ (VolumeRenderingPreset *)firstVolumeRenderingPreset {
    return [self presets].firstObject;
}

@end
