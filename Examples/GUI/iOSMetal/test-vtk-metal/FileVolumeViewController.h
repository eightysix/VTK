#import <TargetConditionals.h>
#import "BaseVolumeViewController.h"

#if TARGET_OS_OSX
#import <Cocoa/Cocoa.h>
#else
#import <UIKit/UIKit.h>
#endif

@class VolumeRenderingPreset;

#if TARGET_OS_OSX
@interface FileVolumeViewController : BaseVolumeViewController
#else
@interface FileVolumeViewController : BaseVolumeViewController <UIDocumentPickerDelegate>
#endif

- (NSArray<NSString *> *)documentTypes;
- (void)loadFromURL:(NSURL *)url;

@property (nonatomic, readonly, nullable) VolumeRenderingPreset *currentPreset;
@property (nonatomic, readonly) NSInteger currentPresetIndex;

- (void)applyCurrentPreset;
- (IBAction)nextPreset:(nullable id)sender;
- (IBAction)previousPreset:(nullable id)sender;
- (IBAction)loadFile:(nullable id)sender;

@end
