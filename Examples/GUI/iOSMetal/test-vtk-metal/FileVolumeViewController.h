#import "VTKMetalBaseViewController.h"

@class VolumeRenderingPreset;

@interface FileVolumeViewController : VTKMetalBaseViewController <UIDocumentPickerDelegate>

- (NSString *)loadButtonTitle;
- (NSArray<NSString *> *)documentTypes;
- (void)loadFromURL:(NSURL *)url;

@property (nonatomic, readonly, nullable) VolumeRenderingPreset *currentPreset;
@property (nonatomic, readonly) NSInteger currentPresetIndex;

- (IBAction)nextPreset:(nullable id)sender;
- (IBAction)previousPreset:(nullable id)sender;

@end
