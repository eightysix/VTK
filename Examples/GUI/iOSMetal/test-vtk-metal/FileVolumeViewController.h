#import "BaseVolumeViewController.h"

@class VolumeRenderingPreset;

@interface FileVolumeViewController : BaseVolumeViewController <UIDocumentPickerDelegate>

- (NSArray<NSString *> *)documentTypes;
- (void)loadFromURL:(NSURL *)url;

@property (nonatomic, readonly, nullable) VolumeRenderingPreset *currentPreset;
@property (nonatomic, readonly) NSInteger currentPresetIndex;

- (void)applyCurrentPreset;
- (IBAction)nextPreset:(nullable id)sender;
- (IBAction)previousPreset:(nullable id)sender;
- (IBAction)loadFile:(nullable id)sender;

@end
