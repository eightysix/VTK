#import "VTKMetalBaseViewController.h"

@interface FileVolumeViewController : VTKMetalBaseViewController <UIDocumentPickerDelegate>

- (NSString *)loadButtonTitle;
- (NSArray<NSString *> *)documentTypes;
- (void)loadFromURL:(NSURL *)url;

@end
