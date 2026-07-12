#import "VTKMetalBaseViewController.h"

@interface NIFTIVolumeViewController : VTKMetalBaseViewController <UIDocumentPickerDelegate>
@property (nonatomic, copy) NSString *niftiFilePath;
- (void)loadNIFTIFile;
@end
