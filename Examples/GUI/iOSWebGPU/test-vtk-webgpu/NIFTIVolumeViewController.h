#import "VTKBaseViewController.h"

@interface NIFTIVolumeViewController : VTKBaseViewController <UIDocumentPickerDelegate>
@property (nonatomic, copy) NSString *niftiFilePath;
- (void)loadNIFTIFile;
@end
