#import "VTKMetalBaseViewController.h"

@interface DICOMVolumeViewController : VTKMetalBaseViewController <UIDocumentPickerDelegate>
@property (nonatomic, copy) NSString *dicomDirectoryPath;
- (void)loadDICOMFolder;
@end
