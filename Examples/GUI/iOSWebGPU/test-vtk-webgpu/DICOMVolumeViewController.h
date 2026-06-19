#import "VTKBaseViewController.h"

@interface DICOMVolumeViewController : VTKBaseViewController <UIDocumentPickerDelegate>
@property (nonatomic, copy) NSString *dicomDirectoryPath;
- (void)loadDICOMFolder;
@end
