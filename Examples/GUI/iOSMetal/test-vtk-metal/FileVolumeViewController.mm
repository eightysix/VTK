#import "FileVolumeViewController.h"

#include "vtkVolume.h"
#include "vtkMetalRenderer.h"
#include "vtkIOSMetalRenderWindow.h"

@implementation FileVolumeViewController

- (NSString *)loadButtonTitle
{
  return @"Load";
}

- (NSArray<NSString *> *)documentTypes
{
  return @[ @"public.data" ];
}

- (void)loadFromURL:(NSURL *)url
{
  // Subclasses must override
}

- (void)viewDidLoad
{
  [super viewDidLoad];

  UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
  [button setTitle:[self loadButtonTitle] forState:UIControlStateNormal];
  button.titleLabel.font = [UIFont boldSystemFontOfSize:18];
  button.backgroundColor = [[UIColor blackColor] colorWithAlphaComponent:0.5];
  button.layer.cornerRadius = 8;
  button.contentEdgeInsets = UIEdgeInsetsMake(12, 24, 12, 24);
  [button setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
  [button addTarget:self action:@selector(loadFile) forControlEvents:UIControlEventTouchUpInside];
  button.translatesAutoresizingMaskIntoConstraints = NO;
  [self.view addSubview:button];

  [NSLayoutConstraint activateConstraints:@[
    [button.centerXAnchor constraintEqualToAnchor:self.view.centerXAnchor],
    [button.bottomAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor constant:-20]
  ]];
}

- (void)loadFile
{
  UIDocumentPickerViewController *picker =
      [[UIDocumentPickerViewController alloc] initWithDocumentTypes:[self documentTypes]
                                                              inMode:UIDocumentPickerModeOpen];
  picker.delegate = self;
  picker.allowsMultipleSelection = NO;
  [self presentViewController:picker animated:YES completion:nil];
}

- (void)documentPicker:(UIDocumentPickerViewController *)controller
    didPickDocumentsAtURLs:(NSArray<NSURL *> *)urls
{
  if (urls.count == 0) {
    return;
  }

  NSURL *url = urls.firstObject;
  BOOL coordinated = [url startAccessingSecurityScopedResource];

  [self loadFromURL:url];

  if (coordinated) {
    [url stopAccessingSecurityScopedResource];
  }
}

- (void)documentPickerWasCancelled:(UIDocumentPickerViewController *)controller
{
}

- (void)setupVTKPipeline
{
}

@end
