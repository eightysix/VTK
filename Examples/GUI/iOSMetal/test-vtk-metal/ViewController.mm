#import "ViewController.h"
#import "VTKMetalBaseViewController.h"
#import "CubeViewController.h"
#import "ConeViewController.h"
#import "VolumeViewController.h"
#import "DICOMVolumeViewController.h"
#import "NIFTIVolumeViewController.h"

@implementation ViewController

- (void)viewDidLoad
{
  [super viewDidLoad];
  self.selectedIndex = 0;
}

- (UIViewController*)viewControllerAtIndex:(NSInteger)index
{
  switch (index)
  {
    case 0: return [[CubeViewController alloc] init];
    case 1: return [[ConeViewController alloc] init];
    case 2: return [[VolumeViewController alloc] init];
    case 3: return [[DICOMVolumeViewController alloc] init];
    case 4: return [[NIFTIVolumeViewController alloc] init];
    case 5: return [[VTKMetalBaseViewController alloc] init];
  }
  return nil;
}

- (void)setSelectedIndex:(NSInteger)selectedIndex
{
  _selectedIndex = selectedIndex;
  
  for (UIViewController* child in self.childViewControllers)
  {
    [child willMoveToParentViewController:nil];
    [child.view removeFromSuperview];
    [child removeFromParentViewController];
  }
  
  UIViewController* vc = [self viewControllerAtIndex:selectedIndex];
  vc.view.frame = self.view.bounds;
  vc.view.autoresizingMask =
  UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  [self.view addSubview:vc.view];
  [self addChildViewController:vc];
  [vc didMoveToParentViewController:self];
}

- (UIViewController*)currentViewController
{
  return self.childViewControllers.lastObject;
}

@end
