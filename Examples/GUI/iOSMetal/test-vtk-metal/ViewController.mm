#import "ViewController.h"
#import "VTKMetalBaseViewController.h"
#import "CubeViewController.h"
#import "ConeViewController.h"
#import "WaveletVolumeViewController.h"
#import "DICOMVolumeViewController.h"
#import "NIFTIVolumeViewController.h"

@implementation ViewController

- (void)viewDidLoad
{
  [super viewDidLoad];
  self.selectedIndex = 0;
}

- (id)viewControllerAtIndex:(NSInteger)index
{
  switch (index)
  {
    case 0: return [[CubeViewController alloc] init];
    case 1: return [[ConeViewController alloc] init];
    case 2: return [[WaveletVolumeViewController alloc] init];
    case 3: return [[DICOMVolumeViewController alloc] init];
    case 4: return [[NIFTIVolumeViewController alloc] init];
    case 5: return [[VTKMetalBaseViewController alloc] init];
  }
  return nil;
}

- (void)setSelectedIndex:(NSInteger)selectedIndex
{
  _selectedIndex = selectedIndex;

  for (id child in self.childViewControllers)
  {
#if TARGET_OS_OSX
    NSViewController *childVC = child;
#else
    UIViewController *childVC = child;
    [childVC willMoveToParentViewController:nil];
#endif
    [childVC.view removeFromSuperview];
    [childVC removeFromParentViewController];
  }

#if TARGET_OS_OSX
  NSViewController *vc = [self viewControllerAtIndex:selectedIndex];
  vc.view.frame = self.view.bounds;
  vc.view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  [self addChildViewController:vc];
  [self.view addSubview:vc.view];
#else
  UIViewController *vc = [self viewControllerAtIndex:selectedIndex];
  vc.view.frame = self.view.bounds;
  vc.view.autoresizingMask =
    UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  [self.view addSubview:vc.view];
  [self addChildViewController:vc];
  [vc didMoveToParentViewController:self];
#endif
}

- (id)currentViewController
{
  return self.childViewControllers.lastObject;
}

@end
