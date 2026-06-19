#import "ViewController.h"
#import "ConeViewController.h"
#import "CubeViewController.h"
#import "VolumeViewController.h"
#import "DICOMVolumeViewController.h"

@implementation ViewController

- (void)viewDidLoad {
  [super viewDidLoad];

  ConeViewController *coneVC = [[ConeViewController alloc] init];
  coneVC.tabBarItem = [[UITabBarItem alloc] initWithTitle:@"Cone"
                                                    image:nil
                                                      tag:0];

  CubeViewController *cubeVC = [[CubeViewController alloc] init];
  cubeVC.tabBarItem = [[UITabBarItem alloc] initWithTitle:@"Cube"
                                                    image:nil
                                                      tag:1];

  VolumeViewController *volumeVC = [[VolumeViewController alloc] init];
  volumeVC.tabBarItem = [[UITabBarItem alloc] initWithTitle:@"Volume"
                                                     image:nil
                                                       tag:2];

  DICOMVolumeViewController *dicomVC = [[DICOMVolumeViewController alloc] init];
  dicomVC.tabBarItem = [[UITabBarItem alloc] initWithTitle:@"DICOM"
                                                    image:nil
                                                      tag:3];

  UITabBarController *tabBar = [[UITabBarController alloc] init];
  tabBar.viewControllers = @[ coneVC, cubeVC, volumeVC, dicomVC ];
  tabBar.view.frame = self.view.bounds;
  tabBar.view.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  [self.view addSubview:tabBar.view];
  [self addChildViewController:tabBar];
  [tabBar didMoveToParentViewController:self];
}

@end
