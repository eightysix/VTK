#import "ViewController.h"
#import "VTKMetalBaseViewController.h"
#import "CubeViewController.h"
#import "ConeViewController.h"
#import "VolumeViewController.h"

@implementation ViewController

- (void)viewDidLoad
{
  [super viewDidLoad];

  CubeViewController* cubeVC = [[CubeViewController alloc] init];
  cubeVC.tabBarItem = [[UITabBarItem alloc] initWithTitle:@"Cube"
                                                     image:nil
                                                       tag:0];

  ConeViewController* coneVC = [[ConeViewController alloc] init];
  coneVC.tabBarItem = [[UITabBarItem alloc] initWithTitle:@"Cone"
                                                     image:nil
                                                       tag:1];

  VolumeViewController* volumeVC = [[VolumeViewController alloc] init];
  volumeVC.tabBarItem = [[UITabBarItem alloc] initWithTitle:@"Volume"
                                                       image:nil
                                                         tag:2];

  UITabBarController* tabBar = [[UITabBarController alloc] init];
  tabBar.viewControllers = @[ cubeVC, coneVC, volumeVC ];
  tabBar.view.frame = self.view.bounds;
  tabBar.view.autoresizingMask =
    UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  [self.view addSubview:tabBar.view];
  [self addChildViewController:tabBar];
}

@end
