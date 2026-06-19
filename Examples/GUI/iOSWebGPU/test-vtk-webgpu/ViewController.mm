#import "ViewController.h"
#import "ConeViewController.h"
#import "CubeViewController.h"

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

  UITabBarController *tabBar = [[UITabBarController alloc] init];
  tabBar.viewControllers = @[ coneVC, cubeVC ];
  tabBar.view.frame = self.view.bounds;
  tabBar.view.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  [self.view addSubview:tabBar.view];
  [self addChildViewController:tabBar];
  [tabBar didMoveToParentViewController:self];
}

@end
