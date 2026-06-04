#import "AppDelegate.h"

@interface AppDelegate ()
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    NSMenu *menubar = [NSMenu new];
    NSMenuItem *appMenuItem = [[NSMenuItem alloc] initWithTitle:@"vtk-webgpu-demo" action:nil keyEquivalent:@""];
    [menubar addItem:appMenuItem];
    [NSApp setMainMenu:menubar];

    NSMenu *appMenu = [[NSMenu alloc] initWithTitle:@"vtk-webgpu-demo"];
    [appMenu addItemWithTitle:@"Quit vtk-webgpu-demo" action:@selector(terminate:) keyEquivalent:@"q"];
    [appMenuItem setSubmenu:appMenu];

    NSRect frame = NSMakeRect(0, 0, 800, 600);
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    [self.window setTitle:@"vtk-webgpu-demo"];
    [self.window center];

    NSTextField *label = [[NSTextField alloc] initWithFrame:NSMakeRect(50, 280, 700, 40)];
    [label setStringValue:@"VTK WebGPU Demo — Cocoa Test"];
    [label setFont:[NSFont systemFontOfSize:24]];
    [label setBezeled:NO];
    [label setDrawsBackground:NO];
    [label setEditable:NO];
    [label setSelectable:NO];
    [label setAlignment:NSTextAlignmentCenter];
    [[self.window contentView] addSubview:label];

    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)applicationWillTerminate:(NSNotification *)aNotification {
}

- (BOOL)applicationSupportsSecureRestorableState:(NSApplication *)app {
    return YES;
}

@end
