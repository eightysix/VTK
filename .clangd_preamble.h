// clangd preamble — force-included before every file so standalone headers
// have access to Foundation/UIKit/Metal types (NSObject, NSInteger, etc.)
// The __OBJC__ guard keeps this a no-op for pure-C translation units.
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#endif
