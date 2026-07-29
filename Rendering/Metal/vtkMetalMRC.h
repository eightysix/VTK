// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
// SPDX-License-Identifier: BSD-3-Clause
#ifndef vtkMetalMRC_h
#define vtkMetalMRC_h

// MRC ownership helpers for Objective-C manual retain/release.
// All vtkMetal* .mm files compiled with -fno-objc-arc should include this header.

#import <Foundation/Foundation.h>

namespace vtkMetalMRC
{

template <typename T>
inline void ReleaseAndNil(T& obj)
{
    if (obj)
    {
        [obj release];
        obj = nil;
    }
}

template <typename T>
inline void AssignRetained(T& dst, T src)
{
    if (dst != src)
    {
        [src retain];
        [dst release];
        dst = src;
    }
}

template <typename T>
inline void AssignConsumed(T& dst, T src)
{
    if (dst != src)
    {
        [dst release];
        dst = src;
    }
    else
    {
        [src release];
    }
}

} // namespace vtkMetalMRC

#endif
