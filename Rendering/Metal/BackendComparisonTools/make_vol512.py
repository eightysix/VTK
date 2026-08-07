#!/usr/bin/env python3
"""Regenerate /tmp/bc/vol512.npy (512^3 headsq array) from the ExternalData tree.

Requires: pip install vtk numpy
Usage:    python3 make_vol512.py <headsq-prefix> <out.npy>
Example:  python3 make_vol512.py \
            /path/build_macos_metal/ExternalData/Testing/Data/headsq/quarter \
            /tmp/bc/vol512.npy
"""
import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

prefix, out = sys.argv[1], sys.argv[2]

reader = vtk.vtkVolume16Reader()
reader.SetDataDimensions(64, 64)
reader.SetImageRange(1, 93)
reader.SetDataByteOrderToLittleEndian()
reader.SetFilePrefix(prefix)
reader.SetDataSpacing(3.2, 3.2, 1.5)

resize = vtk.vtkImageResize()
resize.SetInputConnection(reader.GetOutputPort())
resize.SetResizeMethodToOutputDimensions()
resize.SetOutputDimensions(512, 512, 512)
resize.Update()

out_img = resize.GetOutput()
arr = vtk_to_numpy(out_img.GetPointData().GetScalars()).reshape(512, 512, 512)
np.save(out, arr)
print(f'wrote {out}: shape={arr.shape} dtype={arr.dtype} range=[{arr.min()}, {arr.max()}] '
      f'spacing={tuple(out_img.GetSpacing())}')
