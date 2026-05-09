#!/bin/bash

set -e

export PATH="~/Developer/Swift/Frameworks/gdcm/:$PATH"

echo "Step 1: Removing build folder..."
rm -rf build

echo "Step 2: Removing Remote/vtkDICOM folder..."
rm -rf Remote/vtkDICOM

echo "Step 3: Cloning vtk-dicom..."
mkdir -p Remote
git clone https://github.com/dgobbi/vtk-dicom.git Remote/vtkDICOM

echo "Step 4: Running CMake..."
cmake -B build -DVTK_IOS_BUILD:BOOL=ON \
    -DIOS_DEVICE_ARCHITECTURES:STRING=arm64 \
    -DIOS_SIMULATOR_ARCHITECTURES:STRING=arm64 \
    -DIOS_EMBED_BITCODE:BOOL=ON \
    -DIOS_DEPLOYMENT_TARGET:STRING=17.5 \
    -DModule_vtkDICOM:BOOL=ON \
    -DVTK_MODULE_ENABLE_VTK_vtkDICOM:STRING=YES \
    -DVTK_MODULE_ENABLE_VTK_DICOM:STRING=YES \
    -DVTK_MODULE_ENABLE_VTK_vtkIOSQL:STRING=YES \
    -DBUILD_DICOM_PROGRAMS:BOOL=ON \
    -DUSE_GDCM:BOOL=ON \
    -DUSE_ITK_GDCM:BOOL=OFF \
    -DVTK_MODULE_ENABLE_VTK_RenderingOpenGL2:BOOL=ON \
    -DVTK_MODULE_ENABLE_VTK_RenderingVolumeOpenGL2:BOOL=ON \
    -DUSE_DCMTK:BOOL=OFF \
    -DCMAKE_FRAMEWORK_INSTALL_PREFIX:PATH=frameworks \
    -DVTK_BUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DVTK_MODULE_ENABLE_VTK_RenderingImage:BOOL=ON

echo "Step 5: Building..."
cmake --build build -j8

echo "Done!"