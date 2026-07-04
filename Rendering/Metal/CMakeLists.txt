set(classes
  vtkMetalRenderWindow
  vtkMetalRenderer
  vtkMetalActor
  vtkMetalPolyDataMapper
  vtkMetalCamera
  vtkMetalProperty
  vtkMetalLight)

set(headers)
set(nowrap_headers)
set(sources)

# Platform-specific classes
if(APPLE_IOS)
  vtk_object_factory_declare(
    BASE vtkRenderWindow
    OVERRIDE vtkIOSMetalRenderWindow)
  list(APPEND classes vtkIOSMetalRenderWindow)
  list(APPEND sources vtkIOSMetalRenderWindow.mm)
  list(APPEND headers vtkIOSMetalRenderWindow.h)
endif()

# Collect header/source pairs
foreach(class IN LISTS classes)
  list(APPEND sources ${class}.cxx)
  list(APPEND headers ${class}.h)
endforeach()

# Metal shader compilation
vtk_encode_string(
  INPUT Shaders/MetalShaders.metal
  HEADER vtkMetalShaders.h
  NAME vtkMetalShaders)
list(APPEND sources ${vtkMetalShaders_SOURCE})
list(APPEND headers ${vtkMetalShaders_HEADER})

vtk_object_factory_configure(
  SOURCE_FILE vtk_object_factory_source
  HEADER_FILE vtk_object_factory_header
  EXPORT_MACRO "VTKRENDERINGMETAL_EXPORT"
  EXTRA_INCLUDES
  "<vtkObjectFactoryCollection.h>"
  "<vtkLogger.h>")

vtk_module_add_module(
  VTK::RenderingMetal
  CLASSES ${classes}
  HEADERS ${headers}
  SOURCES ${sources} ${vtk_object_factory_source}
  PRIVATE_HEADERS ${vtk_object_factory_header})

# Link Metal framework
if(APPLE_IOS)
  vtk_module_link(VTK::RenderingMetal PUBLIC "-framework Metal" "-framework QuartzCore" "-framework UIKit")
  if(VTK_REQUIRED_OBJCXX_FLAGS)
    set_source_files_properties(
      ${sources}
      PROPERTIES COMPILE_FLAGS "${VTK_REQUIRED_OBJCXX_FLAGS}")
  endif()
else()
  vtk_module_link(VTK::RenderingMetal PUBLIC "-framework Metal" "-framework QuartzCore")
endif()

vtk_add_test_mangling(VTK::RenderingMetal)
