set(classes
  vtkMetalRenderWindow
  vtkMetalRenderer
  vtkMetalActor
  vtkMetalPolyDataMapper
  vtkMetalCamera
  vtkMetalProperty
  vtkMetalLight)

set(headers)
set(sources)

# All Metal sources use .mm (Objective-C++) for Metal API access
foreach(class IN LISTS classes)
  list(APPEND sources ${class}.mm)
  list(APPEND headers ${class}.h)
endforeach()

# Platform-specific classes
if(APPLE_IOS)
  vtk_object_factory_declare(
    BASE vtkRenderWindow
    OVERRIDE vtkIOSMetalRenderWindow)
  list(APPEND sources vtkIOSMetalRenderWindow.mm)
  list(APPEND headers vtkIOSMetalRenderWindow.h)
endif()

# Metal shader compilation - embed shader source as C++ string constant
vtk_encode_string(
  INPUT "Shaders/MetalShaders.metal"
  NAME "vtkMetalShaders"
  EXPORT_SYMBOL "VTKRENDERINGMETAL_EXPORT"
  EXPORT_HEADER "vtkRenderingMetalModule.h"
  HEADER_OUTPUT metal_shader_header
  SOURCE_OUTPUT metal_shader_source)
list(APPEND sources "${metal_shader_source}")
list(APPEND headers "${metal_shader_header}")

vtk_object_factory_configure(
  SOURCE_FILE vtk_object_factory_source
  HEADER_FILE vtk_object_factory_header
  EXPORT_MACRO "VTKRENDERINGMETAL_EXPORT"
  EXTRA_INCLUDES
  "<vtkObjectFactoryCollection.h>"
  "<vtkLogger.h>")

vtk_module_add_module(
  VTK::RenderingMetal
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
