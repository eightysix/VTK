set(classes
  vtkMetalRenderWindow
  vtkMetalRenderer
  vtkMetalDepthPeeler
  vtkMetalOrderIndependentTranslucentPass
  vtkMetalActor
  vtkMetalPolyDataMapper
  vtkMetalBatchedPolyDataMapper
  vtkMetalCompositePolyDataMapperDelegator
  vtkMetalPolyDataMapper2D
  vtkMetalImageMapper
  vtkMetalCamera
  vtkMetalProperty
  vtkMetalLight
  vtkMetalTexture
  vtkMetalHardwareSelector
  vtkMetalGlyph3DMapper
  vtkMetalGPUVolumeRayCastMapper)

set(headers)
set(sources)

# Standalone headers (not matching a .mm source)
list(APPEND headers vtkMetalPickTypes.h)

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
elseif(APPLE)
  vtk_object_factory_declare(
    BASE vtkRenderWindow
    OVERRIDE vtkCocoaMetalRenderWindow)
  list(APPEND sources vtkCocoaMetalRenderWindow.mm)
  list(APPEND headers vtkCocoaMetalRenderWindow.h)
endif()

# Register the remaining Metal classes as factory overrides for their base
# classes (RenderingCore / RenderingVolume).  Each override carries a
# "RenderingBackend=Metal" override attribute so that the object factory can
# select the Metal classes at runtime via
# `--vtk-factory-prefer RenderingBackend=Metal` (used by the generic
# Rendering/Core test suite, mirroring the OpenGL and WebGPU backends).
vtk_object_factory_declare(
  BASE vtkRenderer
  OVERRIDE vtkMetalRenderer)
vtk_object_factory_declare(
  BASE vtkActor
  OVERRIDE vtkMetalActor)
vtk_object_factory_declare(
  BASE vtkPolyDataMapper
  OVERRIDE vtkMetalPolyDataMapper)
vtk_object_factory_declare(
  BASE vtkCompositePolyDataMapperDelegator
  OVERRIDE vtkMetalCompositePolyDataMapperDelegator)
vtk_object_factory_declare(
  BASE vtkGlyph3DMapper
  OVERRIDE vtkMetalGlyph3DMapper)
vtk_object_factory_declare(
  BASE vtkPolyDataMapper2D
  OVERRIDE vtkMetalPolyDataMapper2D)
vtk_object_factory_declare(
  BASE vtkImageMapper
  OVERRIDE vtkMetalImageMapper)
vtk_object_factory_declare(
  BASE vtkCamera
  OVERRIDE vtkMetalCamera)
vtk_object_factory_declare(
  BASE vtkLight
  OVERRIDE vtkMetalLight)
vtk_object_factory_declare(
  BASE vtkProperty
  OVERRIDE vtkMetalProperty)
vtk_object_factory_declare(
  BASE vtkTexture
  OVERRIDE vtkMetalTexture)
vtk_object_factory_declare(
  BASE vtkHardwareSelector
  OVERRIDE vtkMetalHardwareSelector)
vtk_object_factory_declare(
  BASE vtkGPUVolumeRayCastMapper
  OVERRIDE vtkMetalGPUVolumeRayCastMapper)

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

# Color read-back (GetPixelData / image-baseline regression tests) requires the
# CAMetalLayer drawable to be blittable (framebufferOnly=NO) and copies the
# whole frame into a shared texture every render. Keep that overhead out of
# production builds by compiling it in only when the test suites are enabled.
# The offscreen render target is likewise test-only: it lets the benchmark
# harness render into a private texture instead of a CAMetalLayer drawable (whose
# nextDrawable is paced to the display refresh cadence on recent macOS), so the
# Metal timing is not throttled by the compositor.
if(VTK_BUILD_TESTING)
  target_compile_definitions(vtkRenderingMetal PRIVATE
    VTK_METAL_ENABLE_COLOR_READBACK
    VTK_METAL_ENABLE_OFFSCREEN_TARGET)
endif()

# Link Metal framework
if(APPLE_IOS)
  vtk_module_link(VTK::RenderingMetal PUBLIC "-framework Metal" "-framework QuartzCore" "-framework UIKit" "-framework Accelerate")
  if(VTK_REQUIRED_OBJCXX_FLAGS)
    set_source_files_properties(
      ${sources}
      PROPERTIES COMPILE_FLAGS "${VTK_REQUIRED_OBJCXX_FLAGS}")
  endif()
elseif(APPLE)
  vtk_module_link(VTK::RenderingMetal PUBLIC "-framework Metal" "-framework QuartzCore" "-framework Cocoa" "-framework Accelerate")
  if(VTK_REQUIRED_OBJCXX_FLAGS)
    set_source_files_properties(
      ${sources}
      PROPERTIES COMPILE_FLAGS "${VTK_REQUIRED_OBJCXX_FLAGS}")
  endif()
else()
  vtk_module_link(VTK::RenderingMetal PUBLIC "-framework Metal" "-framework QuartzCore")
endif()

vtk_add_test_mangling(VTK::RenderingMetal)
