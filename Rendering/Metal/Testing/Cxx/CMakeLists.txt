vtk_add_test_cxx(vtkRenderingMetalCxxTests tests
  TestMetalRenderWindow.cxx,NO_DATA,NO_SERDES
  TestMetalCamera.cxx,NO_DATA,NO_SERDES
  TestMetalLight.cxx,NO_DATA,NO_SERDES
  TestMetalActorProperty.cxx,NO_DATA,NO_SERDES
  TestMetalPointRender.cxx,NO_DATA,NO_SERDES
  TestMetalDepthPeeling.cxx,NO_DATA,NO_SERDES
  TestMetalCompositePolyDataMapper.cxx,NO_DATA,NO_SERDES
  TestMetalGlyph3DMapper.cxx,NO_DATA,NO_SERDES
  TestMetalHardwareSelector.cxx,NO_DATA,NO_SERDES
  TestMetalPolyDataMapper2D.cxx,NO_DATA,NO_SERDES
  TestMetalTexture.cxx,NO_DATA,NO_SERDES
  TestMetalVolumeRayCast.cxx,NO_DATA,NO_SERDES)

vtk_test_cxx_executable(vtkRenderingMetalCxxTests tests)

# Standalone dual-backend visual comparison harness. It renders every Metal
# test scene with both the Metal and the OpenGL backends and writes GL/Metal/
# diff PNGs plus a numeric error metric. It must live in its own executable:
# it auto-initializes the OpenGL backend (needed for the vtkShaderProgram
# object-factory override), which must NOT be active in the regression tests
# above, otherwise their vtkProperty::New()/vtkTexture::New() calls would also
# return OpenGL classes.
vtk_module_test_executable(vtkMetalGLVisualComparison TestMetalGLVisualComparison.cxx)
target_link_libraries(vtkMetalGLVisualComparison
  PRIVATE
    vtkRenderingOpenGL2
    vtkRenderingVolumeOpenGL2)

add_test(
  NAME VTK::RenderingMetalCxx-TestMetalGLVisualComparison
  COMMAND "$<TARGET_FILE:vtkMetalGLVisualComparison>"
          "--out" "${_vtk_build_TEST_OUTPUT_DIRECTORY}/visual_compare")
set_tests_properties(VTK::RenderingMetalCxx-TestMetalGLVisualComparison
  PROPERTIES
    LABELS "${_vtk_build_test_labels}")
