vtk_add_test_cxx(vtkRenderingMetalCxxTests tests
  TestMetalRenderWindow.cxx,NO_DATA,NO_VALID,NO_SERDES
  TestMetalCamera.cxx,NO_DATA,NO_VALID,NO_SERDES
  TestMetalLight.cxx,NO_DATA,NO_VALID,NO_SERDES
  TestMetalActorProperty.cxx,NO_DATA,NO_VALID,NO_SERDES
  TestMetalPointRender.cxx,NO_DATA,NO_VALID,NO_SERDES
  TestMetalDepthPeeling.cxx,NO_DATA,NO_VALID,NO_SERDES
  TestMetalCompositePolyDataMapper.cxx,NO_DATA,NO_VALID,NO_SERDES
  TestMetalGlyph3DMapper.cxx,NO_DATA,NO_VALID,NO_SERDES
  TestMetalHardwareSelector.cxx,NO_DATA,NO_VALID,NO_SERDES
  TestMetalPolyDataMapper2D.cxx,NO_DATA,NO_VALID,NO_SERDES
  TestMetalTexture.cxx,NO_DATA,NO_VALID,NO_SERDES
  TestMetalVolumeRayCast.cxx,NO_DATA,NO_VALID,NO_SERDES)

vtk_test_cxx_executable(vtkRenderingMetalCxxTests tests)
