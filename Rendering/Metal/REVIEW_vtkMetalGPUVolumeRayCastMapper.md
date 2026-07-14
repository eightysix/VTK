# Review: vtkMetalGPUVolumeRayCastMapper.mm

## Critical Bugs

### 1. `UpdateBlockTextures` destroys blocks created by caller (line 1838)
- `ClearBlocks()` inside `UpdateBlockTextures` destroyed blocks created by `UpdateVolumeTexture`
- Partitions feature was effectively broken
- **Status**: FIXED - Removed erroneous ClearBlocks() call

### 2. `SortedBlockOrder` fixed-size array overflow (header line 159)
- `SortedBlockOrder[64]` overflows if partitions > 64 blocks
- **Status**: FIXED - Changed to std::vector<int>

### 3. Label map transfer texture reuse without size check (lines 1387-1393)
- Texture reused without checking if numLabels changed
- **Status**: FIXED - Added dimension check before reuse

## Correctness Issues

### 4. Integer overflow in block data extraction (line 1896)
- `fullDims[1] * fullDims[0]` was `int * int`, overflows for large volumes
- **Status**: FIXED - Cast fullDims to vtkIdType before multiplication

### 5. Block rendering uses full-volume vertex buffer
- Per-block path reuses full-volume geometry
- **Status**: DEFERRED (shader-side change needed)

### 6. Confusing matrix transposition pattern (lines 2312-2322)
- In-place transpose/read/retranspose/invert pattern
- **Status**: FIXED - Replaced with clear inverse-transpose pattern using vtkMatrix4x4::Invert

### 7. Per-block staging buffer allocation (line 1909)
- N separate command buffers for N blocks
- **Status**: DEFERRED (performance)

## Performance Issues

### 8. Shader recompilation on pipeline invalidation (lines 314-315, 2461-2462)
- **Status**: DEFERRED

### 9. Per-block uniform buffer update via memcpy
- **Status**: DEFERRED

### 10. Per-block staging buffer allocation
- **Status**: DEFERRED

### 11. VolumeSampler uses mip filter without mipmaps (line 2499)
- **Status**: FIXED - Removed mipFilter setting

### 12. Duplicated rendering code paths
- ~175 lines duplicated between image-sampling and standard paths
- **Status**: FIXED - Extracted `BindEncoderResources` and `DrawBlocks` helper methods

### 13. Full 4x4 matrix inverse on CPU every frame
- 3 redundant model-matrix inverses per frame
- **Status**: FIXED - Hoisted `invModelMatrix` to GPURender scope, passed to `SetClippingPlaneUniforms` (3→1 per frame)

### 14. `UpdateMaskTexture` uses `GetComponent` in serial loop
- **Status**: FIXED - Added optimized paths for common types with vtkSMPTools::For

### 15. Mask texture uses `MTLStorageModeShared`
- **Status**: DEFERRED

## Minor Issues

### 16. `VolumeTextureView` and `ColorOpacityTextureView` unused
- **Status**: FIXED - Removed unused aliases

### 17. Empty `PreRender`, `RenderBlock`, `PostRender` overrides
- **Status**: DEFERRED (required by pure virtual interface — cannot remove)

### 18. Inconsistent CFRelease pattern
- **Status**: DEFERRED (pattern is actually consistent: check null → CFRelease → set nullptr)

## Summary of Applied Fixes

1. **Critical Bug #1**: Removed `ClearBlocks()` from `UpdateBlockTextures` that was destroying blocks created by the caller
2. **Critical Bug #2**: Changed `SortedBlockOrder` from fixed `int[64]` to `std::vector<int>` to prevent buffer overflow
3. **Critical Bug #3**: Added dimension validation for `LabelMapTransferTexture` before reuse
4. **Correctness #4**: Cast `fullDims` to `vtkIdType` before multiplication to prevent integer overflow
5. **Correctness #6**: Replaced confusing in-place transpose/read/retranspose/invert with clear inverse-transpose pattern
6. **Performance #11**: Removed unused `mipFilter` from `VolumeSampler`
7. **Performance #12**: Extracted `BindEncoderResources` and `DrawBlocks` helpers to eliminate ~175 lines of duplicated rendering code
8. **Performance #13**: Eliminated 2 redundant 4x4 matrix inverses per frame (3→1) by hoisting `invModelMatrix`
9. **Performance #14**: Optimized `UpdateMaskTexture` with direct pointer access and parallelization
10. **Minor #16**: Removed unused `VolumeTextureView` and `ColorOpacityTextureView` aliases
