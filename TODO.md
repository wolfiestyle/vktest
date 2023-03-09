# TODO

- Optimize glTF loading. Write custom importer that outputs GPU-friendly data
- Build a global descriptor set for all textures and material settings used on
  the scene, indexed by push constant or another buffer. Evaluate performance
  vs push descriptors
- Rough visibility determination with bounding boxes, rebuild command buffer
  when visibility changes. Could use occlusion queries
- Depth pre-pass, do shading with depth test equal + depth writes off. It might
  not be necessary if objects are properly sorted in depth order and not
  overlapping.
- Group objects by pipeline (shader, material settings, culling mode..)
- Alpha tested (cutout) materials will likely need a separate pass. Shaders
  with discard can be slower than opaque ones. Could just alpha blend them, but
  now they can't write to depth (no shadows)
- Possibly join all same size textures on an array sampler, but would have to
  compare performance/complexity with push constant version. Will need a per-face
  texture index, could be an external array indexed by gl_VertexIndex/3 or
  actual per-face attributes if using mesh shaders
- Put all lightmaps on an single array image. Can be done for shadow maps too
- Implement shadow maps, store results per light an only recalculate when
  objects move. Possibly implement stencil shadows as an alternative
- Animation data stored on a streaming buffer (only weights), skinning done on
  vertex shader or a compute pass. The matrix tranform part of skeletal animation
  probably needs to be done on CPU, since it's a tree object. There could be
  GPU optimization for humanoid skeletons or other standard rigs
- Do async resource loading, synchronize between frames
- Configurable settings (multisampling, anisotropic filtering, shadow quality, 
  depth buffer mode, engine fine tuning, ...)
- OpenXR integration. It requires special creation procedures for the instance,
  device and uses it's own swapchain
