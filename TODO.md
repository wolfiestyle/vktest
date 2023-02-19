# TODO

- Implement complex model loading (gltf, FBX)
- Support for models with multiple textures/materials
- Allocate a single buffer for all geometry data of a mesh (vertices, indices).
  This might need to be done as a separate import/baking process.
- Build a global descriptor set for all textures and material settings used on
  the scene, indexed by push constant or another buffer
- Possibly use a buffer with draw data + CmdDrawIndexedIndirect
- Rough visibility determination with bounding boxes, rebuild command buffer
  when visibility changes. Could use occlusion queries
- Depth pre-pass, do shading with depth test equal + depth writes off. It might
  not be necessary if objects are properly sorted in depth order and not
  overlapping.
- Sort objects by shader, possibly limit to 1 shader per mesh, then each draw
  call set will basically have no state changes other than the push constant
- Alpha tested (cutout) materials will likely need a separate pass. Shaders
  with discard can be slower than opaque ones. Could just alpha blend them, but
  now they can't write to depth (no shadows)
- Possibly join all same size textures on an array sampler, but would have to
  compare performance/complexity with push constant version. Also I don't know
  if shared vertices will cause bleeding or other artifacts. Might need more
  data passed to the fragment shader like a per-face texture index
- Draw skybox where depth = 1.0 (far plane)
- Put all lightmaps on an single array sampler or big atlas texture
- Implement shadow maps, store results per light an only recalculate when
  objects move. Possibly implement stencil shadows as an alternative
- Animation data stored on a streaming buffer (only weights), skinning done on
  vertex shader or a compute pass. The matrix tranform part of skeletal animation
  probably needs to be done on CPU, since it's a tree object. There could be
  GPU optimization for humanoid skeletons or other standard rigs
- Do async resource loading, synchronize between frames
- Configurable settings (multisampling, anisotropic filtering, shadow quality, 
  depth buffer mode, engine fine tuning, ...)
- OpenXR integration (we need to accept a pre-made vulkan instance)
