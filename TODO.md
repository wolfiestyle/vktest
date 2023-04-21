# TODO

- Material system. Group objects by pipeline (shader, pipeline state), ideally
  sorted to avoid state changes
- Rough visibility determination with bounding boxes, can be done as a compute
  pre-pass, then use the results with draw indirect.
- Depth pre-pass, do shading with depth test equal + depth writes off
- Can do alpha test in depth pre-pass, depth test equal will fail for the
  discarded fragments, now opaque and cutout share the same pass
- Implement order independant transparency
- Setup for bindless (descriptor indexing). Needs a single global vertex/index
  buffer. Index materials by push constant or gl_DrawID
- Put all lightmaps on an single array image. Can be done for shadow maps too
- Could use raytracing for realtime lightmap generation, also the IBL maps
- Implement shadow maps, store results per light an only recalculate when
  objects move. Possibly implement stencil shadows as an alternative
- Animation could be done in series of compute pre-passes, transform vertices
  before vertex shader
- Light probes with spherical harmonics
- Do async resource loading, synchronize between frames
- Configurable settings (multisampling, anisotropic filtering, shadow quality, 
  engine fine tuning, ...)
- OpenXR integration. It requires special creation procedures for the instance,
  device and uses it's own swapchain
