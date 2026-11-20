#version 450
#include "cubemap.inc.glsl"
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D inputTex;
layout(binding = 1) uniform writeonly imageCube outputTex;

void main() {
    ivec2 size = imageSize(outputTex);
    if (any(greaterThanEqual(gl_GlobalInvocationID.xy, uvec2(size)))) {
        return;
    }

    vec3 dir = getCubemapDir(gl_GlobalInvocationID, size);
    vec2 uv = cubemapDirToEquirect(dir);
    vec4 color = texture(inputTex, uv);
    imageStore(outputTex, ivec3(gl_GlobalInvocationID), color);
}
