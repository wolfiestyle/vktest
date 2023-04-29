#version 450
#include "tonemap.inc.glsl"
layout(push_constant) uniform PushConstants {
    mat4 viewproj_inv;
    vec4 params; // .x = lod, .z = far_depth
};

layout(binding = 0) uniform samplerCube skybox;

layout(location = 0) in vec3 eyeDirection;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 radiance = textureLod(skybox, eyeDirection, params.x).rgb;
    vec3 color = tonemapReinhard(radiance);
    outColor = vec4(color, 1.0);
}
