#version 450
#include "tonemap.inc.glsl"
layout(binding = 0) uniform samplerCube skybox;

layout(location = 0) in vec3 eyeDirection;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 radiance = texture(skybox, eyeDirection).rgb;
    vec3 color = tonemapReinhard(radiance);
    outColor = vec4(color, 1.0);
}
