#version 450
layout(binding = 0) uniform UniformBufferObject {
    mat4 mvp;
} ubo;

layout(location = 0) out vec3 eyeDirection;

vec2 positions[6] = vec2[](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(1.0, 1.0),   vec2(-1.0, 1.0), vec2(1.0, -1.0)
);

void main() {
    vec4 pos = vec4(positions[gl_VertexIndex], 0.99999, 1.0);
    eyeDirection = (inverse(ubo.mvp) * pos).xzy;
    gl_Position = pos;
}
