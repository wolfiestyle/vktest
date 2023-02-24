#version 450
layout(binding = 0) uniform UniformBufferObject {
    mat4 mvp;
    mat4 viewproj_inv;
} ubo;

layout(location = 0) out vec3 eyeDirection;

void main() {
    int x = ((gl_VertexIndex & 1) << 1) - 1;
    int y = (((gl_VertexIndex + 1) / 3 & 1) << 1) - 1;
    vec4 pos = vec4(vec2(x, y), 0.99999, 1.0);
    eyeDirection = (ubo.viewproj_inv * pos).xzy;
    gl_Position = pos;
}
