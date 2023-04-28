#version 450
layout(push_constant) uniform PushConstants {
    mat4 viewproj_inv;
    vec4 lod;
};

layout(location = 0) out vec3 eyeDirection;

void main() {
    int x = ((gl_VertexIndex & 1) << 1) - 1;
    int y = (gl_VertexIndex & 2) - 1;
    vec4 pos = vec4(x, y, 1.0, 1.0);
    eyeDirection = (viewproj_inv * pos).xyz;
    gl_Position = pos;
}
