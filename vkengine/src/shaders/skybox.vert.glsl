#version 450
layout(push_constant) uniform PushConstants {
    mat4 viewproj_inv;
    vec4 params; // .x = lod, .z = far_depth
};

layout(location = 0) out vec3 eyeDirection;

void main() {
    int x = (gl_VertexIndex & 1) * 4 - 1;
    int y = (gl_VertexIndex >> 1 & 1) * 4 - 1;
    vec4 pos = vec4(x, y, params.z, 1.0);
    eyeDirection = (viewproj_inv * pos).xyz;
    gl_Position = pos;
}
