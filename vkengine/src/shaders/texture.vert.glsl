#version 450
layout(binding = 0) uniform ObjectUniforms {
    mat4 mvp;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inTexCoord;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = mvp * vec4(inPosition, 1.0);
    fragNormal = inNormal;
    fragTexCoord = vec2(inTexCoord.x, 1.0 - inTexCoord.y);
}
