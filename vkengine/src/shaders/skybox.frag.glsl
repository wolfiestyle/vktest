#version 450
layout(binding = 1) uniform samplerCube skybox;

layout(location = 0) in vec3 eyeDirection;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(skybox, eyeDirection);
}
