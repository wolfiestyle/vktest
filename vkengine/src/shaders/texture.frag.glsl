#version 450
layout(binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    vec4 light_dir;
};
layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    float light_val = max(0.1, dot(fragNormal, light_dir.xyz));
    outColor = texture(texSampler, fragTexCoord) * light_val;
}
