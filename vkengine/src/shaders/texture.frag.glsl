#version 450
layout(binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
};

struct MaterialData {
    vec4 base_color;
};
layout(std140, binding = 1) readonly buffer MaterialBuffer {
    MaterialData materials[];
};

layout(binding = 2) uniform sampler2D texSampler;

layout(push_constant) uniform PushConstants {
    uint mat_id;
};

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 diffuse = ambient.rgb + max(0.0, dot(fragNormal, light_dir.xyz)) * light_color.rgb;
    vec3 color = texture(texSampler, fragTexCoord).rgb * fragColor * diffuse;
    vec3 base_color = materials[mat_id].base_color.rgb;
    outColor = vec4(color * base_color, 1.0);
}
