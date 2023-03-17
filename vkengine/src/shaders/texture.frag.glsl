#version 450
layout(binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
};
layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 diffuse = ambient.rgb + max(0.0, dot(fragNormal, light_dir.xyz)) * light_color.rgb;
    vec3 color = texture(texSampler, fragTexCoord).rgb * diffuse;
    vec3 base_color = vec3(light_dir.a, light_color.a, ambient.a);
    outColor = vec4(color * base_color, 1.0);
}
