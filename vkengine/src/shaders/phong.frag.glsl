#version 450
layout(binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    mat4 model;
    vec4 light_dir;
    vec4 light_color;
    vec4 view_pos;
};

struct MaterialData {
    vec4 base_color;
    vec4 base_pbr;
    vec4 emissive;
};
layout(std140, binding = 1) readonly buffer MaterialBuffer {
    MaterialData materials[];
};

layout(binding = 2) uniform sampler2D texColor;
layout(binding = 3) uniform sampler2D texMetalRough;
layout(binding = 4) uniform sampler2D texEmissive;

layout(push_constant) uniform PushConstants {
    uint mat_id;
};

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

float directional_light(vec3 light_dir, vec3 normal, vec3 view_dir, float smoothness) {
    vec3 light_d = normalize(-light_dir);
    float diffuse = max(0.0, dot(normal, light_d));
    float shininess = smoothness * smoothness * 128.0;
    float specular = 0.0;

    if (diffuse > 0.0) {
        //vec3 reflect_dir = reflect(-light_d, normal);
        //float spec = pow(max(0.0, dot(view_dir, reflect_dir)), shininess);
        vec3 half_dir = normalize(light_d + view_dir);
        float spec = pow(max(0.0, dot(normal, half_dir)), shininess);
        specular = spec * smoothness;
    }
    return diffuse + specular;
}

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 view_dir = normalize(view_pos.xyz - fragPos);
    vec3 ambient = vec3(light_dir.a, light_color.a, view_pos.a);
    vec4 pbr = materials[mat_id].base_pbr * texture(texMetalRough, fragTexCoord);
    vec3 light_val = ambient + directional_light(light_dir.xyz, normal, view_dir, 1.0 - pbr.g) * light_color.rgb;
    vec3 base_color = materials[mat_id].base_color.rgb * texture(texColor, fragTexCoord).rgb * fragColor;
    vec3 emissive = texture(texEmissive, fragTexCoord).rgb * materials[mat_id].emissive.rgb;
    vec3 color = base_color * light_val + emissive;
    outColor = vec4(color, 1.0);
}
