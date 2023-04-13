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
layout(binding = 5) uniform sampler2D texNormal;

layout(push_constant) uniform PushConstants {
    uint mat_id;
};

layout(location = 0) in FragIn {
    vec3 Pos;
    vec2 TexCoord;
    vec3 Color;
    mat3 TBN;
} frag;

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
    vec3 view_dir = normalize(view_pos.xyz - frag.Pos);
    vec3 ambient = vec3(light_dir.a, light_color.a, view_pos.a);
    vec4 pbr = materials[mat_id].base_pbr * texture(texMetalRough, frag.TexCoord);
    vec3 normal_map = texture(texNormal, frag.TexCoord).rgb * 2.0 - 1.0;
    vec3 normal = normalize(frag.TBN * normal_map) * materials[mat_id].emissive.a;
    vec3 light_val = ambient + directional_light(light_dir.xyz, normal, view_dir, 1.0 - pbr.g) * light_color.rgb;
    vec3 base_color = materials[mat_id].base_color.rgb * texture(texColor, frag.TexCoord).rgb * frag.Color;
    vec3 emissive = texture(texEmissive, frag.TexCoord).rgb * materials[mat_id].emissive.rgb;
    vec3 color = base_color * light_val + emissive;
    outColor = vec4(color, 1.0);
}
