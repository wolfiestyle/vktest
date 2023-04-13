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
layout(binding = 4) uniform sampler2D texNormal;
layout(binding = 5) uniform sampler2D texEmissive;
layout(binding = 6) uniform sampler2D texOcclusion;

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

const float PI = 3.14159265359;
const float Epsilon = 0.00001;
const vec3 Fdielectric = vec3(0.04);

float distribution_ggx(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometry_schlick_ggx(float NdotV, float k) {
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometry_smith(float NdotL, float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = r * r / 8.0;
    float ggx1 = geometry_schlick_ggx(NdotL, k);
    float ggx2 = geometry_schlick_ggx(NdotV, k);
    return ggx1 * ggx2;
}

vec3 fresnel_schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 direct_light(vec3 light_dir, vec3 light_color, vec3 N, vec3 albedo, vec2 metalrough) {
    vec3 V = normalize(view_pos.xyz - frag.Pos);
    vec3 L = normalize(-light_dir); // light_pos - frag.Pos
    vec3 H = normalize(V + L);
    //float distance = length(light_pos - frag.Pos);
    //float attenuation = 1.0 / (distance * distance);
    vec3 radiance = light_color; // * attenuation

    vec3 F0 = mix(Fdielectric, albedo, metalrough.r);
    float NdotH = max(dot(N, H), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float NDF = distribution_ggx(NdotH, metalrough.g);
    float G = geometry_smith(NdotL, NdotV, metalrough.g);
    vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (NDF * G * F) / max(4.0 * NdotV * NdotL, Epsilon);

    vec3 kD = mix(vec3(1.0) - F, vec3(0.0), metalrough.r);
    vec3 diffuse = kD * albedo / PI;

    return (diffuse + specular) * radiance * NdotL;
}

void main() {
    vec3 albedo = texture(texColor, frag.TexCoord).rgb * materials[mat_id].base_color.rgb * frag.Color;
    vec2 metalrough = texture(texMetalRough, frag.TexCoord).bg * materials[mat_id].base_pbr.bg;
    vec3 normal_map = texture(texNormal, frag.TexCoord).rgb * 2.0 - 1.0;
    vec3 normal = normalize(frag.TBN * normal_map) * materials[mat_id].emissive.a;
    vec3 ambient = vec3(0.03) * texture(texOcclusion, frag.TexCoord).r * albedo;
    vec3 direct = direct_light(light_dir.xyz, light_color.rgb, normal, albedo, metalrough);
    vec3 emissive = texture(texEmissive, frag.TexCoord).rgb * materials[mat_id].emissive.rgb;
    vec3 color = ambient + direct + emissive;
    outColor = vec4(color, 1.0);
}
