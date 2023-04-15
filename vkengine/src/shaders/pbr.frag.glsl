#version 450
layout(set = 0, binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    mat4 model;
    vec4 light;
    vec4 light_color;
    vec4 view_pos;
};

layout(set = 1, binding = 0) uniform sampler2D texColor;
layout(set = 1, binding = 1) uniform sampler2D texMetalRough;
layout(set = 1, binding = 2) uniform sampler2D texNormal;
layout(set = 1, binding = 3) uniform sampler2D texEmissive;
layout(set = 1, binding = 4) uniform sampler2D texOcclusion;

layout(push_constant) uniform PushConstants {
    vec4 base_color;
    vec3 base_pbr;
    float normal_scale;
    vec3 emissive;
    float unused0;
} material;

layout(location = 0) in FragIn {
    vec3 Pos;
    vec2 TexCoord;
    vec4 Color;
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

vec3 direct_light(vec4 light, vec3 light_color, vec3 N, vec3 albedo, vec2 metalrough) {
    vec3 V = normalize(view_pos.xyz - frag.Pos);
    vec3 dir = light.xyz - frag.Pos * light.w;
    vec3 L = normalize(dir);
    vec3 H = normalize(V + L);
    float attenuation = mix(1.0, dot(dir, dir), light.w);
    vec3 radiance = light_color / attenuation;

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
    vec4 albedo = texture(texColor, frag.TexCoord) * material.base_color * frag.Color;
    vec2 metalrough = texture(texMetalRough, frag.TexCoord).bg * material.base_pbr.bg;
    vec3 normal_map = texture(texNormal, frag.TexCoord).rgb * 2.0 - 1.0;
    vec3 emissive = texture(texEmissive, frag.TexCoord).rgb * material.emissive;
    float occlusion = (texture(texOcclusion, frag.TexCoord).r - 1.0) * material.base_pbr.r + 1.0;
    vec3 normal = normalize(frag.TBN * normal_map * vec3(vec2(material.normal_scale), 1.0));
    vec3 ambient = light_color.a * occlusion * albedo.rgb;
    vec3 direct = direct_light(light, light_color.rgb, normal, albedo.rgb, metalrough);
    vec3 color = ambient + direct + emissive;
    outColor = vec4(color, 1.0);
}
