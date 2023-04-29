#version 450
#include "pbr.inc.glsl"
#include "tonemap.inc.glsl"

struct LightData {
    vec4 pos;   // .w: 0 = directional, 1 = point/spot
    vec4 color; // .w = spot_offset
    vec4 spot;  // .w = spot_scale
};

layout(set = 0, binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    mat4 model;
    vec3 view_pos;
    uint num_lights;
};

layout(set = 0, binding = 1, std140) readonly buffer LightBuffer {
    LightData lights[];
};

layout(set = 0, binding = 2) uniform samplerCube irradianceMap;
layout(set = 0, binding = 3) uniform samplerCube prefilterMap;
layout(set = 0, binding = 4) uniform sampler2D BRDF_lut;

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
    uint uv_sets;
} material;

layout(location = 0) in FragIn {
    vec3 Pos;
    vec4 Color;
    mat3 TBN;
    vec2 uvColor;
    vec2 uvMetalRough;
    vec2 uvNormal;
    vec2 uvEmissive;
    vec2 uvOcclusion;
} frag;

layout(location = 0) out vec4 outColor;

vec3 pbr_light(vec3 N, vec3 albedo, float metallic, float roughness, float ao) {
    vec3 V = normalize(view_pos - frag.Pos);
    vec3 R = reflect(-V, N);
    float NdotV = max(dot(N, V), 0.0);
    vec3 F0 = mix(Fdielectric, albedo, metallic);
    float roughSq = roughness * roughness;

    // direct lighting
    vec3 direct = vec3(0.0);
    for (int i = 0; i < num_lights; ++i) {
        float type = lights[i].pos.w;
        vec3 dir = (2.0 * type - 1.0) * lights[i].pos.xyz - frag.Pos * type;
        vec3 L = normalize(dir);
        vec3 H = normalize(V + L);
        float spot = clamp(dot(lights[i].spot.xyz, -L) * lights[i].spot.w + lights[i].color.w, 0.0, 1.0);
        float attenuation = 1.0 / mix(1.0, dot(dir, dir), type);
        vec3 radiance = lights[i].color.rgb * spot * attenuation;

        float NdotH = max(dot(N, H), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        float NDF = distributionGGX(NdotH, roughSq);
        float G = geometrySmith(NdotL, NdotV, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        vec3 specular = (NDF * G * F) / max(4.0 * NdotV * NdotL, Epsilon);

        vec3 kD = mix(vec3(1.0) - F, vec3(0.0), metallic);
        vec3 diffuse = kD * albedo / PI;
        direct += (diffuse + specular) * radiance * NdotL;
    }

    // IBL
    vec3 kS = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD = mix(1.0 - kS, vec3(0.0), metallic);
    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 iblDiffuse = kD * irradiance * albedo;

    int maxLod = textureQueryLevels(prefilterMap) - 1;
    vec3 radiance = textureLod(prefilterMap, R, roughness * maxLod).rgb;
    vec2 envBRDF = texture(BRDF_lut, vec2(NdotV, roughness)).rg;
    vec3 iblSpec = radiance * (kS * envBRDF.x + envBRDF.y);
    vec3 ambient = iblDiffuse * ao + iblSpec;

    return ambient + direct;
}

void main() {
    vec4 albedo = texture(texColor, frag.uvColor) * material.base_color * frag.Color;
    vec2 metalrough = texture(texMetalRough, frag.uvMetalRough).bg * material.base_pbr.bg;
    vec3 normal_map = texture(texNormal, frag.uvNormal).rgb * 2.0 - 1.0;
    vec3 emissive = texture(texEmissive, frag.uvEmissive).rgb * material.emissive;
    float occlusion = (texture(texOcclusion, frag.uvOcclusion).r - 1.0) * material.base_pbr.r + 1.0;
    vec3 normal = normalize(frag.TBN * normal_map * vec3(material.normal_scale.xx, 1.0));
    vec3 radiance = pbr_light(normal, albedo.rgb, metalrough.r, metalrough.g, occlusion) + emissive;
    vec3 color = tonemapReinhard(radiance);
    outColor = vec4(color, albedo.a);
}
