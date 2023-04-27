#version 450
#include "pbr.inc.glsl"
#include "tonemap.inc.glsl"
layout(constant_id = 0) const uint NumLights = 1;
const int UVbits = 1;
const uint NumUVs = 1 << UVbits;

struct LightData {
    vec4 pos;
    vec4 color;
};

layout(set = 0, binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    mat4 model;
    vec4 view_pos;
    LightData lights[NumLights];
};

layout(set = 0, binding = 1) uniform samplerCube irradianceMap;
layout(set = 0, binding = 2) uniform samplerCube prefilterMap;
layout(set = 0, binding = 3) uniform sampler2D BRDF_lut;

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
    vec2 TexCoord[NumUVs];
    vec4 Color;
    mat3 TBN;
} frag;

layout(location = 0) out vec4 outColor;

vec3 pbr_light(vec3 N, vec3 albedo, float metallic, float roughness, float ao) {
    vec3 V = normalize(view_pos.xyz - frag.Pos);
    vec3 R = reflect(-V, N);
    float NdotV = max(dot(N, V), 0.0);
    vec3 F0 = mix(Fdielectric, albedo, metallic);
    float roughSq = roughness * roughness;

    // direct lighting
    vec3 direct = vec3(0.0);
    for (int i = 0; i < lights.length(); ++i) {
        vec3 dir = lights[i].pos.xyz - frag.Pos * lights[i].pos.w;
        vec3 L = normalize(dir);
        vec3 H = normalize(V + L);
        float attenuation = mix(1.0, dot(dir, dir), lights[i].pos.w);
        vec3 radiance = lights[i].color.rgb / attenuation;

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
    uint color_uv = bitfieldExtract(material.uv_sets, 0, UVbits);
    uint metrgh_uv = bitfieldExtract(material.uv_sets, UVbits, UVbits);
    uint normal_uv = bitfieldExtract(material.uv_sets, UVbits * 2, UVbits);
    uint emiss_uv = bitfieldExtract(material.uv_sets, UVbits * 3, UVbits);
    uint occl_uv = bitfieldExtract(material.uv_sets, UVbits * 4, UVbits);
    vec4 albedo = texture(texColor, frag.TexCoord[color_uv]) * material.base_color * frag.Color;
    vec2 metalrough = texture(texMetalRough, frag.TexCoord[metrgh_uv]).bg * material.base_pbr.bg;
    vec3 normal_map = texture(texNormal, frag.TexCoord[normal_uv]).rgb * 2.0 - 1.0;
    vec3 emissive = texture(texEmissive, frag.TexCoord[emiss_uv]).rgb * material.emissive;
    float occlusion = (texture(texOcclusion, frag.TexCoord[occl_uv]).r - 1.0) * material.base_pbr.r + 1.0;
    vec3 normal = normalize(frag.TBN * normal_map * vec3(vec2(material.normal_scale), 1.0));
    vec3 radiance = pbr_light(normal, albedo.rgb, metalrough.r, metalrough.g, occlusion) + emissive;
    vec3 color = tonemapReinhard(radiance);
    outColor = vec4(color, albedo.a);
}
