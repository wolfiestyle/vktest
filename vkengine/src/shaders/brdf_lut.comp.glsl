#version 450
#include "sampling.inc.glsl"
#include "pbr.inc.glsl"
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

const uint NumSamples = 1024 * 4;

layout(binding = 0) uniform writeonly image2D outputTex;

vec2 integrateBRDF(float NdotV, float roughness) {
    vec3 V = vec3(sqrt(1.0 - NdotV * NdotV), 0.0, NdotV);
    //vec3 N = vec3(0.0, 0.0, 1.0);
    vec2 res = vec2(0.0);
    for (uint i = 0; i < NumSamples; ++i) {
        vec2 Xi = sampleHammersley(i, NumSamples);
        vec3 H = importanceSampleGGX(Xi, roughness);
        vec3 L = normalize(reflect(-V, H));
        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);
        if (NdotL > 0.0) {
            float G = smithGGXCorrelated(NdotL, NdotV, roughness);
            float G_vis = G * VdotH * NdotL / NdotH;
            float Fc = pow(1.0 - VdotH, 5.0);
            res.x += (1.0 - Fc) * G_vis;
            res.y += Fc * G_vis;
        }
    }
    return 4.0 * res / float(NumSamples);
}

void main() {
    vec2 coord = (gl_GlobalInvocationID.xy + 1) / vec2(imageSize(outputTex));
    vec2 value = integrateBRDF(coord.x, coord.y);
    imageStore(outputTex, ivec2(gl_GlobalInvocationID.xy), vec4(value, 0.0, 0.0));
}
