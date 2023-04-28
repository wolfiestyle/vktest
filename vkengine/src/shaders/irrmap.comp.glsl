#version 450
#include "sampling.inc.glsl"
#include "cubemap.inc.glsl"
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

const uint NumSamples = 1024 * 32;

layout(binding = 0) uniform samplerCube inputTex;
layout(binding = 1) uniform writeonly imageCube outputTex;

void main() {
    vec3 N = getCubemapDir(gl_GlobalInvocationID, imageSize(outputTex));
    mat3 TBN = computeTangentBasis(N);

    vec3 color = vec3(0.0);
    float total_weight = 0.0;
    for (uint i = 0; i < NumSamples; ++i) {
        vec2 Xi = sampleHammersley(i, NumSamples);
        vec3 L = TBN * sampleHemisphere(Xi);
        float NdotL = max(dot(N, L), 0.0);
        color += texture(inputTex, L).rgb * NdotL;
        total_weight += NdotL;
    }
    vec3 irradiance = color / total_weight;
    imageStore(outputTex, ivec3(gl_GlobalInvocationID), vec4(irradiance, 1.0));
}
