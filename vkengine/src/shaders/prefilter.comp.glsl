#version 450
#include "sampling.inc.glsl"
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) uniform samplerCube inputTex;
layout(binding = 1, rgba16f) uniform writeonly imageCube outputTex;

layout(push_constant) uniform PushConstants {
    float roughness;
};

void main() {
    vec3 N = getCubemapDir(gl_GlobalInvocationID, imageSize(outputTex));
    mat3 TBN = computeTangentBasis(N);

    vec3 color = vec3(0.0);
    float total_weight = 0.0;
    for (uint i = 0; i < NumSamples; ++i) {
        vec2 Xi = sampleHammersley(i);
        vec3 H = TBN * importanceSampleGGX(Xi, roughness);
        vec3 L = normalize(reflect(-N, H));
        float NdotL = max(dot(N, L), 0.0);
        color += texture(inputTex, L).rgb * NdotL;
        total_weight += NdotL;
    }
    vec3 prefiltered = color / (total_weight != 0.0 ? total_weight : NumSamples);
    imageStore(outputTex, ivec3(gl_GlobalInvocationID), vec4(prefiltered, 1.0));
}
