#version 450
#include "sampling.inc.glsl"
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) uniform samplerCube inputTex;
layout(binding = 1) uniform writeonly imageCube outputTex;

void main() {
    vec3 N = getCubemapDir(gl_GlobalInvocationID, imageSize(outputTex));
    mat3 TBN = computeTangentBasis(N);

    vec3 color = vec3(0.0);
    float total_weight = 0.0;
    for (uint i = 0; i < NumSamples; ++i) {
        vec2 Xi = sampleHammersley(i);
        vec3 world_dir = TBN * sampleHemisphere(Xi);
        float cosTheta = max(dot(world_dir, N), 0.0);
        color += texture(inputTex, world_dir).rgb * cosTheta;
        total_weight += cosTheta;
    }
    vec3 irradiance = color / total_weight;
    imageStore(outputTex, ivec3(gl_GlobalInvocationID), vec4(irradiance, 1.0));
}
