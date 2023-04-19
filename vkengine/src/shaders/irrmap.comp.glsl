#version 450
#include "sampling.inc.glsl"
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0) uniform samplerCube inputTex;
layout(binding = 1, rgba16f) uniform writeonly imageCube outputTex;

const uint NumSamples = 4096;

void main() {
    vec3 N = getCubemapDir(gl_GlobalInvocationID, imageSize(outputTex));
    vec3 S, T;
    computeTangentBasis(N, S, T);

    vec3 color = vec3(0.0);
    float total_weight = 0.0;
    for (uint i = 0; i < NumSamples; ++i) {
        vec2 u = sampleHammersley(i, NumSamples);
        vec3 tangent_dir = sampleHemisphere(u);
        vec3 world_dir = S * tangent_dir.x + T * tangent_dir.y + N * tangent_dir.z;
        float cosTheta = max(dot(world_dir, N), 0.0);
        color += texture(inputTex, world_dir).rgb * cosTheta;
        total_weight += cosTheta;
    }
    vec3 irradiance = color / total_weight;
    imageStore(outputTex, ivec3(gl_GlobalInvocationID), vec4(irradiance, 1.0));
}
