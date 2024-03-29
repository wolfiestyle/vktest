#version 450
#include "sampling.inc.glsl"
#include "cubemap.inc.glsl"
#include "pbr.inc.glsl"
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(constant_id = 1) const uint MipLevels = 1;
const uint NumSamples = 1024 * 4;

layout(binding = 0) uniform samplerCube inputTex;
layout(binding = 1) uniform writeonly imageCube outputTex;

layout(push_constant) uniform PushConstants {
    uint level;
};

vec3 computePrefiltered(vec3 N, ivec2 texSize) {
    mat3 TBN = computeTangentBasis(N);
    //vec3 V = N;
    float roughness = float(level) / float(MipLevels - 1);
    float roughSq = roughness * roughness;
    float saTexel = TwoPI / (texSize.x * texSize.y * 3);

    vec3 color = vec3(0.0);
    float total_weight = 0.0;
    for (uint i = 0; i < NumSamples; ++i) {
        vec2 Xi = sampleHammersley(i, NumSamples);
        vec3 H = TBN * importanceSampleGGX(Xi, roughSq);
        vec3 L = normalize(reflect(-N, H));
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        if (NdotL > 0.0) {
            float pdf = distributionGGX(NdotH, roughSq) * 0.25;
            float saSample = 1.0 / (NumSamples * pdf + Epsilon);
            float mipLevel = roughness != 0.0 ? 0.5 * log2(saSample / saTexel) : 0.0;
            color += textureLod(inputTex, L, mipLevel).rgb * NdotL;
            total_weight += NdotL;
        }
    }
    return color / (total_weight != 0.0 ? total_weight : NumSamples);
}

void main() {
    ivec2 img_size = imageSize(outputTex);
    bvec2 cmp = lessThan(gl_GlobalInvocationID.xy, img_size);
    if (cmp.x && cmp.y) {
        vec3 N = getCubemapDir(gl_GlobalInvocationID, img_size);
        vec3 prefiltered = computePrefiltered(N, textureSize(inputTex, 0));
        imageStore(outputTex, ivec3(gl_GlobalInvocationID), vec4(prefiltered, 1.0));
    }
}
