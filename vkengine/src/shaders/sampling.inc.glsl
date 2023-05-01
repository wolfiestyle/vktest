#include "constants.inc.glsl"

float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 sampleHammersley(uint i, uint NumSamples) {
    return vec2(float(i) / float(NumSamples), radicalInverse_VdC(i));
}

vec3 sampleHemisphere(vec2 Xi) {
    float phi = TwoPI * Xi.y;
    float sinTheta = sqrt(max(1.0 - Xi.x * Xi.x, 0.0));
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, Xi.x);
}

vec3 importanceSampleGGX(vec2 Xi, float roughSq) {
    float phi = TwoPI * Xi.x;
    float cosThetaSq = max((1.0 - Xi.y) / ((roughSq * roughSq - 1.0) * Xi.y + 1.0), 0.0);
    float cosTheta = sqrt(cosThetaSq);
    float sinTheta = sqrt(1.0 - cosThetaSq);
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

mat3 computeTangentBasis(vec3 N) {
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);
    return mat3(T, B, N);
}
