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

vec3 sampleHemisphere(vec2 u) {
    float phi = 2.0 * PI * u.y;
    float u1p = sqrt(max(1.0 - u.x * u.x, 0.0));
    return vec3(cos(phi) * u1p, sin(phi) * u1p, u.x);
}

vec3 importanceSampleGGX(vec2 Xi, float roughSq) {
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / ((roughSq * roughSq - 1.0) * Xi.y + 1.0));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

mat3 computeTangentBasis(vec3 N) {
    vec3 B = vec3(0.0, 1.0, 0.0);
    float NdotUp = dot(N, vec3(0.0, 1.0, 0.0));
    if (1.0 - abs(NdotUp) <= Epsilon) {
        B = NdotUp > 0.0 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 0.0, -1.0);
    }
    vec3 T = normalize(cross(B, N));
    B = cross(N, T);
    return mat3(T, B, N);
}
