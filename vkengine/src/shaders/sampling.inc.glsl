#include "constants.inc.glsl"

layout(constant_id = 0) const uint NumSamples = 4096;

float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 sampleHammersley(uint i) {
    return vec2(float(i) / float(NumSamples), radicalInverse_VdC(i));
}

vec3 sampleHemisphere(vec2 u) {
    float phi = 2.0 * PI * u.y;
    float u1p = sqrt(max(1.0 - u.x * u.x, 0.0));
    return vec3(cos(phi) * u1p, sin(phi) * u1p, u.x);
}

vec3 importanceSampleGGX(vec2 Xi, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

vec3 getCubemapDir(uvec3 img_coord, ivec2 img_size) {
    vec2 st = img_coord.xy / vec2(img_size);
    vec2 uv = 2.0 * vec2(st.x, 1.0 - st.y) - vec2(1.0);
    vec3 ret;
    switch (img_coord.z) {
        case 0: ret = vec3(1.0,  uv.y, -uv.x); break;
        case 1: ret = vec3(-1.0, uv.y,  uv.x); break;
        case 2: ret = vec3(uv.x, 1.0, -uv.y); break;
        case 3: ret = vec3(uv.x, -1.0, uv.y); break;
        case 4: ret = vec3(uv.x, uv.y, 1.0); break;
        case 5: ret = vec3(-uv.x, uv.y, -1.0); break;
    }
    return normalize(ret);
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
