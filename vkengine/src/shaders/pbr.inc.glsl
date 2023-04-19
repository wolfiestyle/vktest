#include "constants.inc.glsl"

const vec3 Fdielectric = vec3(0.04);

float distributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometrySchlickGGX(float NdotV, float k) {
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(float NdotL, float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = r * r / 8.0;
    float ggx1 = geometrySchlickGGX(NdotL, k);
    float ggx2 = geometrySchlickGGX(NdotV, k);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
