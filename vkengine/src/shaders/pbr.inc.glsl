#include "constants.inc.glsl"

const vec3 Fdielectric = vec3(0.04);

float distributionGGX(float NdotH, float roughness) {
    float a2 = pow(roughness, 4.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometrySchlickGGX(float cosTheta, float k) {
    return cosTheta / (cosTheta * (1.0 - k) + k);
}

float geometrySmith(float NdotL, float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = r * r / 8.0;
    float ggx1 = geometrySchlickGGX(NdotL, k);
    float ggx2 = geometrySchlickGGX(NdotV, k);
    return ggx1 * ggx2;
}

float smithGGXCorrelated(float NdotL, float NdotV, float roughness) {
    float a2 = pow(roughness, 4.0);
    float ggxL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    float ggxV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    return 0.5 / (ggxL + ggxV);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    vec3 Fr = max(vec3(1.0 - roughness), F0) - F0;
    return F0 + Fr * pow(1.0 - cosTheta, 5.0);
}
