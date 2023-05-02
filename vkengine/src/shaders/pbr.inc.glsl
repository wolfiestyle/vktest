#include "constants.inc.glsl"

const vec3 Fdielectric = vec3(0.04);

float distributionGGX(float NdotH, float roughSq) {
    float a2 = roughSq * roughSq;
    float denom = (NdotH * a2 - NdotH) * NdotH + 1.0;
    return a2 * InvPI / (denom * denom);
}

float geometrySchlickGGX(float cosTheta, float k) {
    return cosTheta / (cosTheta * (1.0 - k) + k);
}

float geometrySmith(float NdotL, float NdotV, float roughSq) {
    float k = roughSq / 2.0;
    float ggxL = geometrySchlickGGX(NdotL, k);
    float ggxV = geometrySchlickGGX(NdotV, k);
    return ggxL * ggxV;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    vec3 Fr = max(vec3(1.0 - roughness), F0) - F0;
    return F0 + Fr * pow(1.0 - cosTheta, 5.0);
}

vec3 multiScatterIBL(vec2 Fab, vec3 FssEss, vec3 F0) {
    float Ess = Fab.x + Fab.y;
    float Ems = 1.0 - Ess;
    vec3 Favg = F0 + (1.0 - F0) / 21.0;
    return FssEss * Favg * Ems / (1.0 - Ems * Favg);
}
