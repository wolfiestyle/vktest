const float Lwhite = 3.0;
const float LwhiteSqInv = 1.0 / (Lwhite * Lwhite);

float colorToLuminance(vec3 color) {
    return dot(color, vec3(0.2125, 0.7154, 0.0721));
}

vec3 tonemapReinhard(vec3 L) {
    return L * (L * LwhiteSqInv + 1.0) / (1.0 + L);
}
