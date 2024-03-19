#version 450
layout(push_constant) uniform PushConstants {
    mat4 proj;
    vec4 extra; // x = scale
};

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragTexCoord;

vec3 srgb_to_linear(vec3 srgb) {
    bvec3 cutoff = greaterThan(srgb, vec3(0.04045));
    vec3 lower = srgb / 12.92;
    vec3 higher = pow((srgb + 0.055) / 1.055, vec3(2.4));
    return mix(lower, higher, cutoff);
}

void main() {
    gl_Position = proj * vec4(inPosition * extra.x, 0.0, 1.0);
    fragColor = vec4(srgb_to_linear(inColor.rgb), inColor.a);
    fragTexCoord = inTexCoord;
}
