#version 450
layout(constant_id = 0) const uint NumLights = 1;
const int UVbits = 1;
const uint NumUVs = 1 << UVbits;

struct LightData {
    vec4 pos;
    vec4 color;
};

layout(binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    mat4 model;
    vec4 view_pos;
    LightData lights[NumLights];
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inTexCoord_0;
layout(location = 4) in vec2 inTexCoord_1;
layout(location = 5) in vec4 inColor;

layout(location = 0) out FragOut {
    vec3 Pos;
    vec2 TexCoord[NumUVs];
    vec4 Color;
    mat3 TBN;
} frag;

void main() {
    gl_Position = mvp * vec4(inPosition, 1.0);
    frag.Pos = (model * vec4(inPosition, 1.0)).xyz;
    frag.TexCoord = vec2[](inTexCoord_0, inTexCoord_1);
    frag.Color = inColor;
    vec3 bitangent = cross(inNormal, inTangent.xyz) * inTangent.w;
    vec3 t = normalize((model * vec4(inTangent.xyz, 0.0)).xyz);
    vec3 b = normalize((model * vec4(bitangent, 0.0)).xyz);
    vec3 n = normalize((model * vec4(inNormal, 0.0)).xyz);
    frag.TBN = mat3(t, b, n);
}
