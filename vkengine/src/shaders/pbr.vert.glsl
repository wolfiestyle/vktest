#version 450
layout(binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    mat4 model;
    vec4 light;
    vec4 light_color;
    vec4 view_pos;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in vec4 inColor;

layout(location = 0) out FragOut {
    vec3 Pos;
    vec2 TexCoord;
    vec3 Color;
    mat3 TBN;
} frag;

void main() {
    gl_Position = mvp * vec4(inPosition, 1.0);
    frag.Pos = (model * vec4(inPosition, 1.0)).xyz;
    frag.TexCoord = inTexCoord;
    frag.Color = inColor.rgb;
    vec3 bitangent = cross(inNormal, inTangent.xyz) * inTangent.w;
    vec3 t = normalize((model * vec4(inTangent.xyz, 0.0)).xyz);
    vec3 b = normalize((model * vec4(bitangent, 0.0)).xyz);
    vec3 n = normalize((model * vec4(inNormal, 0.0)).xyz);
    frag.TBN = mat3(t, b, n);
}
