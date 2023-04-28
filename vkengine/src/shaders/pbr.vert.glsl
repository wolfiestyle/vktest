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

layout(push_constant) uniform PushConstants {
    vec4 base_color;
    vec3 base_pbr;
    float normal_scale;
    vec3 emissive;
    uint uv_sets;
} material;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inTexCoords[NumUVs];
layout(location = 5) in vec4 inColor;

layout(location = 0) out FragOut {
    vec3 Pos;
    vec4 Color;
    mat3 TBN;
    vec2 uvColor;
    vec2 uvMetalRough;
    vec2 uvNormal;
    vec2 uvEmissive;
    vec2 uvOcclusion;
} frag;

void main() {
    gl_Position = mvp * vec4(inPosition, 1.0);
    frag.Pos = (model * vec4(inPosition, 1.0)).xyz;
    frag.Color = inColor;
    vec3 bitangent = cross(inNormal, inTangent.xyz) * inTangent.w;
    vec3 t = normalize((model * vec4(inTangent.xyz, 0.0)).xyz);
    vec3 b = normalize((model * vec4(bitangent, 0.0)).xyz);
    vec3 n = normalize((model * vec4(inNormal, 0.0)).xyz);
    frag.TBN = mat3(t, b, n);
    uint color_uvset = bitfieldExtract(material.uv_sets, 0, UVbits);
    uint metrgh_uvset = bitfieldExtract(material.uv_sets, UVbits, UVbits);
    uint normal_uvset = bitfieldExtract(material.uv_sets, UVbits * 2, UVbits);
    uint emiss_uvset = bitfieldExtract(material.uv_sets, UVbits * 3, UVbits);
    uint occl_uvset = bitfieldExtract(material.uv_sets, UVbits * 4, UVbits);
    frag.uvColor = inTexCoords[color_uvset];
    frag.uvMetalRough = inTexCoords[metrgh_uvset];
    frag.uvNormal = inTexCoords[normal_uvset];
    frag.uvEmissive = inTexCoords[emiss_uvset];
    frag.uvOcclusion = inTexCoords[occl_uvset];
}
