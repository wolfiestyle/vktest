#version 450
const int UVbits = 1;
const uint NumUVs = 1 << UVbits;

layout(binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    mat4 model;
    vec3 view_pos;
    uint num_lights;
};

layout(push_constant) uniform PushConstants {
    vec4 base_color;
    vec4 base_pbr;
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
    vec3 Normal;
    vec4 Tangent;
    vec2 uvColor;
    vec2 uvMetalRough;
    vec2 uvNormal;
    vec2 uvEmissive;
    vec2 uvOcclusion;
} frag;

void main() {
    gl_Position = mvp * vec4(inPosition, 1.0);
    frag.Pos = (model * vec4(inPosition, 1.0)).xyz;
    frag.Color = vec4(pow(inColor.rgb, vec3(2.2)), inColor.a);
    frag.Normal = normalize((model * vec4(inNormal, 0.0)).xyz);
    frag.Tangent = vec4(normalize((model * vec4(inTangent.xyz, 0.0)).xyz), inTangent.w);
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
