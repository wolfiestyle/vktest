#version 450
layout(binding = 0) uniform ObjectUniforms {
    mat4 mvp;
    mat4 model;
    vec4 light_dir;
    vec4 light_color;
    vec4 view_pos;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in vec4 inColor;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec3 fragColor;

void main() {
    gl_Position = mvp * vec4(inPosition, 1.0);
    fragPos = (model * vec4(inPosition, 1.0)).xyz;
    fragNormal = inNormal;
    fragTexCoord = inTexCoord;
    fragColor = inColor.rgb;
}
