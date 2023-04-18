#version 450
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0) uniform samplerCube inputTex;
layout(binding = 1, rgba16f) uniform writeonly imageCube outputTex;

layout(push_constant) uniform PushConstants {
    float delta_phi;
    float deltha_theta;
};

const float PI = 3.14159265359;
const float TwoPI = PI * 2.0;
const float HalfPI = PI * 0.5;
const float Epsilon = 0.00001;

vec3 get_sampling_vector() {
    vec2 st = gl_GlobalInvocationID.xy / vec2(imageSize(outputTex));
    vec2 uv = 2.0 * vec2(st.x, 1.0 - st.y) - vec2(1.0);
    vec3 ret;
    if (gl_GlobalInvocationID.z == 0)      ret = vec3(1.0,  uv.y, -uv.x);
    else if (gl_GlobalInvocationID.z == 1) ret = vec3(-1.0, uv.y,  uv.x);
    else if (gl_GlobalInvocationID.z == 2) ret = vec3(uv.x, 1.0, -uv.y);
    else if (gl_GlobalInvocationID.z == 3) ret = vec3(uv.x, -1.0, uv.y);
    else if (gl_GlobalInvocationID.z == 4) ret = vec3(uv.x, uv.y, 1.0);
    else if (gl_GlobalInvocationID.z == 5) ret = vec3(-uv.x, uv.y, -1.0);
    return normalize(ret);
}

void main() {
    vec3 N = get_sampling_vector();
    vec3 T = cross(N, vec3(0.0, 1.0, 0.0));
    T = mix(cross(N, vec3(1.0, 0.0, 0.0)), T, step(Epsilon, dot(T, T)));
    T = normalize(T);
    vec3 S = normalize(cross(N, T));
    uint num_samples = 0;
    vec3 color = vec3(0.0);
    for (float phi = 0.0; phi < TwoPI; phi += delta_phi) {
        float sinPhi = sin(phi);
        float cosPhi = cos(phi);
        vec3 temp = cosPhi * T + sinPhi * S;
        for (float theta = 0.0; theta < HalfPI; theta += deltha_theta) {
            float sinTheta = sin(theta);
            float cosTheta = cos(theta);
            vec3 sample_dir = cosTheta * N + sinTheta * temp;
            color += texture(inputTex, sample_dir).rgb * cosTheta * sinTheta;
            num_samples++;
        }
    }
    vec3 irradiance = color * PI / float(num_samples);
    imageStore(outputTex, ivec3(gl_GlobalInvocationID), vec4(irradiance, 1.0));
}
