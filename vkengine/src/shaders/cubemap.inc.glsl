#include "constants.inc.glsl"

vec3 getCubemapDir(uvec3 coord, vec2 img_size) {
    vec2 st = (vec2(coord.xy) + 0.5) / img_size;
    vec2 uv = 2.0 * vec2(st.x, 1.0 - st.y) - vec2(1.0);
    vec3 ret;
    switch (coord.z) {
        case 0: ret = vec3(1.0,  uv.y, -uv.x); break;
        case 1: ret = vec3(-1.0, uv.y,  uv.x); break;
        case 2: ret = vec3(uv.x, 1.0, -uv.y); break;
        case 3: ret = vec3(uv.x, -1.0, uv.y); break;
        case 4: ret = vec3(uv.x, uv.y, 1.0); break;
        case 5: ret = vec3(-uv.x, uv.y, -1.0); break;
    }
    return normalize(ret);
}

vec2 cubemapDirToEquirect(vec3 dir) {
    // Odd-sized faces can sample directly at a pole, where longitude is arbitrary.
    float longitude = all(equal(dir.xz, vec2(0.0))) ? 0.0 : atan(dir.z, dir.x);
    return vec2(
        0.5 + longitude / TwoPI,
        acos(dir.y) / PI
    );
}
