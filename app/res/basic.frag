#version 450

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in	vec3 inPos;

layout(set = 0, binding = 0) uniform WorldUniforms {
	mat4 proj;
	mat4 view;

	vec4 camera_pos;
	vec4 camera_dir;
};

layout(location = 0) out vec4 outColor;

void main() {
	vec3 dir_to_frag = normalize(inPos - camera_pos.xyz);
	float lighting = max(0.9950472*pow(dot(camera_dir.xyz, dir_to_frag), 6.0), 0.1) * max(dot(inNormal, -dir_to_frag), 0.2);
	outColor = vec4(lighting * inColor, 1);
}
