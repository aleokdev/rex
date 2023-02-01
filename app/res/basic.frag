#version 450

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in	vec3 inPos;
layout(location = 3) in	vec3 inCameraPos;

layout(location = 0) out vec4 outColor;

void main() {
	vec3 light_dir =  normalize(inCameraPos - inPos);
	float lighting = max(dot(inNormal, light_dir), 0.3);
	outColor = vec4(lighting * inColor, 1);
}
