#version 450

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

void main() {
	outColor = vec4(inColor, 1.0) * clamp(dot(inNormal, normalize(vec3(2, -1, 0))), 0.1, 1);
}