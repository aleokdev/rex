#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec3 outNormal;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 mvp;
	mat4 normal;
} uniforms;

void main() {
	gl_Position = uniforms.mvp * vec4(inPosition, 1.0);
	outColor = vec3(inUV, 0.0);
	outNormal = mat3(uniforms.normal) * inNormal;
}