#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec3 outNormal;

layout(set = 0, binding = 0) uniform CameraUniforms {
	mat4 vp;
} cameraUniforms;

layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	mat4 normal;
} modelUniforms;

void main() {
	gl_Position = modelUniforms.model * cameraUniforms.vp * vec4(inPosition, 1.0);
	outColor = vec3(inUV, 0.0);
	outNormal = mat3(modelUniforms.normal) * inNormal;
}