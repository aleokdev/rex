#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outPos;
layout(location = 3) out vec3 outCameraPos;

layout(set = 0, binding = 0) uniform WorldUniforms {
	mat4 proj;
	mat4 view;

	vec4 camera_pos;
};

layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
};

void main()
{
	vec4 pos = vec4(inPosition, 1);
	pos = proj * view * model * pos;
	outNormal = mat3(transpose(inverse(model))) * inNormal;
	outColor = inColor;
	outPos = pos.xyz;
	outCameraPos = camera_pos.xyz;
	gl_Position = pos;
}
