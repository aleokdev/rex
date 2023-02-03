#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outPos;

layout(set = 0, binding = 0) uniform WorldUniforms {
	mat4 proj;
	mat4 view;

	vec4 camera_pos;
	vec4 camera_dir;
};

layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
};

void main()
{
	vec4 pos = vec4(inPosition, 1);
	vec4 world_pos = model * pos;
	pos = proj * view * world_pos;
	outNormal = mat3(transpose(inverse(model))) * inNormal;
	outColor = inColor;
	outPos = world_pos.xyz;
	gl_Position = pos;
}
