#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outColor;

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
	vec3 normal = mat3(transpose(inverse(model))) * inNormal;
	vec3 light_dir =  normalize(camera_pos.xyz - pos.xyz);
	outColor = max(dot(normal, light_dir), 0.) * vec3(1.);
	gl_Position = pos;
}
