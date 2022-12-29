#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outColor;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 p;
	mat4 v;

	vec4 camera_pos;
};

void main()
{
	vec4 pos = vec4(inPosition, 1);
	pos = p * v * pos;
	vec3 light_dir =  normalize(camera_pos.xyz - pos.xyz);
	outColor = max(dot(inNormal, light_dir), 0.) * vec3(1.);
	gl_Position = pos;
}
