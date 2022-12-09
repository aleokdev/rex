#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 m;
};

void main()
{
	vec4 pos = vec4(inPosition, 1);
	pos = pos * m;
	pos /= pos.w;
	gl_Position = pos;
}
