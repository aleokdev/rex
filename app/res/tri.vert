#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outColor;

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 m;
};

void main()
{
	vec4 pos = vec4(inPosition, 1);
	pos = m * pos;
	//pos /= pos.w;
	outColor = vec3((gl_VertexIndex % 3) == 0, (gl_VertexIndex % 3) == 1, (gl_VertexIndex % 3) == 2);
	gl_Position = pos;
}
