#version 450

// Vertex attributes
/*
layout (location = 0) in vec4 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inColor;
*/
// Instanced attributes
layout (location = 4) in vec3 instancePos;
layout (location = 5) in float instanceScale;

struct Vertex
{
	vec4 posXYZnormalX;
	vec4 inColor;
};

layout (binding = 0) uniform UBO 
{
	mat4 projection;
	mat4 modelview;
} ubo;

layout (binding = 1) readonly buffer Vertices
{
	Vertex vertices[];
};

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec3 outViewVec;
layout (location = 3) out vec3 outLightVec;

out gl_PerVertex
{
	vec4 gl_Position;
};

void main() 
{
	Vertex vert = vertices[gl_VertexIndex];
	vec3 inColor = vert.inColor.xyz;
	outColor = inColor;
		
	uint intNormal = floatBitsToUint(vert.posXYZnormalX.w);
	outNormal = unpackSnorm4x8(intNormal).xyz;
	
	vec4 pos = vec4((vert.posXYZnormalX.xyz * instanceScale) + instancePos, 1.0);

	gl_Position = ubo.projection * ubo.modelview * pos;
	
	vec4 wPos = ubo.modelview * vec4(pos.xyz, 1.0); 
	vec4 lPos = vec4(0.0, 10.0, 50.0, 1.0);
	outLightVec = lPos.xyz - pos.xyz;
	outViewVec = -pos.xyz;	
}
