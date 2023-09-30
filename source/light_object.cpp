#include "light_object.h"

LightGL::LightGL() :
   ObjectGL(), TurnLightOn( true ), SpotlightCutoffAngle( 180.0f ), SpotlightFeather( 0.0f ), FallOffRadius( 1000.0f ),
   SpotlightDirection( 0.0f, 0.0f, -1.0f ), AreaLight()
{
}

void LightGL::transferUniformsToShader(const ShaderGL* shader, int index) const
{
   const glm::vec3 position =
      (AreaLight.Vertices[0] + AreaLight.Vertices[1] + AreaLight.Vertices[2] + AreaLight.Vertices[3]) * 0.25f;
   glUniform4fv( shader->getLightPositionLocation( index ), 1, &position[0] );
   glUniform4fv( shader->getLightEmissionLocation( index ), 1, &EmissionColor[0] );
   glUniform4fv( shader->getLightAmbientLocation( index ), 1, &AmbientReflectionColor[0] );
   glUniform4fv( shader->getLightDiffuseLocation( index ), 1, &DiffuseReflectionColor[0] );
   glUniform4fv( shader->getLightSpecularLocation( index ), 1, &SpecularReflectionColor[0] );
   glUniform3fv( shader->getLightSpotlightDirectionLocation( index ), 1, &SpotlightDirection[0] );
   glUniform1f( shader->getLightSpotlightCutoffAngleLocation( index ), SpotlightCutoffAngle );
   glUniform1f( shader->getLightSpotlightFeatherLocation( index ), SpotlightFeather );
   glUniform1f( shader->getLightFallOffRadiusLocation( index ), FallOffRadius );
}

void LightGL::setObjectWithTransform(
   GLenum draw_mode,
   const TYPE& type,
   const glm::mat4& transform,
   const std::string& obj_file_path,
   const std::string& mtl_file_path
)
{
   Type = type;
   DrawMode = draw_mode;
   std::vector<glm::vec3> vertices, normals;
   std::vector<glm::vec2> textures;
   readObjectFile( vertices, normals, textures, obj_file_path );

   const bool normals_exist = !normals.empty();
   const glm::mat4 vector_transform = glm::transpose( glm::inverse( transform ) );
   for (uint i = 0; i < vertices.size(); ++i) {
      vertices[i] = glm::vec3(transform * glm::vec4(vertices[i], 1.0f));
      DataBuffer.push_back( vertices[i].x );
      DataBuffer.push_back( vertices[i].y );
      DataBuffer.push_back( vertices[i].z );
      if (normals_exist) {
         normals[i] = glm::normalize( glm::vec3(vector_transform * glm::vec4(normals[i], 0.0f)) );
         DataBuffer.push_back( normals[i].x );
         DataBuffer.push_back( normals[i].y );
         DataBuffer.push_back( normals[i].z );
      }
      VerticesCount++;
   }
   int n = 3;
   if (normals_exist) n += 3;
   const auto n_bytes_per_vertex = static_cast<int>(n * sizeof( GLfloat ));
   prepareVertexBuffer( n_bytes_per_vertex );
   if (normals_exist) prepareNormal();
   prepareIndexBuffer();

   assert( vertices.size() == 4 );

   if (!mtl_file_path.empty()) setMaterial( mtl_file_path );

   const glm::vec3 a0 = glm::cross( vertices[1] - vertices[0], vertices[2] - vertices[0] );
   const glm::vec3 a1 = glm::cross( vertices[3] - vertices[0], vertices[2] - vertices[0] );
   AreaLight.Area = (glm::length( a0 ) + glm::length( a1 )) * 0.5f;
   AreaLight.Emission = EmissionColor;
   AreaLight.Normal = normals[0];
   AreaLight.Vertices[0] = vertices[0];
   AreaLight.Vertices[1] = vertices[1];
   AreaLight.Vertices[2] = vertices[2];
   AreaLight.Vertices[3] = vertices[3];

   // need to trasnform BoundingBox ...
}