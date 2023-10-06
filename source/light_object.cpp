#include "light_object.h"

LightGL::LightGL() :
   ObjectGL(), TurnLightOn( true ), SpotlightCutoffAngle( 180.0f ), SpotlightFeather( 0.0f ),
   FallOffRadius( 1000.0f ), SpotlightDirection( 0.0f, 0.0f, -1.0f ), Areas{}, Triangles{}
{
}

glm::vec4 LightGL::getCentroid() const
{
   glm::vec3 sum(0.0f);
   for (const auto& t : Triangles) {
      sum += t[0] + t[1] + t[2];
   }
   return { sum / static_cast<float>(Triangles.size() * 3), 1.0f };
}

void LightGL::transferUniformsToShader(const ShaderGL* shader, int index) const
{
   const GLuint program = shader->getShaderProgram();
   glProgramUniform4fv( program, shader->getLightPositionLocation( index ), 1, &getCentroid()[0] );
   glProgramUniform4fv( program, shader->getLightEmissionLocation( index ), 1, &EmissionColor[0] );
   glProgramUniform4fv( program, shader->getLightAmbientLocation( index ), 1, &AmbientReflectionColor[0] );
   glProgramUniform4fv( program, shader->getLightDiffuseLocation( index ), 1, &DiffuseReflectionColor[0] );
   glProgramUniform4fv( program, shader->getLightSpecularLocation( index ), 1, &SpecularReflectionColor[0] );
   glProgramUniform3fv( program, shader->getLightSpotlightDirectionLocation( index ), 1, &SpotlightDirection[0] );
   glProgramUniform1f( program, shader->getLightSpotlightCutoffAngleLocation( index ), SpotlightCutoffAngle );
   glProgramUniform1f( program, shader->getLightSpotlightFeatherLocation( index ), SpotlightFeather );
   glProgramUniform1f( program, shader->getLightFallOffRadiusLocation( index ), FallOffRadius );
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
      DataBuffer.emplace_back( vertices[i].x );
      DataBuffer.emplace_back( vertices[i].y );
      DataBuffer.emplace_back( vertices[i].z );
      if (normals_exist) {
         normals[i] = glm::normalize( glm::vec3(vector_transform * glm::vec4(normals[i], 0.0f)) );
         DataBuffer.emplace_back( normals[i].x );
         DataBuffer.emplace_back( normals[i].y );
         DataBuffer.emplace_back( normals[i].z );
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

   Areas.clear();
   Triangles.clear();
   const auto size = static_cast<int>(IndexBuffer.size());
   for (int i = 0; i < size; i += 3) {
      const GLuint n0 = IndexBuffer[i];
      const GLuint n1 = IndexBuffer[i + 1];
      const GLuint n2 = IndexBuffer[i + 2];
      const glm::vec3 normal = glm::cross( vertices[n1] - vertices[n0], vertices[n2] - vertices[n0] );
      Areas.emplace_back( glm::length( normal ) * 0.5f );
      Triangles.emplace_back( std::array<glm::vec3, 3>{ vertices[n0], vertices[n1], vertices[n2] } );
   }
   SpotlightDirection = normals[0];
   BoundingBox.MinPoint = glm::vec3(transform * glm::vec4(BoundingBox.MinPoint, 1.0f));
   BoundingBox.MaxPoint = glm::vec3(transform * glm::vec4(BoundingBox.MaxPoint, 1.0f));
   DataBuffer.clear();
   IndexBuffer.clear();
}