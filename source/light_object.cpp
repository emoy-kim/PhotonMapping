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
   readObjectFile( obj_file_path );

   const bool normals_exist = !Normals.empty();
   const glm::mat4 vector_transform = glm::transpose( glm::inverse( transform ) );
   for (size_t i = 0; i < Vertices.size(); ++i) {
      Vertices[i] = glm::vec3(transform * glm::vec4(Vertices[i], 1.0f));
      DataBuffer.emplace_back( Vertices[i].x );
      DataBuffer.emplace_back( Vertices[i].y );
      DataBuffer.emplace_back( Vertices[i].z );
      if (normals_exist) {
         Normals[i] = glm::normalize( glm::vec3(vector_transform * glm::vec4(Normals[i], 0.0f)) );
         DataBuffer.emplace_back( Normals[i].x );
         DataBuffer.emplace_back( Normals[i].y );
         DataBuffer.emplace_back( Normals[i].z );
      }
      VerticesCount++;
   }
   int n = 3;
   if (normals_exist) n += 3;
   const auto n_bytes_per_vertex = static_cast<int>(n * sizeof( GLfloat ));
   prepareVertexBuffer( n_bytes_per_vertex );
   if (normals_exist) prepareNormal();
   prepareIndexBuffer();
   DataBuffer.clear();

   assert( Vertices.size() == 4 );

   if (!mtl_file_path.empty()) setMaterial( mtl_file_path );

   Areas.clear();
   Triangles.clear();
   const auto size = static_cast<int>(IndexBuffer.size());
   for (int i = 0; i < size; i += 3) {
      const GLuint n0 = IndexBuffer[i];
      const GLuint n1 = IndexBuffer[i + 1];
      const GLuint n2 = IndexBuffer[i + 2];
      const glm::vec3 normal = glm::cross( Vertices[n1] - Vertices[n0], Vertices[n2] - Vertices[n0] );
      Areas.emplace_back( glm::length( normal ) * 0.5f );
      Triangles.emplace_back( std::array<glm::vec3, 3>{ Vertices[n0], Vertices[n1], Vertices[n2] } );
   }
   SpotlightDirection = Normals[0];
   BoundingBox.MinPoint = glm::vec3(transform * glm::vec4(BoundingBox.MinPoint, 1.0f));
   BoundingBox.MaxPoint = glm::vec3(transform * glm::vec4(BoundingBox.MaxPoint, 1.0f));
}