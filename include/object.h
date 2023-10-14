#pragma once

#include "shader.h"

struct Rect
{
   alignas(16) glm::vec3 MinPoint;
   alignas(16) glm::vec3 MaxPoint;

   Rect() : MinPoint(), MaxPoint() {}
};

class ObjectGL
{
public:
   enum class TYPE { ARBITRARY = 0, PLANE, SPHERE, BOX, LIGHT  };

   enum class MATERIAL_TYPE { LAMBERT = 0, MIRROR, GLASS };

   enum LOCATION_LAYOUT { VERTEX = 0, NORMAL, TEXTURE };

   ObjectGL();
   virtual ~ObjectGL();

   void setObject(
      GLenum draw_mode,
      const std::vector<glm::vec3>& vertices,
      const std::vector<glm::vec3>& normals,
      const std::vector<glm::vec2>& textures
   );
   void setObject(
      GLenum draw_mode,
      const std::vector<glm::vec3>& vertices,
      const std::vector<glm::vec3>& normals,
      const std::vector<glm::vec2>& textures,
      const std::string& texture_file_path,
      bool is_grayscale = false
   );
   virtual void setObject(
      GLenum draw_mode,
      const TYPE& type,
      const std::string& obj_file_path,
      const std::string& mtl_file_path
   );
   virtual void setObjectWithTransform(
      GLenum draw_mode,
      const TYPE& type,
      const glm::mat4& transform,
      const std::string& obj_file_path,
      const std::string& mtl_file_path
   );
   int addTexture(const std::string& texture_file_path, bool is_grayscale = false);
   void addTexture(int width, int height, bool is_grayscale = false);
   void releaseCustomBuffer(const std::string& name)
   {
      const auto it = CustomBuffers.find( name );
      if (it != CustomBuffers.end()) {
         glDeleteBuffers( 1, &it->second );
         CustomBuffers.erase( it );
      }
   }
   void clear()
   {
      Vertices.clear();
      Normals.clear();
      Textures.clear();
   }
   [[nodiscard]] bool isLight() const { return Type == TYPE::LIGHT; }
   [[nodiscard]] bool isLambert() const { return MaterialType == MATERIAL_TYPE::LAMBERT; }
   [[nodiscard]] int getMaterialType() const { return static_cast<int>(MaterialType); }
   [[nodiscard]] GLuint getVAO() const { return VAO; }
   [[nodiscard]] GLuint getVBO() const { return VBO; }
   [[nodiscard]] GLuint getIBO() const { return IBO; }
   [[nodiscard]] GLenum getDrawMode() const { return DrawMode; }
   [[nodiscard]] GLsizei getIndexNum() const { return IndexNum; }
   [[nodiscard]] GLsizei getVertexNum() const { return VerticesCount; }
   [[nodiscard]] GLuint getTextureID(int index) const { return TextureID[index]; }
   [[nodiscard]] Rect getBoundingBox() const { return BoundingBox; }
   [[nodiscard]] glm::vec4 getEmissionColor() const { return EmissionColor; }
   [[nodiscard]] glm::vec4 getAmbientReflectionColor() const { return AmbientReflectionColor; }
   [[nodiscard]] glm::vec4 getDiffuseReflectionColor() const { return DiffuseReflectionColor; }
   [[nodiscard]] glm::vec4 getSpecularReflectionColor() const { return SpecularReflectionColor; }
   [[nodiscard]] float getSpecularReflectionExponent() const { return SpecularReflectionExponent; }
   [[nodiscard]] const std::vector<GLuint>& getIndices() const { return IndexBuffer; }
   [[nodiscard]] const std::vector<glm::vec3>& getVertices() const { return Vertices; }
   [[nodiscard]] const std::vector<glm::vec3>& getNormals() const { return Normals; }

   template<typename T>
   [[nodiscard]] GLuint addCustomBufferObject(const std::string& name, int data_size)
   {
      GLuint buffer = 0;
      glCreateBuffers( 1, &buffer );
      glNamedBufferStorage( buffer, sizeof( T ) * data_size, nullptr, GL_DYNAMIC_STORAGE_BIT );
      CustomBuffers[name] = buffer;
      return buffer;
   }

protected:
   TYPE Type;
   MATERIAL_TYPE MaterialType;
   GLuint VAO;
   GLuint VBO;
   GLuint IBO;
   GLenum DrawMode;
   GLsizei IndexNum;
   GLsizei VerticesCount;
   Rect BoundingBox;
   std::vector<GLuint> TextureID;
   std::vector<GLfloat> DataBuffer;
   std::vector<GLuint> IndexBuffer;
   std::vector<glm::vec3> Vertices;
   std::vector<glm::vec3> Normals;
   std::vector<glm::vec2> Textures;
   std::map<std::string, GLuint> CustomBuffers;
   glm::vec4 EmissionColor;
   glm::vec4 AmbientReflectionColor; // It is usually set to the same color with DiffuseReflectionColor.
                                     // Otherwise, it should be in balance with DiffuseReflectionColor.
   glm::vec4 DiffuseReflectionColor; // the intrinsic color
   glm::vec4 SpecularReflectionColor;
   float SpecularReflectionExponent;
   float RefractiveIndex;

   [[nodiscard]] bool prepareTexture2DUsingFreeImage(const std::string& file_path, bool is_grayscale) const;
   void prepareNormal() const;
   void prepareTexture(bool normals_exist) const;
   void prepareVertexBuffer(int n_bytes_per_vertex);
   void prepareIndexBuffer();
   static void findNormals(
      std::vector<glm::vec3>& normals,
      const std::vector<glm::vec3>& vertices,
      const std::vector<GLuint>& vertex_indices
   );
   void readObjectFile(const std::string& file_path);
   void setMaterial(const std::string& mtl_file_path);
   [[nodiscard]] static bool isNumber(const std::string& n)
   {
      return !n.empty() && std::find_if_not( n.begin(), n.end(), [](auto c) { return std::isdigit( c ); } ) == n.end();
   }
};