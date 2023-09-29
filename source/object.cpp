#include "object.h"

ObjectGL::ObjectGL() :
   Type( TYPE::ARBITRARY ), VAO( 0 ), VBO( 0 ), IBO( 0 ), DrawMode( 0 ), VerticesCount( 0 ), BoundingBox(),
   EmissionColor( 0.1f, 0.1f, 0.1f, 1.0f ), AmbientReflectionColor( 0.2f, 0.2f, 0.2f, 1.0f ),
   DiffuseReflectionColor( 0.8f, 0.8f, 0.8f, 1.0f ), SpecularReflectionColor( 0.8f, 0.8f, 0.8f, 1.0f ),
   SpecularReflectionExponent( 16.0f )
{
}

ObjectGL::~ObjectGL()
{
   if (IBO != 0) glDeleteBuffers( 1, &IBO );
   if (VBO != 0) glDeleteBuffers( 1, &VBO );
   if (VAO != 0) glDeleteVertexArrays( 1, &VAO );
   for (const auto& texture_id : TextureID) {
      if (texture_id != 0) glDeleteTextures( 1, &texture_id );
   }
   for (const auto& buffer : CustomBuffers) {
      if (buffer.second != 0) glDeleteBuffers( 1, &buffer.second );
   }
}

void ObjectGL::setEmissionColor(const glm::vec4& emission_color)
{
   EmissionColor = emission_color;
}

void ObjectGL::setAmbientReflectionColor(const glm::vec4& ambient_reflection_color)
{
   AmbientReflectionColor = ambient_reflection_color;
}

void ObjectGL::setDiffuseReflectionColor(const glm::vec4& diffuse_reflection_color)
{
   DiffuseReflectionColor = diffuse_reflection_color;
}

void ObjectGL::setSpecularReflectionColor(const glm::vec4& specular_reflection_color)
{
   SpecularReflectionColor = specular_reflection_color;
}

void ObjectGL::setSpecularReflectionExponent(const float& specular_reflection_exponent)
{
   SpecularReflectionExponent = specular_reflection_exponent;
}

bool ObjectGL::prepareTexture2DUsingFreeImage(const std::string& file_path, bool is_grayscale) const
{
   const FREE_IMAGE_FORMAT format = FreeImage_GetFileType( file_path.c_str(), 0 );
   FIBITMAP* texture = FreeImage_Load( format, file_path.c_str() );
   if (!texture) return false;

   FIBITMAP* texture_converted;
   const uint n_bits_per_pixel = FreeImage_GetBPP( texture );
   const uint n_bits = is_grayscale ? 8 : 32;
   if (is_grayscale) {
      texture_converted = n_bits_per_pixel == n_bits ? texture : FreeImage_GetChannel( texture, FICC_RED );
   }
   else {
      texture_converted = n_bits_per_pixel == n_bits ? texture : FreeImage_ConvertTo32Bits( texture );
   }

   const auto width = static_cast<GLsizei>(FreeImage_GetWidth( texture_converted ));
   const auto height = static_cast<GLsizei>(FreeImage_GetHeight( texture_converted ));
   GLvoid* data = FreeImage_GetBits( texture_converted );
   glTextureStorage2D( TextureID.back(), 1, is_grayscale ? GL_R8 : GL_RGBA8, width, height );
   glTextureSubImage2D( TextureID.back(), 0, 0, 0, width, height, is_grayscale ? GL_RED : GL_BGRA, GL_UNSIGNED_BYTE, data );

   FreeImage_Unload( texture_converted );
   if (n_bits_per_pixel != n_bits) FreeImage_Unload( texture );
   return true;
}

int ObjectGL::addTexture(const std::string& texture_file_path, bool is_grayscale)
{
   GLuint texture_id = 0;
   glCreateTextures( GL_TEXTURE_2D, 1, &texture_id );
   TextureID.emplace_back( texture_id );
   if (!prepareTexture2DUsingFreeImage( texture_file_path, is_grayscale )) {
      glDeleteTextures( 1, &texture_id );
      TextureID.erase( TextureID.end() - 1 );
      std::cerr << "Could not read image file " << texture_file_path.c_str() << "\n";
      return -1;
   }

   glTextureParameteri( texture_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
   glTextureParameteri( texture_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTextureParameteri( texture_id, GL_TEXTURE_WRAP_S, GL_REPEAT );
   glTextureParameteri( texture_id, GL_TEXTURE_WRAP_T, GL_REPEAT );
   glGenerateTextureMipmap( texture_id );
   return static_cast<int>(TextureID.size() - 1);
}

void ObjectGL::addTexture(int width, int height, bool is_grayscale)
{
   GLuint texture_id = 0;
   glCreateTextures( GL_TEXTURE_2D, 1, &texture_id );
   glTextureStorage2D(
      texture_id, 1,
      is_grayscale ? GL_R8 : GL_RGBA8,
      width, height
   );
   glTextureParameteri( texture_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
   glTextureParameteri( texture_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTextureParameteri( texture_id, GL_TEXTURE_WRAP_S, GL_REPEAT );
   glTextureParameteri( texture_id, GL_TEXTURE_WRAP_T, GL_REPEAT );
   glGenerateTextureMipmap( texture_id );
   TextureID.emplace_back( texture_id );
}

void ObjectGL::prepareTexture(bool normals_exist) const
{
   const uint offset = normals_exist ? 6 : 3;
   glVertexArrayAttribFormat( VAO, TEXTURE, 2, GL_FLOAT, GL_FALSE, offset * sizeof( GLfloat ) );
   glEnableVertexArrayAttrib( VAO, TEXTURE );
   glVertexArrayAttribBinding( VAO, TEXTURE, 0 );
}

void ObjectGL::prepareNormal() const
{
   glVertexArrayAttribFormat( VAO, NORMAL, 3, GL_FLOAT, GL_FALSE, 3 * sizeof( GLfloat ) );
   glEnableVertexArrayAttrib( VAO, NORMAL );
   glVertexArrayAttribBinding( VAO, NORMAL, 0 );
}

void ObjectGL::prepareVertexBuffer(int n_bytes_per_vertex)
{
   glCreateBuffers( 1, &VBO );
   glNamedBufferStorage( VBO, sizeof( GLfloat ) * DataBuffer.size(), DataBuffer.data(), GL_DYNAMIC_STORAGE_BIT );

   glCreateVertexArrays( 1, &VAO );
   glVertexArrayVertexBuffer( VAO, 0, VBO, 0, n_bytes_per_vertex );
   glVertexArrayAttribFormat( VAO, VERTEX, 3, GL_FLOAT, GL_FALSE, 0 );
   glEnableVertexArrayAttrib( VAO, VERTEX );
   glVertexArrayAttribBinding( VAO, VERTEX, 0 );
}

void ObjectGL::prepareIndexBuffer()
{
   assert( VAO != 0 );

   if (IBO != 0) glDeleteBuffers( 1, &IBO );

   glCreateBuffers( 1, &IBO );
   glNamedBufferStorage( IBO, sizeof( GLuint ) * IndexBuffer.size(), IndexBuffer.data(), GL_DYNAMIC_STORAGE_BIT );
   glVertexArrayElementBuffer( VAO, IBO );
}

void ObjectGL::setObject(
   GLenum draw_mode,
   const std::vector<glm::vec3>& vertices,
   const std::vector<glm::vec3>& normals,
   const std::vector<glm::vec2>& textures
)
{
   DrawMode = draw_mode;
   VerticesCount = 0;
   DataBuffer.clear();
   for (size_t i = 0; i < vertices.size(); ++i) {
      DataBuffer.push_back( vertices[i].x );
      DataBuffer.push_back( vertices[i].y );
      DataBuffer.push_back( vertices[i].z );
      DataBuffer.push_back( normals[i].x );
      DataBuffer.push_back( normals[i].y );
      DataBuffer.push_back( normals[i].z );
      DataBuffer.push_back( textures[i].x );
      DataBuffer.push_back( textures[i].y );
      VerticesCount++;
   }
   const int n_bytes_per_vertex = 8 * sizeof( GLfloat );
   prepareVertexBuffer( n_bytes_per_vertex );
   prepareNormal();
   prepareTexture( true );
}

void ObjectGL::setObject(
   GLenum draw_mode,
   const std::vector<glm::vec3>& vertices,
   const std::vector<glm::vec3>& normals,
   const std::vector<glm::vec2>& textures,
   const std::string& texture_file_path,
   bool is_grayscale
)
{
   setObject( draw_mode, vertices, normals, textures );
   addTexture( texture_file_path, is_grayscale );
}

void ObjectGL::findNormals(
   std::vector<glm::vec3>& normals,
   const std::vector<glm::vec3>& vertices,
   const std::vector<GLuint>& vertex_indices
)
{
   normals.resize( vertices.size(), glm::vec3(0.0f) );
   const auto size = static_cast<int>(vertex_indices.size());
   for (int i = 0; i < size; i += 3) {
      const GLuint n0 = vertex_indices[i];
      const GLuint n1 = vertex_indices[i + 1];
      const GLuint n2 = vertex_indices[i + 2];
      const glm::vec3 normal = glm::cross( vertices[n1] - vertices[n0], vertices[n2] - vertices[n0] );
      normals[n0] += normal;
      normals[n1] += normal;
      normals[n2] += normal;
   }
   for (auto& n : normals) n = glm::normalize( n );
}

void ObjectGL::separateObjectFile(
   const std::string& file_path,
   const std::string& out_file_root_path,
   const std::string& separator
)
{
   std::ifstream file(file_path);
   if (!file.is_open()) {
      std::cout << "The object file is not correct.\n";
      return;
   }

   std::ofstream obj;
   int file_index = 0;
   int v = 0, t = 0, n = 0;
   int next_v = 0, next_t = 0, next_n = 0;
   while (!file.eof()) {
      std::string line;
      std::getline( file, line );

      const std::regex space_delimiter("[ ]");
      const std::sregex_token_iterator line_it(line.begin(), line.end(), space_delimiter, -1);
      const std::vector<std::string> parsed(line_it, std::sregex_token_iterator());
      if (parsed.empty()) continue;

      if (parsed[0] == separator) {
         if (obj.is_open()) obj.close();
         v = next_v;
         t = next_t;
         n = next_n;
         file_index++;
         obj.open( out_file_root_path + std::to_string( file_index ) + ".obj", std::ofstream::out);
      }
      else if (parsed[0] == "f") {
         std::string out = "f ";
         for (int i = 1; i <= 3; ++i) {
            const std::regex delimiter("[/\r\n]");
            const std::sregex_token_iterator it(parsed[i].begin(), parsed[i].end(), delimiter, -1);
            const std::vector<std::string> vtn(it, std::sregex_token_iterator());
            out += std::to_string( std::stoi( vtn[0] ) - v ) + "/";
            if (isNumber( vtn[1] )) out += std::to_string( std::stoi( vtn[1] ) - t ) + "/";
            else out += "/";
            if (isNumber( vtn[2] )) out += std::to_string( std::stoi( vtn[2] ) - n );
            out += " ";
         }
         obj << out << "\n";
      }
      else if (parsed[0] == "v") {
         next_v++;
         obj << line + "\n";
      }
      else if (parsed[0] == "vt") {
         next_t++;
         obj << line + "\n";
      }
      else if (parsed[0] == "vn") {
         next_n++;
         obj << line + "\n";
      }
      else obj << line + "\n";
   }
}

bool ObjectGL::readObjectFile(
   std::vector<glm::vec3>& vertices,
   std::vector<glm::vec3>& normals,
   std::vector<glm::vec2>& textures,
   const std::string& file_path
)
{
   std::ifstream file(file_path);
   if (!file.is_open()) {
      std::cout << "The object file is not correct.\n";
      return false;
   }

   bool found_normals = false, found_textures = false;
   std::vector<glm::vec3> vertex_buffer, normal_buffer;
   std::vector<glm::vec2> texture_buffer;
   std::vector<GLuint> vertex_indices, normal_indices, texture_indices;
   glm::vec3 min_point(std::numeric_limits<float>::max());
   glm::vec3 max_point(std::numeric_limits<float>::lowest());
   while (!file.eof()) {
      std::string word;
      file >> word;

      if (word == "v") {
         glm::vec3 vertex;
         file >> vertex.x >> vertex.y >> vertex.z;
         min_point = glm::min( vertex, min_point );
         max_point = glm::max( vertex, max_point );
         vertex_buffer.emplace_back( vertex );
      }
      else if (word == "vt") {
         glm::vec2 uv;
         file >> uv.x >> uv.y;
         texture_buffer.emplace_back( uv );
         found_textures = true;
      }
      else if (word == "vn") {
         glm::vec3 normal;
         file >> normal.x >> normal.y >> normal.z;
         normal_buffer.emplace_back( normal );
         found_normals = true;
      }
      else if (word == "f") {
         std::string face;
         const std::regex delimiter("[/]");
         for (int i = 0; i < 3; ++i) {
            file >> face;
            const std::sregex_token_iterator it(face.begin(), face.end(), delimiter, -1);
            const std::vector<std::string> vtn(it, std::sregex_token_iterator());
            vertex_indices.emplace_back( std::stoi( vtn[0] ) - 1 );
            if (found_textures && isNumber( vtn[1] )) {
               texture_indices.emplace_back( std::stoi( vtn[1] ) - 1 );
               found_textures = false;
            }
            if (found_normals && isNumber( vtn[2] )) {
               normal_indices.emplace_back( std::stoi( vtn[2] ) - 1 );
               found_normals = false;
            }
         }
      }
      else std::getline( file, word );
   }

   if (!found_normals) findNormals( normal_buffer, vertex_buffer, vertex_indices );

   vertices = std::move( vertex_buffer );
   normals = std::move( normal_buffer );
   if (found_textures && vertex_indices.size() == texture_indices.size()) textures = std::move( texture_buffer );
   IndexBuffer = std::move( vertex_indices );
   BoundingBox.MinPoint = min_point;
   BoundingBox.MaxPoint = max_point;
   return true;
}

void ObjectGL::setObject(GLenum draw_mode, const std::string& obj_file_path)
{
   DrawMode = draw_mode;
   std::vector<glm::vec3> vertices, normals;
   std::vector<glm::vec2> textures;
   if (!readObjectFile( vertices, normals, textures, obj_file_path )) return;

   const bool normals_exist = !normals.empty();
   const bool textures_exist = !textures.empty();
   for (uint i = 0; i < vertices.size(); ++i) {
      DataBuffer.push_back( vertices[i].x );
      DataBuffer.push_back( vertices[i].y );
      DataBuffer.push_back( vertices[i].z );
      if (normals_exist) {
         DataBuffer.push_back( normals[i].x );
         DataBuffer.push_back( normals[i].y );
         DataBuffer.push_back( normals[i].z );
      }
      if (textures_exist) {
         DataBuffer.push_back( textures[i].x );
         DataBuffer.push_back( textures[i].y );
      }
      VerticesCount++;
   }
   int n = 3;
   if (normals_exist) n += 3;
   if (textures_exist) n += 2;
   const auto n_bytes_per_vertex = static_cast<int>(n * sizeof( GLfloat ));
   prepareVertexBuffer( n_bytes_per_vertex );
   if (normals_exist) prepareNormal();
   if (textures_exist) prepareTexture( normals_exist );
   prepareIndexBuffer();
}

void ObjectGL::transferUniformsToShader(const ShaderGL* shader) const
{
   glUniform4fv( shader->getMaterialEmissionLocation(), 1, &EmissionColor[0] );
   glUniform4fv( shader->getMaterialAmbientLocation(), 1, &AmbientReflectionColor[0] );
   glUniform4fv( shader->getMaterialDiffuseLocation(), 1, &DiffuseReflectionColor[0] );
   glUniform4fv( shader->getMaterialSpecularLocation(), 1, &SpecularReflectionColor[0] );
   glUniform1f( shader->getMaterialSpecularExponentLocation(), SpecularReflectionExponent );
}