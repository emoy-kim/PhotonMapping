#include "host/photon_map.h"

void PhotonMap::findNormals(
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

void PhotonMap::readObjectFile(Rect& box, const std::string& file_path)
{
   std::ifstream file(file_path);

   assert( file.is_open() );

   bool found_normals = false;
   std::vector<glm::vec3> vertex_buffer, normal_buffer;
   std::vector<GLuint> vertex_indices, normal_indices;
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
            if (found_normals && isNumber( vtn[2] )) {
               normal_indices.emplace_back( std::stoi( vtn[2] ) - 1 );
               found_normals = false;
            }
         }
      }
      else std::getline( file, word );
   }

   if (!found_normals) findNormals( normal_buffer, vertex_buffer, vertex_indices );

   box.MinPoint = min_point;
   box.MaxPoint = max_point;
   VertexSizes.emplace_back( vertex_buffer.size() );
   IndexSizes.emplace_back( vertex_indices.size() );
   Vertices.insert(
      Vertices.end(),
      std::make_move_iterator( vertex_buffer.begin() ),
      std::make_move_iterator( vertex_buffer.end() )
   );
   Normals.insert(
      Normals.end(),
      std::make_move_iterator( normal_buffer.begin() ),
      std::make_move_iterator( normal_buffer.end() )
   );
   Indices.insert(
      Indices.end(),
      std::make_move_iterator( vertex_indices.begin() ),
      std::make_move_iterator( vertex_indices.end() )
   );
}

PhotonMap::Material PhotonMap::getMaterial(const std::string& mtl_file_path)
{
   std::ifstream file(mtl_file_path);

   assert( file.is_open() );

   Material material;
   while (!file.eof()) {
      std::string line;
      std::getline( file, line );

      const std::regex space_delimiter("[ ]");
      const std::sregex_token_iterator line_it(line.begin(), line.end(), space_delimiter, -1);
      const std::vector<std::string> parsed(line_it, std::sregex_token_iterator());
      if (parsed.empty()) continue;

      if (parsed[0] == "Ka") {
         material.Ambient.r = std::stof( parsed[1] );
         material.Ambient.g = std::stof( parsed[2] );
         material.Ambient.b = std::stof( parsed[3] );
      }
      else if (parsed[0] == "Kd") {
         material.Diffuse.r = std::stof( parsed[1] );
         material.Diffuse.g = std::stof( parsed[2] );
         material.Diffuse.b = std::stof( parsed[3] );
      }
      else if (parsed[0] == "Ks") {
         material.Specular.r = std::stof( parsed[1] );
         material.Specular.g = std::stof( parsed[2] );
         material.Specular.b = std::stof( parsed[3] );
      }
      else if (parsed[0] == "Ke") {
         material.Emission.r = std::stof( parsed[1] );
         material.Emission.g = std::stof( parsed[2] );
         material.Emission.b = std::stof( parsed[3] );
      }
      else if (parsed[0] == "Ns") material.SpecularExponent = std::stof( parsed[1] );
      else if (parsed[0] == "Ni") material.RefractiveIndex = std::stof( parsed[1] );
      else if (parsed[0] == "illum") {
         switch (std::stoi( parsed[1] )) {
            case 5: material.MaterialType = MATERIAL_TYPE::MIRROR; break;
            case 7: material.MaterialType = MATERIAL_TYPE::GLASS; break;
            default: material.MaterialType = MATERIAL_TYPE::LAMBERT; break;
         }
      }
   }
   return material;
}

void PhotonMap::setObjects(const std::vector<std::tuple<std::string, std::string, glm::mat4>>& objects)
{
   for (size_t i = 0; i < objects.size(); ++i) {
      Rect box;
      readObjectFile( box, std::get<0>( objects[i] ) );
      Materials.emplace_back( getMaterial( std::get<1>( objects[i] ) ) );
      ToWorlds.emplace_back( std::get<2>( objects[i] ) );
      WorldBounds.emplace_back(
         glm::vec3(ToWorlds[i] * glm::vec4(box.MinPoint, 1.0f)),
         glm::vec3(ToWorlds[i] * glm::vec4(box.MaxPoint, 1.0f))
      );
   }
}

void PhotonMap::setLights(const std::vector<std::tuple<std::string, std::string, glm::mat4>>& lights)
{
   for (size_t i = 0; i < lights.size(); ++i) {
      Rect box;
      const auto offset = static_cast<int>(Vertices.size());
      const auto index_offset = static_cast<int>(Indices.size());
      readObjectFile( box, std::get<0>( lights[i] ) );
      Materials.emplace_back( getMaterial( std::get<1>( lights[i] ) ) );
      ToWorlds.emplace_back( std::get<2>( lights[i] ) );
      WorldBounds.emplace_back(
         glm::vec3(ToWorlds.back() * glm::vec4(box.MinPoint, 1.0f)),
         glm::vec3(ToWorlds.back() * glm::vec4(box.MaxPoint, 1.0f))
      );

      for (int j = index_offset; j < static_cast<int>(Indices.size()); j += 3) {
         const GLuint n0 = offset + Indices[j];
         const GLuint n1 = offset + Indices[j + 1];
         const GLuint n2 = offset + Indices[j + 2];
         const glm::vec3 normal = glm::cross( Vertices[n1] - Vertices[n0], Vertices[n2] - Vertices[n0] );
         AreaLights.emplace_back(
            glm::length( normal ) * 0.5f,
            Materials.back().Emission,
            Normals[n0],
            Vertices[n0], Vertices[n1], Vertices[n2],
            ToWorlds.back()
         );
      }
   }
}

// this hemisphere is towards the y-axis, and its lower plane is the xz-plane.
glm::vec3 PhotonMap::getRandomPointInUnitHemisphere(float& pdf)
{
   float phi = glm::two_pi<float>() * getRandomValue( 0.0f, 1.0f ); // [0, 2pi]
   float theta = acos( getRandomValue( -1.0f, 1.0f ) ) * 0.5f; // [0, pi/2]
   float cos_theta = cos( theta );
   pdf *= cos_theta * glm::one_over_pi<float>();
   return glm::vec3(std::cos( phi ) * std::sin( theta ), cos_theta, std::sin( phi ) * std::sin( theta ));
}

glm::vec3 PhotonMap::getSamplePointAroundAxis(float& pdf, const glm::vec3& v)
{
   glm::vec3 u = std::abs( v.y ) < 0.9f ?
      glm::normalize( glm::cross( v, glm::vec3(0.0f, 1.0f, 0.0f) ) ) :
      glm::normalize( glm::cross( v, glm::vec3(0.0f, 0.0f, 1.0f) ) );
   glm::vec3 n = glm::normalize( glm::cross( u, v ) );
   return glm::mat3(u, v, n) * getRandomPointInUnitHemisphere( pdf );
}

glm::vec3 PhotonMap::getSampleRayFromLight(glm::vec3& ray_origin, glm::vec3& ray_direction)
{
   const int light_index = getRandomValue( 0.0f, 1.0f ) > 0.5f ? 0 : 1;
   const glm::vec3 v0 = AreaLights[light_index].Vertex0;
   const glm::vec3 v1 = AreaLights[light_index].Vertex1;
   const glm::vec3 v2 = AreaLights[light_index].Vertex2;
   const float a = getRandomValue( 0.0f, 1.0f );
   const float b = getRandomValue( 0.0f, 1.0f );
   ray_origin = (1.0f - a - b) * v0 + a * v1 + b * v2;
   ray_origin = AreaLights[light_index].ToWorld * glm::vec4(ray_origin, 1.0f);

   const glm::vec3 normal = AreaLights[light_index].Normal;
   float pdf = 1.0f / (AreaLights[light_index].Area * static_cast<float>(AreaLights.size()));
   ray_direction = getSamplePointAroundAxis( pdf, normal );
   const glm::vec3 power = AreaLights[light_index].Emission / pdf * std::abs( glm::dot( ray_direction, normal ) );

   ray_direction = glm::transpose( glm::inverse( AreaLights[light_index].ToWorld ) ) * glm::vec4(ray_direction, 0.0f);
   return power;
}

bool PhotonMap::intersectWithBox(const glm::vec3& ray_origin, const glm::vec3& ray_direction, const Rect& box)
{
   float exit = std::numeric_limits<float>::infinity();
   float enter = -std::numeric_limits<float>::infinity();
   if (std::abs( ray_direction.x ) > 0.001f) {
      const float t_min = (box.MinPoint.x - ray_origin.x) / ray_direction.x;
      const float t_max = (box.MaxPoint.x - ray_origin.x) / ray_direction.x;
      const float t_enter = std::min( t_min, t_max );
      const float t_exit = std::max( t_min, t_max );
      if (t_enter > enter) enter = t_enter;
      if (t_exit < exit) exit = t_exit;
      if (enter > exit || exit < 0.0f) return false;
   }
   else if (ray_origin.x < box.MinPoint.x || box.MaxPoint.x < ray_origin.x) return false;

   if (std::abs( ray_direction.y ) > 0.001f) {
      const float t_min = (box.MinPoint.y - ray_origin.y) / ray_direction.y;
      const float t_max = (box.MaxPoint.y - ray_origin.y) / ray_direction.y;
      const float t_enter = std::min( t_min, t_max );
      const float t_exit = std::max( t_min, t_max );
      if (t_enter > enter) enter = t_enter;
      if (t_exit < exit) exit = t_exit;
      if (enter > exit || exit < 0.0f) return false;
   }
   else if (ray_origin.y < box.MinPoint.y || box.MaxPoint.y < ray_origin.y) return false;

   if (std::abs( ray_direction.z ) > 0.001f) {
      const float t_min = (box.MinPoint.z - ray_origin.z) / ray_direction.z;
      const float t_max = (box.MaxPoint.z - ray_origin.z) / ray_direction.z;
      const float t_enter = std::min( t_min, t_max );
      const float t_exit = std::max( t_min, t_max );
      if (t_enter > enter) enter = t_enter;
      if (t_exit < exit) exit = t_exit;
      if (enter > exit || exit < 0.0f) return false;
   }
   else if (ray_origin.z < box.MinPoint.z || box.MaxPoint.z < ray_origin.z) return false;

   //distance = enter > zero ? enter : exit;
   return true;
}

bool PhotonMap::intersectWithTriangle(
   glm::vec3& tuv,
   const glm::vec3& ray_origin,
   const glm::vec3& ray_direction,
   const glm::vec3& p0,
   const glm::vec3& p1,
   const glm::vec3& p2
)
{
   const glm::vec3 e1 = p1 - p0;
   const glm::vec3 e2 = p2 - p0;
   const glm::vec3 q = glm::cross( ray_direction, e2 );
   const float det = glm::dot( e1, q );
   if (std::abs( det ) < 1e-5f) return false;

   const float f = 1.0f / det;
   const glm::vec3 s = ray_origin - p0;
   const float u = f * glm::dot( s, q );
   if (u < 0.0f) return false;

   const glm::vec3 r = glm::cross( s, e1 );
   const float v = f * glm::dot( ray_direction, r );
   if (v < 0.0f || u + v > 1.0f) return false;

   const float t = f * glm::dot( e2, r );
   if (t <= 0.0f) return false;

   tuv = glm::vec3(t, u, v);
   return true;
}

bool PhotonMap::findIntersection(
   IntersectionInfo& intersection,
   const glm::vec3& ray_origin,
   const glm::vec3& ray_direction
) const
{
   bool intersect = false;
   int offset = 0, index_offset = 0;
   float distance = std::numeric_limits<float>::infinity();
   const auto object_num = static_cast<int>(Materials.size());
   for (int i = 0; i < object_num; ++i) {
      const glm::mat4 vector_transform = glm::transpose( glm::inverse( ToWorlds[i] ) );
      if (intersectWithBox( ray_origin, ray_direction, WorldBounds[i] )) {
         for (int j = 0; j < IndexSizes[i]; j += 3) {
            const int k0 = offset + int(Indices[index_offset + j]);
            const int k1 = offset + int(Indices[index_offset + j + 1]);
            const int k2 = offset + int(Indices[index_offset + j + 2]);
            const glm::vec3 p0(ToWorlds[i] * glm::vec4(Vertices[k0], 1.0f));
            const glm::vec3 p1(ToWorlds[i] * glm::vec4(Vertices[k1], 1.0f));
            const glm::vec3 p2(ToWorlds[i] * glm::vec4(Vertices[k2], 1.0f));

            glm::vec3 tuv;
            if (intersectWithTriangle( tuv, ray_origin, ray_direction, p0, p1, p2 )) {
               if (distance > tuv.x) {
                  const glm::vec3 n0 = glm::normalize( glm::vec3(vector_transform * glm::vec4(Normals[k0], 0.0f)) );
                  const glm::vec3 n1 = glm::normalize( glm::vec3(vector_transform * glm::vec4(Normals[k1], 0.0f)) );
                  const glm::vec3 n2 = glm::normalize( glm::vec3(vector_transform * glm::vec4(Normals[k2], 0.0f)) );
                  distance = tuv.x;
                  intersection.ObjectIndex = i;
                  intersection.Normal = (n0 + n1 + n2) / 3.0f;
                  intersection.Position = ray_origin + tuv.x * ray_direction;
                  intersection.ShadingNormal = (1.0f - tuv.y - tuv.z) * n0 + tuv.y * n1 + tuv.z * n2;
                  intersect = true;
               }
            }
         }
      }
      offset += VertexSizes[i];
      index_offset += IndexSizes[i];
   }
   return intersect;
}

float PhotonMap::correctShadingNormal(
   const glm::vec3& wo,
   const glm::vec3& wi,
   const glm::vec3& normal,
   const glm::vec3& shading_normal
)
{
   const float wo_dot_n = glm::dot( wo, normal );
   const float wi_dot_n = glm::dot( wi, normal );
   const float wo_dot_sn = glm::dot( wo, shading_normal );
   const float wi_dot_sn = glm::dot( wi, shading_normal );
   if (wo_dot_n * wo_dot_sn <= 0.0f || wi_dot_n * wi_dot_sn <= 0.0f) return 0.0f;

   const float a = std::abs( wo_dot_n * wi_dot_sn );
   return a < 1e-5f ? 1.0f : std::abs( wo_dot_sn * wi_dot_n ) / a;
}

glm::vec3 PhotonMap::getNextSampleRay(
   glm::vec3& ray_origin,
   glm::vec3& ray_direction,
   const IntersectionInfo& intersection
)
{
   float pdf = 1.0f;
   glm::vec3 outgoing, brdf;
   const glm::vec3 incoming = -ray_direction;
   if (Materials[intersection.ObjectIndex].MaterialType == MATERIAL_TYPE::LAMBERT) {
      outgoing = getSamplePointAroundAxis( pdf, intersection.ShadingNormal );
      if (dot( incoming, intersection.ShadingNormal ) <= 0.0f) brdf = glm::vec3(0.0f);
      else brdf = Materials[intersection.ObjectIndex].Diffuse * glm::one_over_pi<float>();
   }
   else if (Materials[intersection.ObjectIndex].MaterialType == MATERIAL_TYPE::MIRROR) {
      outgoing = glm::reflect( -incoming, intersection.ShadingNormal );
      float d = std::abs( glm::dot( outgoing, intersection.ShadingNormal ) );
      brdf = d < 1e-5f ? glm::vec3(0.0f) : glm::vec3(1.0f / d);
   }
   else if (Materials[intersection.ObjectIndex].MaterialType == MATERIAL_TYPE::GLASS) {
      // need to update correctly ...
      outgoing = glm::reflect( -incoming, intersection.ShadingNormal );
      float d = std::abs( glm::dot( outgoing, intersection.ShadingNormal ) );
      brdf = d < 1e-5f ? glm::vec3(0.0f) : glm::vec3(1.0f / d);
   }

   ray_origin = intersection.Position;
   ray_direction = normalize( outgoing );
   return brdf * correctShadingNormal( outgoing, incoming, intersection.Normal, intersection.ShadingNormal ) / pdf;
}

void PhotonMap::createPhotonMap()
{
   for (int i = 0; i < MaxGlobalPhotonNum; ++i) {
      glm::vec3 ray_origin, ray_direction;
      glm::vec3 power = getSampleRayFromLight( ray_origin, ray_direction );
      for (int d = 0; d < MaxDepth; ++d) {
         if (power.x < 0.0f || power.y < 0.0f || power.z < 0.0f) break;

         IntersectionInfo intersection;
         if (!findIntersection( intersection, ray_origin, ray_direction )) break;

         if (Materials[intersection.ObjectIndex].MaterialType == MATERIAL_TYPE::LAMBERT) {
            GlobalPhotons.emplace_back( power, intersection.Position, -ray_direction );
            if (static_cast<int>(GlobalPhotons.size()) >= MaxGlobalPhotonNum) return;
         }

         if (i > 0) {
            const float russian_roulette = std::min( std::max( power.x, std::max( power.y, power.z ) ), 1.0f );
            if (getRandomValue( 0.0f, 1.0f ) < russian_roulette) power /= russian_roulette;
            else break;
         }

         power *= getNextSampleRay( ray_origin, ray_direction, intersection );
      }
   }
}

//std::vector<glm::vec3> PhotonMap::getBRDFs() const
//{
//   std::vector<glm::vec3> brdfs;
//   for (const auto& object : Objects) {
//      if (object->isLambert()) brdfs.emplace_back( object->getDiffuseReflectionColor() );
//      else brdfs.emplace_back( 1.0f, 1.0f, 1.0f );
//   }
//   return brdfs;
//}