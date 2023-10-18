#pragma once

#include "host/kdtree.h"

class PhotonMap final
{
public:
   static constexpr int MaxDepth = 64;
   static constexpr int MaxGlobalPhotonNum = 1'048'576;

   struct Photon
   {
      glm::vec3 Power;
      glm::vec3 Position;
      glm::vec3 IncomingDirection;

      Photon() : Power(), Position(), IncomingDirection() {}
      Photon(const glm::vec3& power, const glm::vec3& position, const glm::vec3& direction) :
         Power( power ), Position( position ), IncomingDirection( direction ) {}
   };

   struct AreaLight
   {
      float Area;
      glm::vec3 Emission;
      glm::vec3 Normal;
      glm::vec3 Vertex0;
      glm::vec3 Vertex1;
      glm::vec3 Vertex2;
      glm::mat4 ToWorld;

      AreaLight() : Area( 0.0f ), Emission(), Normal(), Vertex0(), Vertex1(), Vertex2(), ToWorld() {}
      AreaLight(
         float area,
         const glm::vec3& emission,
         const glm::vec3& normal,
         const glm::vec3& vertex0,
         const glm::vec3& vertex1,
         const glm::vec3& vertex2,
         const glm::mat4& to_world
      ) :
         Area( area ), Emission( emission ), Normal( normal ), Vertex0( vertex0 ), Vertex1( vertex1 ),
         Vertex2( vertex2 ), ToWorld( to_world ) {}
   };

   struct Rect
   {
      glm::vec3 MinPoint;
      glm::vec3 MaxPoint;

      Rect() : MinPoint(), MaxPoint() {}
      Rect(const glm::vec3& min_point, const glm::vec3& max_point) : MinPoint( min_point ), MaxPoint( max_point ) {}
   };

   enum class MATERIAL_TYPE { LAMBERT = 0, MIRROR, GLASS };

   struct Material
   {
      MATERIAL_TYPE MaterialType;
      float SpecularExponent;
      float RefractiveIndex;
      glm::vec3 Ambient;
      glm::vec3 Diffuse;
      glm::vec3 Specular;
      glm::vec3 Emission;

      Material() :
         MaterialType( MATERIAL_TYPE::LAMBERT ), SpecularExponent( 1.0f ), RefractiveIndex( 1.0f ), Ambient(),
         Diffuse(), Specular(), Emission() {}
   };

   PhotonMap() = default;
   ~PhotonMap() = default;

   void setObjects(const std::vector<std::tuple<std::string, std::string, glm::mat4>>& objects);
   void setLights(const std::vector<std::tuple<std::string, std::string, glm::mat4>>& lights);
   void createPhotonMap();

private:
   struct IntersectionInfo
   {
      int ObjectIndex;
      glm::vec3 Position;
      glm::vec3 Normal;
      glm::vec3 ShadingNormal;

      IntersectionInfo() : ObjectIndex( -1 ), Position(), Normal(), ShadingNormal() {}
   };

   std::shared_ptr<Kdtree<>> GlobalPhotonTree;
   std::vector<Photon> GlobalPhotons;
   std::vector<glm::vec3> Vertices;
   std::vector<glm::vec3> Normals;
   std::vector<GLuint> Indices;
   std::vector<int> VertexSizes;
   std::vector<int> IndexSizes;
   std::vector<glm::mat4> ToWorlds;
   std::vector<Rect> WorldBounds;
   std::vector<Material> Materials;
   std::vector<AreaLight> AreaLights;

   [[nodiscard]] static bool isNumber(const std::string& n)
   {
      return !n.empty() && std::find_if_not( n.begin(), n.end(), [](auto c) { return std::isdigit( c ); } ) == n.end();
   }
   void findNormals(
      std::vector<glm::vec3>& normals,
      const std::vector<glm::vec3>& vertices,
      const std::vector<GLuint>& vertex_indices
   );
   void readObjectFile(Rect& box, const std::string& file_path);
   [[nodiscard]] Material getMaterial(const std::string& mtl_file_path);
   [[nodiscard]] static glm::vec3 getRandomPointInUnitHemisphere(float& pdf);
   [[nodiscard]] static glm::vec3 getSamplePointAroundAxis(float& pdf, const glm::vec3& v);
   [[nodiscard]] glm::vec3 getSampleRayFromLight(glm::vec3& ray_origin, glm::vec3& ray_direction);
   [[nodiscard]] static bool intersectWithBox(
      const glm::vec3& ray_origin,
      const glm::vec3& ray_direction,
      const Rect& box
   );
   [[nodiscard]] static bool intersectWithTriangle(
      glm::vec3& tuv,
      const glm::vec3& ray_origin,
      const glm::vec3& ray_direction,
      const glm::vec3& p0,
      const glm::vec3& p1,
      const glm::vec3& p2
   );
   [[nodiscard]] bool findIntersection(
      IntersectionInfo& intersection,
      const glm::vec3& ray_origin,
      const glm::vec3& ray_direction
   ) const;
   [[nodiscard]] static float correctShadingNormal(
      const glm::vec3& wo,
      const glm::vec3& wi,
      const glm::vec3& normal,
      const glm::vec3& shading_normal
   );
   [[nodiscard]] glm::vec3 getNextSampleRay(
      glm::vec3& ray_origin,
      glm::vec3& ray_direction,
      const IntersectionInfo& intersection
   );
};