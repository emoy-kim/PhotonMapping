#pragma once

#include "kdtree_object.h"

// <file path, type, color, world matrix>
using object_t = std::tuple<std::string, ObjectGL::TYPE, glm::vec4, glm::mat4>;

class PhotonMapGL final
{
public:
   static constexpr int ThreadBlockNum = 128;
   static constexpr int SampleNum = 256;
   static constexpr int MaxDepth = 64;
   static constexpr int MaxGlobalPhotonNum = 1'048'576;

   struct Photon
   {
      alignas(16) glm::vec3 Power;
      alignas(16) glm::vec3 Position;
      alignas(8) glm::vec2 IncomingDirection;

      Photon() : Power(), Position(), IncomingDirection() {}
   };

   PhotonMapGL() = default;
   ~PhotonMapGL() = default;

   void setObjects(const std::vector<object_t>& objects);
   void prepareBuilding();
   [[nodiscard]] const std::vector<glm::mat4>& getWorldMatrices() const { return ToWorlds; }
   [[nodiscard]] const std::vector<std::shared_ptr<ObjectGL>>& getObjects() const { return Objects; }

private:
   std::vector<Photon> Photons;
   std::shared_ptr<KdtreeGL> GlobalPhotonTree;
   std::vector<std::shared_ptr<ObjectGL>> Objects;
   std::vector<glm::mat4> ToWorlds;
   std::vector<Rect> WorldBounds;

};

#if 0
class PhotonMap final
{
public:
   struct Photon
   {
      glm::vec3 Position;
      glm::vec2 IncomingDirection;
      glm::vec3 Power;
      int KDTreeSplittingPlane;

      Photon() : Position(), IncomingDirection(), Power(), KDTreeSplittingPlane( -1 ) {}
   };

   struct NearestPhotons
   {
      bool HeapBuilt;
      int MaxNum;
      int FoundNum;
      glm::vec3 Position;
      std::vector<float> Distances;
      std::vector<const Photon*> Indices;

      NearestPhotons() : HeapBuilt( false ), MaxNum( 0 ), FoundNum( 0 ), Position() {}
   };

   explicit PhotonMap(int max_photons);
   ~PhotonMap() = default;

   void store(const glm::vec3& power, const glm::vec3& position, const glm::vec3& direction);
   void scalePhotonPower(float scale) // Call this function after each light source is processed.
   {
      // scale(= 1 / #emitted Photons) the Power of all Photons once they have been emitted from the light source.
      for (int i = PreviousScale; i <= StoredPhotonNum; ++i) Photons[i].Power *= scale;
      PreviousScale = StoredPhotonNum + 1;
   }
   void balance(); // balance the kd-tree (before use!)
   void findNearestPhotons(NearestPhotons* nearest_photons, int index) const;
   void estimateIrradiance(
      glm::vec3& irradiance,
      const glm::vec3& surface_position,
      const glm::vec3& surface_normal,
      float max_distance,
      int photon_num_to_use
   ) const;
   [[nodiscard]] static glm::vec3 getPhotonDirection(const Photon* p)
   {
      return {
         std::sin( p->IncomingDirection[0] ) * std::cos( p->IncomingDirection[1] ),
         std::sin( p->IncomingDirection[0] ) * std::sin( p->IncomingDirection[1] ),
         std::cos( p->IncomingDirection[0] )
      };
   }

private:
   static void splitMedian(std::vector<Photon*>& photon, int start, int end, int median, int axis);
   void balanceSegment(
      std::vector<Photon*>& balanced_photons,
      std::vector<Photon*>& photons,
      int index,
      int start,
      int end
   );

   int MaxPhotons;
   int StoredPhotonNum;
   int HalfStoredPhotons;
   int PreviousScale;
   glm::vec3 MinBoundingBox;
   glm::vec3 MaxBoundingBox;
   std::vector<Photon> Photons;
};
#endif