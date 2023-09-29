#include "photon_map.h"

void PhotonMapGL::setObjects(const std::vector<object_t>& objects)
{
   Objects.clear();
   ToWorlds.clear();
   WorldBounds.clear();
   Objects.resize( objects.size() );
   ToWorlds.resize( objects.size() );
   WorldBounds.resize( objects.size() );
   for (size_t i = 0; i < objects.size(); ++i) {
      Objects[i] = std::make_shared<ObjectGL>();
      Objects[i]->setObject( GL_TRIANGLES, std::get<0>( objects[i] ) );
      Objects[i]->setObjectType( std::get<1>( objects[i] ) );
      Objects[i]->setDiffuseReflectionColor( std::get<2>( objects[i] ) );
      ToWorlds[i] = std::get<3>( objects[i] );
      WorldBounds[i] = Objects[i]->getBoundingBox();
      WorldBounds[i].MinPoint = glm::vec3(ToWorlds[i] * glm::vec4(WorldBounds[i].MinPoint, 1.0f));
      WorldBounds[i].MaxPoint = glm::vec3(ToWorlds[i] * glm::vec4(WorldBounds[i].MaxPoint, 1.0f));
   }
}

void PhotonMapGL::prepareBuilding()
{

}

#if 0
PhotonMap::PhotonMap(int max_photons) :
   MaxPhotons( max_photons ), StoredPhotonNum( 0 ), HalfStoredPhotons( 0 ), PreviousScale( 1 ), MinBoundingBox( 1e-8f ),
   MaxBoundingBox( -1e-8f ), Photons( max_photons + 1 )
{
}

void PhotonMap::findNearestPhotons(NearestPhotons* nearest_photons, int index) const
{
   const Photon* p = &Photons[index];
   const glm::vec3 direction = nearest_photons->Position - p->Position;
   if (index < HalfStoredPhotons) {
      const float distance = direction[p->KDTreeSplittingPlane];
      if (distance > 0.0f) {
         findNearestPhotons( nearest_photons, 2 * index + 1 );
         if (distance * distance < nearest_photons->Distances[0]) findNearestPhotons( nearest_photons, 2 * index );
      }
      else {
         findNearestPhotons( nearest_photons, 2 * index );
         if (distance * distance < nearest_photons->Distances[0]) findNearestPhotons( nearest_photons, 2 * index + 1 );
      }
   }

   const float squared_distance = glm::dot( direction, direction );
   if (squared_distance < nearest_photons->Distances[0]) {
      if (nearest_photons->FoundNum < nearest_photons->MaxNum) {
         nearest_photons->FoundNum++;
         nearest_photons->Distances[nearest_photons->FoundNum] = squared_distance;
         nearest_photons->Indices[nearest_photons->FoundNum] = p;
      }
      else {
         if (!nearest_photons->HeapBuilt) {
            const int half_found = nearest_photons->FoundNum >> 1;
            for (int k = half_found; k >= 1; --k) {
               int parent = k;
               const Photon* photon = nearest_photons->Indices[k];
               const float distance = nearest_photons->Distances[k];
               while (parent <= half_found) {
                  int j = parent * 2;
                  if (j < nearest_photons->FoundNum && nearest_photons->Distances[j] < nearest_photons->Distances[j + 1]) j++;
                  if (distance >= nearest_photons->Distances[j]) break;

                  nearest_photons->Distances[parent] = nearest_photons->Distances[j];
                  nearest_photons->Indices[parent] = nearest_photons->Indices[j];
                  parent = j;
               }
               nearest_photons->Distances[parent] = distance;
               nearest_photons->Indices[parent] = photon;
            }
            nearest_photons->HeapBuilt = true;
         }

         // insert new photon -- delete largest element, insert new, and reorder the heap
         int parent = 1, j = 2;
         while (j <= nearest_photons->FoundNum) {
            if (j < nearest_photons->FoundNum && nearest_photons->Distances[j] < nearest_photons->Distances[j + 1]) j++;
            if (squared_distance > nearest_photons->Distances[j]) break;

            nearest_photons->Distances[parent] = nearest_photons->Distances[j];
            nearest_photons->Indices[parent] = nearest_photons->Indices[j];
            parent = j;
            j += j;
         }
         if (squared_distance < nearest_photons->Distances[parent]) {
            nearest_photons->Indices[parent] = p;
            nearest_photons->Distances[parent] = squared_distance;
         }
         nearest_photons->Distances[0] = nearest_photons->Distances[1];
      }
   }
}

void PhotonMap::estimateIrradiance(
   glm::vec3& irradiance,
   const glm::vec3& surface_position,
   const glm::vec3& surface_normal,
   float max_distance,
   int photon_num_to_use
) const
{
   NearestPhotons nearest_photons{};
   nearest_photons.MaxNum = photon_num_to_use;
   nearest_photons.Position = surface_position;
   nearest_photons.Distances.clear();
   nearest_photons.Distances.resize( photon_num_to_use + 1 );
   nearest_photons.Distances[0] = max_distance * max_distance;
   nearest_photons.Indices.clear();
   nearest_photons.Indices.resize( photon_num_to_use + 1 );

   findNearestPhotons( &nearest_photons, 1 );
   if (nearest_photons.FoundNum < 8) return;

   irradiance = glm::vec3(0.0);
   for (int i = 1; i <= nearest_photons.FoundNum; ++i) {
      const Photon* p = nearest_photons.Indices[i];
      // sum only when the scene does not have any thin surfaces (for speed)
      if (glm::dot( getPhotonDirection( p ), surface_normal ) < 0.0f) irradiance += p->Power;
   }
   irradiance *= glm::one_over_pi<float>() / nearest_photons.Distances[0];
}

// store puts a photon into the flat array that will form the final kd-tree. Call this function to store a photon.
void PhotonMap::store(const glm::vec3& power, const glm::vec3& position, const glm::vec3& direction)
{
   if (StoredPhotonNum >= MaxPhotons) return;

   StoredPhotonNum++;
   Photon* node = &Photons[StoredPhotonNum];
   for (int i = 0; i < 3; ++i) {
      node->Position[i] = position[i];
      if (node->Position[i] < MinBoundingBox[i]) MinBoundingBox[i] = node->Position[i];
      if (node->Position[i] > MaxBoundingBox[i]) MaxBoundingBox[i] = node->Position[i];
      node->Power[i] = power[i];
   }
   node->IncomingDirection[0] = static_cast<float>(std::acos( direction[2] ));
   node->IncomingDirection[1] = static_cast<float>(std::atan2( direction[1], direction[0] ));
}

// splitMedian splits the photon array into two separate pieces around the median, with all Photons below the median
// in the lower half and all Photons above the median in the upper half. The comparison criteria is the axis
// (indicated by the axis parameter) (inspired by routine in "Algorithms in C++" by Sedgewick)
void PhotonMap::splitMedian(std::vector<Photon*>& photon, int start, int end, int median, int axis)
{
   int left = start, right = end;
   while (right > left) {
      const float v = photon[right]->Position[axis];
      int i = left - 1, j = right;
      while (true) {
         while (photon[++i]->Position[axis] < v);
         while (photon[--j]->Position[axis] > v && j > left);
         if (i >= j) break;

         std::swap( photon[i], photon[j] );
      }

      std::swap( photon[i], photon[right] );
      if (i >= median) right = i - 1;
      if (i <= median) left = i + 1;
   }
}

// See "Realistic Image Synthesis using Photon Mapping" Chapter 6 for an explanation of this function
void PhotonMap::balanceSegment(
   std::vector<Photon*>& balanced_photons,
   std::vector<Photon*>& photons,
   int index,
   int start,
   int end
)
{
   int median = 1;
   while (4 * median <= end - start + 1) median += median;
   if (3 * median <= end - start + 1) median += median + start - 1;
   else median = end - median + 1;

   int axis = 2;
   if (MaxBoundingBox[0] - MinBoundingBox[0] > MaxBoundingBox[1] - MinBoundingBox[1] &&
       MaxBoundingBox[0] - MinBoundingBox[0] > MaxBoundingBox[2] - MinBoundingBox[2]) axis = 0;
   else if (MaxBoundingBox[1] - MinBoundingBox[1] > MaxBoundingBox[2] - MinBoundingBox[2]) axis = 1;

   splitMedian( photons, start, end, median, axis );
   balanced_photons[index] = photons[median];
   balanced_photons[index]->KDTreeSplittingPlane = axis;

   if (median > start) {
      if (start < median - 1) {
         const float tmp = MaxBoundingBox[axis];
         MaxBoundingBox[axis] = balanced_photons[index]->Position[axis];
         balanceSegment( balanced_photons, photons, 2 * index, start, median - 1 );
         MaxBoundingBox[axis] = tmp;
      }
      else balanced_photons[2 * index] = photons[start];
   }

   if (median < end) {
      if (median + 1 < end) {
         const float tmp = MinBoundingBox[axis];
         MinBoundingBox[axis] = balanced_photons[index]->Position[axis];
         balanceSegment( balanced_photons, photons, 2 * index + 1, median + 1, end );
         MinBoundingBox[axis] = tmp;
      }
      else balanced_photons[2 * index + 1] = photons[end];
   }
}

// balance creates a left-balanced kd-tree from the flat photon array.
// This function should be called before the photon map is used for rendering.
void PhotonMap::balance()
{
   if (StoredPhotonNum > 1) {
      std::vector<Photon*> balanced_photons(StoredPhotonNum + 1), photons(StoredPhotonNum + 1);
      for (int i = 0; i <= StoredPhotonNum; ++i) photons[i] = &Photons[i];
      balanceSegment( balanced_photons, photons, 1, 1, StoredPhotonNum );
      photons.clear();
      photons.shrink_to_fit();

      // reorganize balanced kd-tree (make a heap)
      int j = 1, foo = 1;
      Photon foo_photon = Photons[j];
      for (int i = 1; i <= StoredPhotonNum; ++i) {
         const Photon* target = balanced_photons[j];
         const auto distance = static_cast<int>(std::distance(
            Photons.begin(),
            std::find_if(
               Photons.begin(), Photons.end(),
               [&target](const Photon& photon) { return &photon == target; }
            )
         ));
         balanced_photons[j] = nullptr;
         if (distance != foo) Photons[j] = Photons[distance];
         else {
            Photons[j] = foo_photon;
            if (i < StoredPhotonNum) {
               for (; foo <= StoredPhotonNum; ++foo) {
                  if (balanced_photons[foo] != nullptr) break;
               }
               foo_photon = Photons[foo];
               j = foo;
            }
            continue;
         }
         j = distance;
      }
   }
   HalfStoredPhotons = StoredPhotonNum / 2 - 1;
}
#endif