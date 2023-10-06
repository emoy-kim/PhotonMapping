#include "photon_map.h"

PhotonMapGL::PhotonMapGL() :
   LightNum( 0 ), PhotonBuffer( 0 ), AreaLightBuffer( 0 ), WorldBoundsBuffer( 0 ), ObjectVerticesBuffer( 0 ),
   ObjectNormalsBuffer( 0 ), ObjectVertexSizeBuffer( 0 )
{
}

PhotonMapGL::~PhotonMapGL()
{
   if (PhotonBuffer != 0) glDeleteBuffers( 1, &PhotonBuffer );
   if (AreaLightBuffer != 0) glDeleteBuffers( 1, &AreaLightBuffer );
   if (WorldBoundsBuffer != 0) glDeleteBuffers( 1, &WorldBoundsBuffer );
   if (ObjectVerticesBuffer != 0) glDeleteBuffers( 1, &ObjectVerticesBuffer );
   if (ObjectNormalsBuffer != 0) glDeleteBuffers( 1, &ObjectNormalsBuffer );
   if (ObjectVertexSizeBuffer != 0) glDeleteBuffers( 1, &ObjectVertexSizeBuffer );
}

void PhotonMapGL::separateObjectFile(
   const std::string& file_path,
   const std::string& out_file_root_path,
   const std::string& separator
)
{
   std::ifstream file(file_path);

   assert( file.is_open() );

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
         obj.open( out_file_root_path + std::to_string( file_index ) + ".obj", std::ofstream::out );
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

void PhotonMapGL::separateMaterialFile(
   const std::string& file_path,
   const std::string& out_file_root_path,
   const std::string& separator
)
{
   std::ifstream file(file_path);

   assert( file.is_open() );

   std::ofstream mtl;
   int file_index = 0;
   while (!file.eof()) {
      std::string line;
      std::getline( file, line );

      const std::regex space_delimiter("[ ]");
      const std::sregex_token_iterator line_it(line.begin(), line.end(), space_delimiter, -1);
      const std::vector<std::string> parsed(line_it, std::sregex_token_iterator());
      if (parsed.empty()) continue;

      if (parsed[0] == separator) {
         if (mtl.is_open()) mtl.close();
         file_index++;
         mtl.open( out_file_root_path + std::to_string( file_index ) + ".mtl", std::ofstream::out );
      }
      mtl << line + "\n";
   }
}

void PhotonMapGL::setObjects(const std::vector<object_t>& objects)
{
   LightNum = 0;
   Objects.clear();
   ToWorlds.clear();
   WorldBounds.clear();
   LightIndices.clear();
   Objects.resize( objects.size() );
   ToWorlds.resize( objects.size() );
   WorldBounds.resize( objects.size() );
   for (size_t i = 0; i < objects.size(); ++i) {
      const auto object_type = std::get<2>( objects[i] );
      if (object_type == ObjectGL::TYPE::LIGHT) {
         LightNum++;
         LightIndices.emplace_back( i );
         Objects[i] = std::make_shared<LightGL>();
         Objects[i]->setObjectWithTransform(
            GL_TRIANGLES, object_type, std::get<3>( objects[i] ),
            std::get<0>( objects[i] ), std::get<1>( objects[i] )
         );
         ToWorlds[i] = glm::mat4(1.0f);
      }
      else {
         Objects[i] = std::make_shared<ObjectGL>();
         Objects[i]->setObject(
            GL_TRIANGLES, object_type,
            std::get<0>( objects[i] ), std::get<1>( objects[i] )
         );
         ToWorlds[i] = std::get<3>( objects[i] );
      }
      WorldBounds[i] = Objects[i]->getBoundingBox();
      WorldBounds[i].MinPoint = glm::vec3(ToWorlds[i] * glm::vec4(WorldBounds[i].MinPoint, 1.0f));
      WorldBounds[i].MaxPoint = glm::vec3(ToWorlds[i] * glm::vec4(WorldBounds[i].MaxPoint, 1.0f));
   }
}

void PhotonMapGL::prepareBuilding()
{
   assert( LightNum > 0 && PhotonBuffer == 0 && AreaLightBuffer == 0 && WorldBoundsBuffer == 0 );

   // Currently, consider the only one area light.
   const auto light = std::dynamic_pointer_cast<LightGL>(Objects[LightIndices[0]]);
   const auto& areas = light->getAreas();
   const auto& triangles = light->getTriangles();
   const glm::vec3 normal = light->getNormal();
   const glm::vec3 emission = light->getEmissionColor();
   std::vector<AreaLight> area_lights;
   for (size_t i = 0; i < triangles.size(); ++i) {
      area_lights.emplace_back( areas[i], emission, normal, triangles[i] );
   }

   glCreateBuffers( 1, &AreaLightBuffer );
   auto buffer_size = static_cast<int>(sizeof( AreaLight ));
   glNamedBufferStorage( AreaLightBuffer, buffer_size, nullptr, GL_DYNAMIC_STORAGE_BIT );
   glNamedBufferSubData( AreaLightBuffer, 0, buffer_size, area_lights.data() );

   glCreateBuffers( 1, &WorldBoundsBuffer );
   buffer_size = static_cast<int>(WorldBounds.size() * sizeof( Rect ));
   glNamedBufferStorage( WorldBoundsBuffer, buffer_size, nullptr, GL_DYNAMIC_STORAGE_BIT );
   glNamedBufferSubData( WorldBoundsBuffer, 0, buffer_size, WorldBounds.data() );


   glCreateBuffers( 1, &PhotonBuffer );
   buffer_size = static_cast<int>(MaxGlobalPhotonNum * sizeof( Photon ));
   glNamedBufferStorage( PhotonBuffer, buffer_size, nullptr, GL_DYNAMIC_STORAGE_BIT );
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