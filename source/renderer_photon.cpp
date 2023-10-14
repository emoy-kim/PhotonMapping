#include "renderer.h"

void RendererGL::setPhotonMapShaders() const
{
   const std::string shader_directory_path = std::string(CMAKE_SOURCE_DIR) + "/shaders";
   PhotonMapBuilder.BuildPhotonMap->setComputeShader(
      std::string(shader_directory_path + "/photon_map/build_photon_map.comp").c_str()
   );
}

void RendererGL::createPhotonMap()
{
   using u = BuildPhotonMapShaderGL::UNIFORM;

   PhotonMap->prepareBuilding();

   std::vector<uint> seed(1);
   std::seed_seq sequence{ std::chrono::system_clock::now().time_since_epoch().count() };
   sequence.generate( seed.begin(), seed.end() );
   const auto& to_worlds = PhotonMap->getWorldMatrices();
   const auto types = PhotonMap->getObjectMaterialTypes();
   const auto brdfs = PhotonMap->getBRDFs();
   glUseProgram( PhotonMapBuilder.BuildPhotonMap->getShaderProgram() );
   PhotonMapBuilder.BuildPhotonMap->uniform1ui( u::Seed, seed[0] );
   PhotonMapBuilder.BuildPhotonMap->uniform1i( u::MaxGlobalPhotonNum, PhotonMapGL::MaxGlobalPhotonNum );
   PhotonMapBuilder.BuildPhotonMap->uniform1i( u::MaxDepth, PhotonMapGL::MaxDepth );
   PhotonMapBuilder.BuildPhotonMap->uniform1i( u::ObjectNum, PhotonMap->getObjectNum() );
   PhotonMapBuilder.BuildPhotonMap->uniform3fv( u::ObjectBRDFs, static_cast<int>(brdfs.size()), brdfs.data() );
   PhotonMapBuilder.BuildPhotonMap->uniform1iv( u::ObjectMaterialTypes, static_cast<int>(types.size()), types.data() );
   PhotonMapBuilder.BuildPhotonMap->uniformMat4fv( u::WorldMatrices, static_cast<int>(to_worlds.size()), to_worlds.data() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, PhotonMap->getPhotonBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, PhotonMap->getAreaLightBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, PhotonMap->getWorldBoundsBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, PhotonMap->getObjectVerticesBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, PhotonMap->getObjectNormalsBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, PhotonMap->getObjectIndicesBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 6, PhotonMap->getObjectVertexSizeBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 7, PhotonMap->getObjectIndexSizeBuffer() );
   glDispatchCompute( PhotonMapGL::ThreadBlockNum, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
}