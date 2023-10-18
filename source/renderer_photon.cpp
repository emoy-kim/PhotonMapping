#include "renderer.h"

void RendererGL::setPhotonMapShaders() const
{
   const std::string shader_directory_path = std::string(CMAKE_SOURCE_DIR) + "/shaders";
   PhotonMapBuilder.BuildPhotonMap->setComputeShader(
      std::string(shader_directory_path + "/photon_map/build_photon_map.comp").c_str()
   );

   PhotonMapBuilder.VisualizePhotonMapShader->setComputeShader(
      std::string(shader_directory_path + "/photon_map/visualize_photon_map.comp").c_str()
   );
}

void RendererGL::createPhotonMap() const
{
   using u = BuildPhotonMapShaderGL::UNIFORM;

   std::vector<uint> seed(1);
   std::seed_seq sequence{ std::chrono::system_clock::now().time_since_epoch().count() };
   sequence.generate( seed.begin(), seed.end() );

   PhotonMap->setPhotonMap();
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

   buildKdtree( PhotonMap->getGlobalPhotonTree(), PhotonMap->getPhotonBuffer() );
}

void RendererGL::visualizePhotonMap() const
{
   using u = VisualizePhotonMapShaderGL::UNIFORM;

   const KdtreeGL* kdtree = PhotonMap->getGlobalPhotonTree();
   const auto& to_worlds = PhotonMap->getWorldMatrices();
   glUseProgram( PhotonMapBuilder.VisualizePhotonMapShader->getShaderProgram() );
   PhotonMapBuilder.VisualizePhotonMapShader->uniform1i( u::NodeIndex, kdtree->getRootNode() );
   PhotonMapBuilder.VisualizePhotonMapShader->uniform1i( u::Size, kdtree->getUniqueNum() );
   PhotonMapBuilder.VisualizePhotonMapShader->uniform1i( u::Dim, kdtree->getDimension() );
   PhotonMapBuilder.VisualizePhotonMapShader->uniform1i( u::ObjectNum, PhotonMap->getObjectNum() );
   PhotonMapBuilder.VisualizePhotonMapShader->uniformMat4fv( u::InverseViewMatrix, glm::inverse( MainCamera->getViewMatrix() ) );
   PhotonMapBuilder.VisualizePhotonMapShader->uniformMat4fv( u::WorldMatrices, static_cast<int>(to_worlds.size()), to_worlds.data() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, PhotonMap->getPhotonBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getRoot() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, kdtree->getCoordinates() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, PhotonMap->getWorldBoundsBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, PhotonMap->getObjectVerticesBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, PhotonMap->getObjectNormalsBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 6, PhotonMap->getObjectIndicesBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 7, PhotonMap->getObjectVertexSizeBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 8, PhotonMap->getObjectIndexSizeBuffer() );
   glBindImageTexture( 0, GlobalPhotonMap->getTextureID(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8 );
   glDispatchCompute( divideUp( FrameWidth, 32 ), divideUp( FrameHeight, 32 ), 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT );

   writeTexture( GlobalPhotonMap->getTextureID(), FrameWidth, FrameHeight, "global_photon_map" );
   PhotonMap->releasePhotonMap();
}