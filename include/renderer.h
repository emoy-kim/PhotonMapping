/*
 * Author: Jeesun Kim
 * E-mail: emoy.kim_AT_gmail.com
 *
 */

#pragma once

#include "base.h"
#include "canvas.h"
#include "photon_map.h"
#include "kdtree_shader.h"
#include "photon_map_shader.h"

class RendererGL final
{
public:
   RendererGL();
   ~RendererGL() = default;

   RendererGL(const RendererGL&) = delete;
   RendererGL(const RendererGL&&) = delete;
   RendererGL& operator=(const RendererGL&) = delete;
   RendererGL& operator=(const RendererGL&&) = delete;

   void play();

private:
   struct KdtreeBuild
   {
      std::unique_ptr<InitializeShaderGL> Initialize;
      std::unique_ptr<InitializeReferenceShaderGL> InitializeReference;
      std::unique_ptr<CopyCoordinatesShaderGL> CopyCoordinates;
      std::unique_ptr<SortByBlockShaderGL> SortByBlock;
      std::unique_ptr<SortLastBlockShaderGL> SortLastBlock;
      std::unique_ptr<GenerateSampleRanksShaderGL> GenerateSampleRanks;
      std::unique_ptr<MergeRanksAndIndicesShaderGL> MergeRanksAndIndices;
      std::unique_ptr<MergeReferencesShaderGL> MergeReferences;
      std::unique_ptr<RemoveDuplicatesShaderGL> RemoveDuplicates;
      std::unique_ptr<RemoveGapsShaderGL> RemoveGaps;
      std::unique_ptr<PartitionShaderGL> Partition;
      std::unique_ptr<RemovePartitionGapsShaderGL> RemovePartitionGaps;
      std::unique_ptr<SmallPartitionShaderGL> SmallPartition;
      std::unique_ptr<CopyReferenceShaderGL> CopyReference;
      std::unique_ptr<PartitionFinalShaderGL> PartitionFinal;
      std::unique_ptr<VerifyShaderGL> Verify;
      std::unique_ptr<SumNodeNumShaderGL> SumNodeNum;
      std::unique_ptr<SearchShaderGL> Search;
      std::unique_ptr<CopyFoundPointsShaderGL> CopyFoundPoints;
      std::unique_ptr<InitializeKNNShaderGL> InitializeKNN;
      std::unique_ptr<FindNearestNeighborsShaderGL> FindNearestNeighbors;
      std::unique_ptr<CopyEncodedFoundPointsShaderGL> CopyEncodedFoundPoints;

      KdtreeBuild() :
         Initialize( std::make_unique<InitializeShaderGL>() ),
         InitializeReference( std::make_unique<InitializeReferenceShaderGL>() ),
         CopyCoordinates( std::make_unique<CopyCoordinatesShaderGL>() ),
         SortByBlock( std::make_unique<SortByBlockShaderGL>() ),
         SortLastBlock( std::make_unique<SortLastBlockShaderGL>() ),
         GenerateSampleRanks( std::make_unique<GenerateSampleRanksShaderGL>() ),
         MergeRanksAndIndices( std::make_unique<MergeRanksAndIndicesShaderGL>() ),
         MergeReferences( std::make_unique<MergeReferencesShaderGL>() ),
         RemoveDuplicates( std::make_unique<RemoveDuplicatesShaderGL>() ),
         RemoveGaps( std::make_unique<RemoveGapsShaderGL>() ),
         Partition( std::make_unique<PartitionShaderGL>() ),
         RemovePartitionGaps( std::make_unique<RemovePartitionGapsShaderGL>() ),
         SmallPartition( std::make_unique<SmallPartitionShaderGL>() ),
         CopyReference( std::make_unique<CopyReferenceShaderGL>() ),
         PartitionFinal( std::make_unique<PartitionFinalShaderGL>() ),
         Verify( std::make_unique<VerifyShaderGL>() ),
         SumNodeNum( std::make_unique<SumNodeNumShaderGL>() ),
         Search( std::make_unique<SearchShaderGL>() ),
         CopyFoundPoints( std::make_unique<CopyFoundPointsShaderGL>() ),
         InitializeKNN( std::make_unique<InitializeKNNShaderGL>() ),
         FindNearestNeighbors( std::make_unique<FindNearestNeighborsShaderGL>() ),
         CopyEncodedFoundPoints( std::make_unique<CopyEncodedFoundPointsShaderGL>() )
         {}
   };

   struct PhotonMapBuild
   {
      std::unique_ptr<BuildPhotonMapShaderGL> BuildPhotonMap;

      PhotonMapBuild() :
         BuildPhotonMap( std::make_unique<BuildPhotonMapShaderGL>() )
      {}
   };

   inline static RendererGL* Renderer = nullptr;

   GLFWwindow* Window;
   bool Pause;
   bool NeedToUpdate;
   int FrameWidth;
   int FrameHeight;
   glm::ivec2 ClickedPoint;
   std::unique_ptr<CameraGL> MainCamera;
   std::unique_ptr<SceneShaderGL> SceneShader;
   std::unique_ptr<PhotonMapGL> PhotonMap;
   std::unique_ptr<CanvasGL> Canvas;
   KdtreeBuild KdtreeBuilder;
   PhotonMapBuild PhotonMapBuilder;

   [[nodiscard]] static constexpr int divideUp(int a, int b) { return (a + b - 1) / b; }

   void registerCallbacks() const;
   void initialize();
   void writeFrame() const;
   static void writeTexture(GLuint texture_id, int width, int height, const std::string& name = {});
   static void printOpenGLInformation();
   static void cleanup(GLFWwindow* window);
   static void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods);
   static void cursor(GLFWwindow* window, double xpos, double ypos);
   static void mouse(GLFWwindow* window, int button, int action, int mods);

   void setObjects();
   void setShaders() const;
   void drawScene() const;
   void render();

   // renderer_kdtree.cpp
   void setKdtreeShaders() const;
   void sortByAxis(int axis) const;
   void removeDuplicates(int axis) const;
   void sort() const;
   void partitionDimension(int axis, int depth) const;
   void build() const;
   void verify() const;
   void buildKdtree() const;
   //void search();
   //void findNearestNeighbors();

   // renderer_photon.cpp
   void setPhotonMapShaders() const;
   void createPhotonMap();
};