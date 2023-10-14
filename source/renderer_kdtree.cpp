#include "renderer.h"

void RendererGL::setKdtreeShaders() const
{
   const std::string shader_directory_path = std::string(CMAKE_SOURCE_DIR) + "/shaders";
   KdtreeBuilder.Initialize->setComputeShader( std::string(shader_directory_path + "/kdtree/initialize.comp").c_str() );

   KdtreeBuilder.InitializeReference->setComputeShader(
      std::string(shader_directory_path + "/kdtree/initialize_reference.comp").c_str()
   );

   KdtreeBuilder.CopyCoordinates->setComputeShader(
      std::string(shader_directory_path + "/kdtree/copy_coordinates.comp").c_str()
   );

   KdtreeBuilder.SortByBlock->setComputeShader(
      std::string(shader_directory_path + "/kdtree/sort_by_block.comp").c_str()
   );

   KdtreeBuilder.SortLastBlock->setComputeShader(
      std::string(shader_directory_path + "/kdtree/sort_last_block.comp").c_str()
   );

   KdtreeBuilder.GenerateSampleRanks->setComputeShader(
      std::string(shader_directory_path + "/kdtree/generate_sample_ranks.comp").c_str()
   );

   KdtreeBuilder.MergeRanksAndIndices->setComputeShader(
      std::string(shader_directory_path + "/kdtree/merge_ranks_and_indices.comp").c_str()
   );

   KdtreeBuilder.MergeReferences->setComputeShader(
      std::string(shader_directory_path + "/kdtree/merge_references.comp").c_str()
   );

   KdtreeBuilder.RemoveDuplicates->setComputeShader(
      std::string(shader_directory_path + "/kdtree/remove_duplicates.comp").c_str()
   );

   KdtreeBuilder.RemoveGaps->setComputeShader(
      std::string(shader_directory_path + "/kdtree/remove_gaps.comp").c_str()
   );

   KdtreeBuilder.Partition->setComputeShader( std::string(shader_directory_path + "/kdtree/partition.comp").c_str() );

   KdtreeBuilder.RemovePartitionGaps->setComputeShader(
      std::string(shader_directory_path + "/kdtree/remove_partition_gaps.comp").c_str()
   );

   KdtreeBuilder.SmallPartition->setComputeShader(
      std::string(shader_directory_path + "/kdtree/small_partition.comp").c_str()
   );

   KdtreeBuilder.CopyReference->setComputeShader(
      std::string(shader_directory_path + "/kdtree/copy_reference.comp").c_str()
   );

   KdtreeBuilder.PartitionFinal->setComputeShader(
      std::string(shader_directory_path + "/kdtree/partition_final.comp").c_str()
   );

   KdtreeBuilder.Verify->setComputeShader( std::string(shader_directory_path + "/kdtree/verify.comp").c_str() );

   KdtreeBuilder.SumNodeNum->setComputeShader(
      std::string(shader_directory_path + "/kdtree/sum_node_num.comp").c_str()
   );

   KdtreeBuilder.Search->setComputeShader( std::string(shader_directory_path + "/kdtree/search.comp").c_str() );

   KdtreeBuilder.CopyFoundPoints->setComputeShader(
      std::string(shader_directory_path + "/kdtree/copy_found_points.comp").c_str()
   );

   KdtreeBuilder.InitializeKNN->setComputeShader(
      std::string(shader_directory_path + "/kdtree/initialize_knn.comp").c_str()
   );

   KdtreeBuilder.FindNearestNeighbors->setComputeShader(
      std::string(shader_directory_path + "/kdtree/find_nearest_neighbors.comp").c_str()
   );

   KdtreeBuilder.CopyEncodedFoundPoints->setComputeShader(
      std::string(shader_directory_path + "/kdtree/copy_encoded_found_points.comp").c_str()
   );
}

void RendererGL::sortByAxis(KdtreeGL* kdtree, int axis) const
{
   const int size = kdtree->getSize();
   const int dim = kdtree->getDimension();
   const GLuint coordinates = kdtree->getCoordinates();
   glUseProgram( KdtreeBuilder.CopyCoordinates->getShaderProgram() );
   KdtreeBuilder.CopyCoordinates->uniform1i( CopyCoordinatesShaderGL::UNIFORM::Size, size );
   KdtreeBuilder.CopyCoordinates->uniform1i( CopyCoordinatesShaderGL::UNIFORM::Axis, axis );
   KdtreeBuilder.CopyCoordinates->uniform1i( CopyCoordinatesShaderGL::UNIFORM::Dim, dim );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getBuffer( axis ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getReference( axis ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, coordinates );
   glDispatchCompute( KdtreeGL::ThreadBlockNum, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   int stage_num = 0;
   GLuint in_reference, in_buffer;
   GLuint out_reference, out_buffer;
   for (int step = KdtreeGL::SharedSize; step < size; step <<= 1) stage_num++;
   if (stage_num & 1) {
      in_buffer = kdtree->getSortBuffer();
      in_reference = kdtree->getSortReference();
      out_buffer = kdtree->getBuffer( dim );
      out_reference = kdtree->getReference( dim );
   }
   else {
      in_buffer = kdtree->getBuffer( dim );
      in_reference = kdtree->getReference( dim );
      out_buffer = kdtree->getSortBuffer();
      out_reference = kdtree->getSortReference();
   }

   assert( size <= KdtreeGL::SampleStride * kdtree->getMaxSampleNum() );

   int block_num = size / KdtreeGL::SharedSize;
   if (block_num > 0) {
      glUseProgram( KdtreeBuilder.SortByBlock->getShaderProgram() );
      KdtreeBuilder.SortByBlock->uniform1i( SortByBlockShaderGL::UNIFORM::Size, size );
      KdtreeBuilder.SortByBlock->uniform1i( SortByBlockShaderGL::UNIFORM::Axis, axis );
      KdtreeBuilder.SortByBlock->uniform1i( SortByBlockShaderGL::UNIFORM::Dim, dim );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, in_reference );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, in_buffer );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, kdtree->getReference( axis ) );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, kdtree->getBuffer( axis ) );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, coordinates );
      glDispatchCompute( block_num, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
   }
   const int remained_size = size % KdtreeGL::SharedSize;
   if (remained_size > 0) {
      int buffer_index = 0;
      const int start_offset = size - remained_size;
      const std::array<GLuint, 2> buffers{ kdtree->getBuffer( axis ), in_buffer };
      const std::array<GLuint, 2> references{ kdtree->getReference( axis ), in_reference };
      glUseProgram( KdtreeBuilder.SortLastBlock->getShaderProgram() );
      KdtreeBuilder.SortLastBlock->uniform1i( SortLastBlockShaderGL::UNIFORM::StartOffset, start_offset );
      KdtreeBuilder.SortLastBlock->uniform1i( SortLastBlockShaderGL::UNIFORM::Size, remained_size );
      KdtreeBuilder.SortLastBlock->uniform1i( SortLastBlockShaderGL::UNIFORM::Axis, axis );
      KdtreeBuilder.SortLastBlock->uniform1i( SortLastBlockShaderGL::UNIFORM::Dim, dim );
      for (int sorted_size = 1; sorted_size < remained_size; sorted_size <<= 1) {
         KdtreeBuilder.SortLastBlock->uniform1i( SortLastBlockShaderGL::UNIFORM::SortedSize, sorted_size );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, references[buffer_index ^ 1] );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, buffers[buffer_index ^ 1] );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, references[buffer_index] );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, buffers[buffer_index] );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, coordinates );
         glDispatchCompute( divideUp( remained_size, KdtreeGL::ThreadNum ), 1, 1 );
         glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
         buffer_index ^= 1;
      }
      if (buffer_index == 0) {
         glCopyNamedBufferSubData(
            buffers[0], buffers[1],
            static_cast<int>(sizeof( float ) * start_offset),
            static_cast<int>(sizeof( float ) * start_offset),
            static_cast<int>(sizeof( float ) * remained_size)
         );
         glCopyNamedBufferSubData(
            references[0], references[1],
            static_cast<int>(sizeof( int ) * start_offset),
            static_cast<int>(sizeof( int ) * start_offset),
            static_cast<int>(sizeof( int ) * remained_size)
         );
      }
   }

   for (int sorted_size = KdtreeGL::SharedSize; sorted_size < size; sorted_size <<= 1) {
      constexpr int thread_num = KdtreeGL::SampleStride * 2;
      const int remained_threads = size % (sorted_size * 2);
      const int total_thread_num = remained_threads > sorted_size ?
         (size - remained_threads + sorted_size * 2) / thread_num : (size - remained_threads) / thread_num;
      block_num = divideUp( total_thread_num, thread_num );
      glUseProgram( KdtreeBuilder.GenerateSampleRanks->getShaderProgram() );
      KdtreeBuilder.GenerateSampleRanks->uniform1i( GenerateSampleRanksShaderGL::SortedSize, sorted_size );
      KdtreeBuilder.GenerateSampleRanks->uniform1i( GenerateSampleRanksShaderGL::Size, size );
      KdtreeBuilder.GenerateSampleRanks->uniform1i( GenerateSampleRanksShaderGL::Axis, axis );
      KdtreeBuilder.GenerateSampleRanks->uniform1i( GenerateSampleRanksShaderGL::Dim, dim );
      KdtreeBuilder.GenerateSampleRanks->uniform1i( GenerateSampleRanksShaderGL::TotalThreadNum, total_thread_num );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getLeftRanks() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getRightRanks() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, in_reference );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, in_buffer );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, coordinates );
      glDispatchCompute( block_num, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

      glUseProgram( KdtreeBuilder.MergeRanksAndIndices->getShaderProgram() );
      KdtreeBuilder.MergeRanksAndIndices->uniform1i( MergeRanksAndIndicesShaderGL::UNIFORM::SortedSize, sorted_size );
      KdtreeBuilder.MergeRanksAndIndices->uniform1i( MergeRanksAndIndicesShaderGL::UNIFORM::Size, size );
      KdtreeBuilder.MergeRanksAndIndices->uniform1i( MergeRanksAndIndicesShaderGL::UNIFORM::TotalThreadNum, total_thread_num );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getLeftLimits() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getLeftRanks() );
      glDispatchCompute( block_num, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getRightLimits() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getRightRanks() );
      glDispatchCompute( block_num, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

      const int merge_pairs = remained_threads > sorted_size ?
         divideUp( size, KdtreeGL::SampleStride ) : (size - remained_threads) / KdtreeGL::SampleStride;
      glUseProgram( KdtreeBuilder.MergeReferences->getShaderProgram() );
      KdtreeBuilder.MergeReferences->uniform1i( MergeReferencesShaderGL::UNIFORM::SortedSize, sorted_size );
      KdtreeBuilder.MergeReferences->uniform1i( MergeReferencesShaderGL::UNIFORM::Size, size );
      KdtreeBuilder.MergeReferences->uniform1i( MergeReferencesShaderGL::UNIFORM::Axis, axis );
      KdtreeBuilder.MergeReferences->uniform1i( MergeReferencesShaderGL::UNIFORM::Dim, dim );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, out_reference );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, out_buffer );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, in_reference );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, in_buffer );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, coordinates );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, kdtree->getLeftLimits() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 6, kdtree->getRightLimits() );
      glDispatchCompute( merge_pairs, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

      if (remained_threads <= sorted_size) {
         glCopyNamedBufferSubData(
            in_reference, out_reference,
            static_cast<int>(sizeof( int ) * (size - remained_threads)),
            static_cast<int>(sizeof( int ) * (size - remained_threads)),
            static_cast<int>(sizeof( int ) * remained_threads)
         );
         glCopyNamedBufferSubData(
            in_buffer, out_buffer,
            static_cast<int>(sizeof( float ) * (size - remained_threads)),
            static_cast<int>(sizeof( float ) * (size - remained_threads)),
            static_cast<int>(sizeof( float ) * remained_threads)
         );
      }

      std::swap( in_reference, out_reference );
      std::swap( in_buffer, out_buffer );
   }
}

void RendererGL::removeDuplicates(KdtreeGL* kdtree, int axis) const
{
   constexpr int total_thread_num = KdtreeGL::ThreadBlockNum * KdtreeGL::ThreadNum;

   assert( total_thread_num > KdtreeGL::SharedSize / 2  );

   const int size = kdtree->getSize();
   const int dim = kdtree->getDimension();
   const int source_index = dim;
   const int target_index = axis;
   constexpr int block_num = total_thread_num * 2 / KdtreeGL::SharedSize;
   constexpr int segment = total_thread_num / KdtreeGL::WarpSize;
   const int size_per_warp = divideUp( size, segment );
   const GLuint coordinates = kdtree->getCoordinates();
   const GLuint num_after_removal = KdtreeGL::addBuffer<int>( 1 );
   const GLuint unique_num_in_warp = KdtreeGL::addBuffer<int>( segment );
   glUseProgram( KdtreeBuilder.RemoveDuplicates->getShaderProgram() );
   KdtreeBuilder.RemoveDuplicates->uniform1i( RemoveDuplicatesShaderGL::UNIFORM::SizePerWarp, size_per_warp );
   KdtreeBuilder.RemoveDuplicates->uniform1i( RemoveDuplicatesShaderGL::UNIFORM::Size, size );
   KdtreeBuilder.RemoveDuplicates->uniform1i( RemoveDuplicatesShaderGL::UNIFORM::Axis, axis );
   KdtreeBuilder.RemoveDuplicates->uniform1i( RemoveDuplicatesShaderGL::UNIFORM::Dim, dim );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, unique_num_in_warp );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getSortReference() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, kdtree->getSortBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, kdtree->getReference( source_index ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, kdtree->getBuffer( source_index ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, coordinates );
   glDispatchCompute( block_num, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   glUseProgram( KdtreeBuilder.RemoveGaps->getShaderProgram() );
   KdtreeBuilder.RemoveGaps->uniform1i( RemoveGapsShaderGL::UNIFORM::SizePerWarp, size_per_warp );
   KdtreeBuilder.RemoveGaps->uniform1i( RemoveGapsShaderGL::UNIFORM::Size, size );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getReference( target_index ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getBuffer( target_index ) );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, kdtree->getSortReference() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, kdtree->getSortBuffer() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, unique_num_in_warp );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, num_after_removal );
   glDispatchCompute( block_num, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   int num = 0;
   glGetNamedBufferSubData( num_after_removal, 0, sizeof( int ), &num );
   kdtree->setUniqueNum( num );

   KdtreeGL::releaseBuffer( num_after_removal );
   KdtreeGL::releaseBuffer( unique_num_in_warp );
}

void RendererGL::sort(KdtreeGL* kdtree) const
{
   kdtree->prepareSorting();
   KdtreeBuilder.InitializeReference->uniform1i( InitializeReferenceShaderGL::Size, kdtree->getSize() );
   for (int axis = 0; axis < kdtree->getDimension(); ++axis) {
      glUseProgram( KdtreeBuilder.InitializeReference->getShaderProgram() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getReference( axis ) );
      glDispatchCompute( KdtreeGL::ThreadBlockNum, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

      sortByAxis( kdtree, axis );
      removeDuplicates( kdtree, axis );
   }
   kdtree->releaseSorting();
}

void RendererGL::partitionDimension(KdtreeGL* kdtree, int axis, int depth) const
{
   constexpr int total_thread_num = KdtreeGL::ThreadBlockNum * KdtreeGL::ThreadNum;

   assert( total_thread_num > KdtreeGL::SharedSize / 2  );

   constexpr int block_num = total_thread_num * 2 / KdtreeGL::SharedSize;
   constexpr int warp_num = total_thread_num / KdtreeGL::WarpSize;
   const auto max_controllable_depth_for_warp =
         static_cast<int>(std::floor( std::log2( static_cast<double>(warp_num) ) ));
   const int dim = kdtree->getDimension();
   const int size = kdtree->getUniqueNum();
   const GLuint coordinates = kdtree->getCoordinates();
   const GLuint mid_reference = kdtree->getMidReferences( depth & 1 );
   const GLuint last_mid_reference = depth == 0 ? 0 : kdtree->getMidReferences( (depth - 1) & 1 );
   if (depth < max_controllable_depth_for_warp) {
      for (int i = 1; i < dim; ++i) {
         int r = i + axis;
         r = r < dim ? r : r - dim;
         glUseProgram( KdtreeBuilder.Partition->getShaderProgram() );
         KdtreeBuilder.Partition->uniform1i( PartitionShaderGL::UNIFORM::Start, 0 );
         KdtreeBuilder.Partition->uniform1i( PartitionShaderGL::UNIFORM::End, size - 1 );
         KdtreeBuilder.Partition->uniform1i( PartitionShaderGL::UNIFORM::Axis, axis );
         KdtreeBuilder.Partition->uniform1i( PartitionShaderGL::UNIFORM::Dim, dim );
         KdtreeBuilder.Partition->uniform1i( PartitionShaderGL::UNIFORM::Depth, depth );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getRoot() );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getLeftChildNumInWarp() );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, kdtree->getRightChildNumInWarp() );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, kdtree->getReference( dim ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, kdtree->getReference( dim + 1 ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, mid_reference );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 6, last_mid_reference );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 7, kdtree->getReference( r ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 8, kdtree->getReference( axis ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 9, coordinates );
         glDispatchCompute( block_num, 1, 1 );
         glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

         glUseProgram( KdtreeBuilder.RemovePartitionGaps->getShaderProgram() );
         KdtreeBuilder.RemovePartitionGaps->uniform1i( RemovePartitionGapsShaderGL::UNIFORM::Start, 0 );
         KdtreeBuilder.RemovePartitionGaps->uniform1i( RemovePartitionGapsShaderGL::UNIFORM::End, size - 1 );
         KdtreeBuilder.RemovePartitionGaps->uniform1i( RemovePartitionGapsShaderGL::UNIFORM::Depth, depth );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getReference( r ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getReference( dim ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, kdtree->getReference( dim + 1 ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, kdtree->getLeftChildNumInWarp() );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, kdtree->getRightChildNumInWarp() );
         glDispatchCompute( block_num, 1, 1 );
         glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
      }
   }
   else {
      for (int i = 1; i < dim; ++i) {
         int r = i + axis;
         r = r < dim ? r : r - dim;
         glUseProgram( KdtreeBuilder.SmallPartition->getShaderProgram() );
         KdtreeBuilder.SmallPartition->uniform1i( SmallPartitionShaderGL::UNIFORM::Start, 0 );
         KdtreeBuilder.SmallPartition->uniform1i( SmallPartitionShaderGL::UNIFORM::End, size - 1 );
         KdtreeBuilder.SmallPartition->uniform1i( SmallPartitionShaderGL::UNIFORM::Axis, axis );
         KdtreeBuilder.SmallPartition->uniform1i( SmallPartitionShaderGL::UNIFORM::Dim, dim );
         KdtreeBuilder.SmallPartition->uniform1i( SmallPartitionShaderGL::UNIFORM::Depth, depth );
         KdtreeBuilder.SmallPartition->uniform1i( SmallPartitionShaderGL::UNIFORM::MaxControllableDepthForWarp, max_controllable_depth_for_warp );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getRoot() );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getReference( dim ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, mid_reference );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, last_mid_reference );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, kdtree->getReference( r ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 5, kdtree->getReference( axis ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 6, coordinates );
         glDispatchCompute( block_num, 1, 1 );
         glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

         glUseProgram( KdtreeBuilder.CopyReference->getShaderProgram() );
         KdtreeBuilder.CopyReference->uniform1i( CopyReferenceShaderGL::UNIFORM::Size, size );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getReference( r ) );
         glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getReference( dim ) );
         glDispatchCompute( block_num, 1, 1 );
         glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
      }
   }

   if (depth == 0) {
      int root_node = 0;
      glGetNamedBufferSubData( kdtree->getMidReferences( 0 ), 0, sizeof( int ), &root_node );
      kdtree->setRootNode( root_node );
   }
}

void RendererGL::build(KdtreeGL* kdtree) const
{
   kdtree->prepareBuilding();
   const int dim = kdtree->getDimension();
   const int size = kdtree->getUniqueNum();
   const auto depth = static_cast<int>(std::floor( std::log2( static_cast<double>(size) ) ));
   for (int i = 0; i < depth - 1; ++i) {
      partitionDimension( kdtree, i % dim, i );
   }

   constexpr int total_thread_num = KdtreeGL::ThreadBlockNum * KdtreeGL::ThreadNum;
   constexpr int block_num = total_thread_num * 2 / KdtreeGL::SharedSize;
   constexpr int warp_num = total_thread_num / KdtreeGL::WarpSize;
   const auto max_controllable_depth_for_warp =
         static_cast<int>(std::floor( std::log2( static_cast<double>(warp_num) ) ));
   const int loop_levels = std::max( (depth - 1) - max_controllable_depth_for_warp, 0 );
   const int axis = (depth - 1) % dim;
   const GLuint mid_reference = kdtree->getMidReferences( (depth - 1) & 1 );
   const GLuint last_mid_reference = kdtree->getMidReferences( (depth - 2) & 1 );
   for (int loop = 0; loop < (1 << loop_levels); ++loop) {
      int start = 0, end = size - 1;
      for (int i = 1; i <= loop_levels; ++i) {
         const int mid = start + (end - start) / 2;
         if (loop & (1 << (loop_levels - i))) start = mid + 1;
         else end = mid - 1;
      }

      glUseProgram( KdtreeBuilder.PartitionFinal->getShaderProgram() );
      KdtreeBuilder.PartitionFinal->uniform1i( PartitionFinalShaderGL::UNIFORM::Start, start );
      KdtreeBuilder.PartitionFinal->uniform1i( PartitionFinalShaderGL::UNIFORM::End, end );
      KdtreeBuilder.PartitionFinal->uniform1i( PartitionFinalShaderGL::UNIFORM::Depth, (depth - 1) - loop_levels );
      KdtreeBuilder.PartitionFinal->uniform1i( PartitionFinalShaderGL::UNIFORM::MidReferenceOffset, loop * warp_num );
      KdtreeBuilder.PartitionFinal->uniform1i( PartitionFinalShaderGL::UNIFORM::LastMidReferenceOffset, loop * warp_num / 2 );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getRoot() );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, mid_reference );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, last_mid_reference );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, kdtree->getReference( axis ) );
      glDispatchCompute( block_num, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
   }
   kdtree->releaseBuilding();
}

void RendererGL::verify(KdtreeGL* kdtree) const
{
   kdtree->prepareVerifying();
   GLuint child, next_child;
   const GLuint root = kdtree->getRoot();
   const GLuint node_sums = kdtree->getNodeSums();
   const auto log_size = static_cast<int>(std::floor( std::log2( static_cast<double>(kdtree->getUniqueNum()) ) ));
   glUseProgram( KdtreeBuilder.Verify->getShaderProgram() );
   for (int i = 0; i <= log_size; ++i) {
      const int needed_threads = 1 << i;
      const int block_num = std::clamp( needed_threads / KdtreeGL::ThreadNum, 1, KdtreeGL::ThreadBlockNum );
      child = kdtree->getMidReferences( i & 1 );
      next_child = kdtree->getMidReferences( (i + 1) & 1 );
      KdtreeBuilder.Verify->uniform1i( VerifyShaderGL::UNIFORM::Size, needed_threads );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, node_sums );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, next_child );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, child );
      glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, root );
      glDispatchCompute( block_num, 1, 1 );
      glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );
   }

   glUseProgram( KdtreeBuilder.SumNodeNum->getShaderProgram() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, node_sums );
   glDispatchCompute( 1, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   int node_num = 0;
   glGetNamedBufferSubData( node_sums, 0, sizeof( int ), &node_num );
   kdtree->setNodeNum( node_num );

   kdtree->releaseVerifying();
}

void RendererGL::buildKdtree(KdtreeGL* kdtree, GLuint photon_buffer) const
{
   kdtree->initialize();
   glUseProgram( KdtreeBuilder.Initialize->getShaderProgram() );
   KdtreeBuilder.Initialize->uniform1i( InitializeShaderGL::UNIFORM::Size, kdtree->getSize() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, kdtree->getRoot() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, kdtree->getCoordinates() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, photon_buffer );
   glDispatchCompute( KdtreeGL::ThreadBlockNum, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   sort( kdtree );
   build( kdtree );
   verify( kdtree );
}

#if 0
void RendererGL::search()
{
   Object->prepareSearching( { query } );
   const int block_num = divideUp( query_num, KdtreeGL::WarpSize );
   glUseProgram( KdtreeBuilder.Search->getShaderProgram() );
   KdtreeBuilder.Search->uniform1f( "SearchRadius", SearchRadius );
   KdtreeBuilder.Search->uniform1i( "NodeIndex", Object->getRootNode() );
   KdtreeBuilder.Search->uniform1i( "QueryNum", query_num );
   KdtreeBuilder.Search->uniform1i( "Size", Object->getUniqueNum() );
   KdtreeBuilder.Search->uniform1i( "Dim", Object->getDimension() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getSearchLists() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getSearchListLengths() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, Object->getRoot() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, Object->getCoordinates() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, Object->getQueries() );
   glDispatchCompute( block_num, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   glUseProgram( KdtreeBuilder.CopyFoundPoints->getShaderProgram() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, FoundPoints->getVBO() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getSearchLists() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, Object->getSearchListLengths() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, Object->getCoordinates() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 4, Object->getQueries() );
   glDispatchCompute( block_num, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   Object->releaseSearching();
}

void RendererGL::findNearestNeighbors()
{
   Object->prepareKNN( { query }, NeighborNum );
   const int block_num = divideUp( query_num, KdtreeGL::WarpSize );
   glUseProgram( KdtreeBuilder.InitializeKNN->getShaderProgram() );
   KdtreeBuilder.InitializeKNN->uniform1i( "QueryNum", query_num );
   KdtreeBuilder.InitializeKNN->uniform1i( "NeighborNum", NeighborNum );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getSearchLists() );
   glDispatchCompute( block_num, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   glUseProgram( KdtreeBuilder.FindNearestNeighbors->getShaderProgram() );
   KdtreeBuilder.FindNearestNeighbors->uniform1i( "NodeIndex", Object->getRootNode() );
   KdtreeBuilder.FindNearestNeighbors->uniform1i( "QueryNum", query_num );
   KdtreeBuilder.FindNearestNeighbors->uniform1i( "NeighborNum", NeighborNum );
   KdtreeBuilder.FindNearestNeighbors->uniform1i( "Size", Object->getUniqueNum() );
   KdtreeBuilder.FindNearestNeighbors->uniform1i( "Dim", Object->getDimension() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, Object->getSearchLists() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getRoot() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, Object->getCoordinates() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, Object->getQueries() );
   glDispatchCompute( block_num, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   glUseProgram( KdtreeBuilder.CopyEncodedFoundPoints->getShaderProgram() );
   KdtreeBuilder.CopyEncodedFoundPoints->uniform1i( "NeighborNum", NeighborNum );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 0, FoundPoints->getVBO() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 1, Object->getSearchLists() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 2, Object->getCoordinates() );
   glBindBufferBase( GL_SHADER_STORAGE_BUFFER, 3, Object->getQueries() );
   glDispatchCompute( block_num, 1, 1 );
   glMemoryBarrier( GL_SHADER_STORAGE_BARRIER_BIT );

   Object->releaseKNN();
}
#endif