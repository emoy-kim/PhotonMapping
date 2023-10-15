#include "kdtree.h"

KdtreeGL::KdtreeGL(int size) :
   Dim( 3 ), Size( size ), UniqueNum( 0 ), RootNode( -1 ), NodeNum( 0 ), Sort(), Root( 0 ), Coordinates( 0 ),
   LeftChildNumInWarp( 0 ), RightChildNumInWarp( 0 ), NodeSums( 0 ), MidReferences{ 0, 0 }, Reference( Dim + 2, 0 ),
   Buffer( Dim + 1, 0 )
{
}

void KdtreeGL::initialize()
{
   Coordinates = addBuffer<float>( Dim * (Size + 1) );

   float max_value[Dim];
   for (int i = 0; i < Dim; ++i) max_value[i] = std::numeric_limits<float>::max();
   glNamedBufferSubData(
      Coordinates,
      static_cast<int>(Dim * Size * sizeof( float )),
      static_cast<int>(Dim * sizeof( float )),
      max_value
   );

   Root = addBuffer<KdtreeNodeGL>( Size );
}

void KdtreeGL::prepareSorting()
{
   const int max_sample_num = Size / SampleStride + 1;
   Sort.MaxSampleNum = max_sample_num;
   Sort.LeftRanks = addBuffer<int>( max_sample_num );
   Sort.RightRanks = addBuffer<int>( max_sample_num );
   Sort.LeftLimits = addBuffer<int>( max_sample_num );
   Sort.RightLimits = addBuffer<int>( max_sample_num );
   Sort.Reference = addBuffer<int>( Size );
   Sort.Buffer = addBuffer<float>( Size );

   for (int i = 0; i <= Dim + 1; ++i) {
      Reference[i] = addBuffer<int>( Size );
   }
   for (int i = 0; i <= Dim; ++i) {
      Buffer[i] = addBuffer<float>( Size );
   }
}

void KdtreeGL::releaseSorting()
{
   releaseBuffer( Sort.LeftRanks );
   releaseBuffer( Sort.RightRanks );
   releaseBuffer( Sort.LeftLimits );
   releaseBuffer( Sort.RightLimits );
   releaseBuffer( Sort.Reference );
   releaseBuffer( Sort.Buffer );
   for (int i = 0; i <= Dim; ++i) {
      releaseBuffer( Buffer[i] );
   }
}

void KdtreeGL::prepareBuilding()
{
   constexpr int warp_num = ThreadBlockNum * ThreadNum / WarpSize;
   LeftChildNumInWarp = addBuffer<int>( warp_num );
   RightChildNumInWarp = addBuffer<int>( warp_num );
   MidReferences[0] = addBuffer<int>( UniqueNum );
   MidReferences[1] = addBuffer<int>( UniqueNum );
}

void KdtreeGL::releaseBuilding()
{
   releaseBuffer( LeftChildNumInWarp );
   releaseBuffer( RightChildNumInWarp );
   releaseBuffer( MidReferences[0] );
   releaseBuffer( MidReferences[1] );
   for (int i = 0; i <= Dim + 1; ++i) {
      releaseBuffer( Reference[i] );
   }
}

void KdtreeGL::prepareVerifying()
{
   MidReferences[0] = addBuffer<int>( UniqueNum * 2 );
   MidReferences[1] = addBuffer<int>( UniqueNum * 2 );
   NodeSums = addBuffer<int>( ThreadBlockNum );

   assert( RootNode >= 0 );

   constexpr int zero = 0;
   glClearNamedBufferData( NodeSums, GL_R32I, GL_RED_INTEGER, GL_INT, &zero );
   glClearNamedBufferSubData( MidReferences[0], GL_R32I, 0, sizeof( int ), GL_RED_INTEGER, GL_INT, &RootNode );
}

void KdtreeGL::releaseVerifying()
{
   releaseBuffer( MidReferences[0] );
   releaseBuffer( MidReferences[1] );
   releaseBuffer( NodeSums );
}

void KdtreeGL::prepareSearching(const std::vector<glm::vec3>& queries)
{
   const auto query_num = static_cast<int>(queries.size());
   Search.Lists = addBuffer<int>( UniqueNum * query_num );
   Search.ListLengths = addBuffer<int>( query_num );
   Search.Queries = addBuffer<float>( query_num * Dim );

   constexpr int zero = 0;
   glClearNamedBufferData( Search.ListLengths, GL_R32I, GL_RED_INTEGER, GL_INT, &zero );
   glNamedBufferSubData(
      Search.Queries, 0,
      static_cast<int>(query_num * Dim * sizeof( float )),
      glm::value_ptr( queries[0] )
   );
}

void KdtreeGL::releaseSearching()
{
   releaseBuffer( Search.Lists );
   releaseBuffer( Search.ListLengths );
   releaseBuffer( Search.Queries );
}

void KdtreeGL::prepareKNN(const std::vector<glm::vec3>& queries, int neighbor_num)
{
   const auto query_num = static_cast<int>(queries.size());
   Search.Lists = addBuffer<uint>( neighbor_num * query_num * 2 );
   Search.Queries = addBuffer<float>( query_num * Dim );
   glNamedBufferSubData(
      Search.Queries, 0,
      static_cast<int>(query_num * Dim * sizeof( float )),
      glm::value_ptr( queries[0] )
   );
}

void KdtreeGL::releaseKNN()
{
   releaseBuffer( Search.Lists );
   releaseBuffer( Search.Queries );
}