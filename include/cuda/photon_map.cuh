#pragma once

#ifdef USE_CUDA
#include <regex>
#include <random>
#include <fstream>
#include <filesystem>
#include "cuda/kdtree.cuh"
#include <curand_kernel.h>
#include <math_constants.h>
#include <FreeImage.h>

namespace cuda
{
   static constexpr int SampleNum = 30;
   static constexpr int MaxDepth = 64;
   static constexpr int GatheringDepth = 4;
   static constexpr int NeighborNum = 64;
   static constexpr int MaxGlobalPhotonNum = 1'048'576;
   static constexpr float OneOverPi = CUDART_2_OVER_PI_F * 0.5f;

   struct Mat
   {
      float4 c0, c1, c2, c3;

      __host__ __device__
      Mat() :
         c0( make_float4( 0.0f, 0.0f, 0.0f, 0.0f ) ), c1( make_float4( 0.0f, 0.0f, 0.0f, 0.0f ) ),
         c2( make_float4( 0.0f, 0.0f, 0.0f, 0.0f ) ), c3( make_float4( 0.0f, 0.0f, 0.0f, 0.0f ) ) {}
      __host__ __device__
      explicit Mat(float scalar) :
         c0( make_float4( scalar, 0.0f, 0.0f, 0.0f ) ), c1( make_float4( 0.0f, scalar, 0.0f, 0.0f ) ),
         c2( make_float4( 0.0f, 0.0f, scalar, 0.0f ) ), c3( make_float4( 0.0f, 0.0f, 0.0f, scalar ) ) {}
      __host__ __device__
      explicit Mat(const float4& v0, const float4& v1, const float4& v2, const float4& v3) :
         c0( v0 ), c1( v1 ), c2( v2 ), c3( v3 ) {}
   };

   inline __host__ __device__ float3 operator+(float3 a, float3 b)
   {
       return make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
   }

   inline __host__ __device__ void operator+=(float3& a, float3 b)
   {
       a.x += b.x;
       a.y += b.y;
       a.z += b.z;
   }

   inline __host__ __device__ void operator+=(float3& a, float b)
   {
       a.x += b;
       a.y += b;
       a.z += b;
   }

   inline __host__ __device__ float4 operator+(float4 a, float4 b)
   {
       return make_float4( a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w );
   }

   inline __host__ __device__ float3 operator-(const float3& a)
   {
       return make_float3( -a.x, -a.y, -a.z );
   }

   inline __host__ __device__ float3 operator-(float3 a, float3 b)
   {
       return make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
   }

   inline __host__ __device__ float4 operator-(float4 a, float4 b)
   {
       return make_float4( a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w );
   }

   inline __host__ __device__ float3 operator-(float b, float3 a)
   {
       return make_float3( b - a.x, b - a.y, b - a.z );
   }

   inline __host__ __device__ float3 operator/(float3 a, float b)
   {
       return make_float3( a.x / b, a.y / b, a.z / b );
   }

   inline __host__ __device__ void operator/=(float3& a, float b)
   {
       a.x /= b;
       a.y /= b;
       a.z /= b;
   }

   inline __host__ __device__ float3 operator*(float3 a, float b)
   {
       return make_float3( a.x * b, a.y * b, a.z * b );
   }

   inline __host__ __device__ float3 operator*(float b, float3 a)
   {
       return make_float3( b * a.x, b * a.y, b * a.z );
   }

   inline __host__ __device__ float3 operator*(float3 a, float3 b)
   {
       return make_float3( a.x * b.x, a.y * b.y, a.z * b.z );
   }

   inline __host__ __device__ void operator*=(float3& a, float3 b)
   {
       a.x *= b.x;
       a.y *= b.y;
       a.z *= b.z;
   }

   inline __host__ __device__ void operator*=(float3& a, float b)
   {
       a.x *= b;
       a.y *= b;
       a.z *= b;
   }

   inline __host__ __device__ float4 operator*(float4 a, float4 b)
   {
       return make_float4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
   }

   inline __host__ __device__ void operator*=(float4& a, float b)
   {
       a.x *= b;
       a.y *= b;
       a.z *= b;
       a.w *= b;
   }

   inline __host__ __device__ float3 cross(const float3& v1, const float3& v2)
   {
      return make_float3(
         v1.y * v2.z - v2.y * v1.z,
         v1.z * v2.x - v2.z * v1.x,
         v1.x * v2.y - v2.x * v1.y
      );
   }

   inline __host__ __device__ float dot(const float3& v1, const float3& v2)
   {
      return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
   }

   inline __host__ __device__ float dot(const float4& v1, const float4& v2)
   {
      return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
   }

   inline __host__ __device__ float length(const float3& v)
   {
      return sqrt( dot( v, v ) );
   }

   inline __host__ __device__ float3 normalize(const float3& v)
   {
      return v / length( v );
   }

   inline __host__ __device__ float3 reflect(const float3& i, const float3& n)
   {
      return i - n * dot( n, i ) * 2.0f;
   }

   inline __host__ __device__ float3 refract(const float3& i, const float3& n, float eta)
   {
      const float n_dot_i = dot( n, i );
      const float k = 1.0f - eta * eta * (1.0f - n_dot_i * n_dot_i);
      return k >= 0.0f ? eta * i - (eta * n_dot_i + sqrt( k )) * n : make_float3( 0.0f, 0.0f, 0.0f );
   }

   inline __host__ __device__ int divideUp(int a, int b)
   {
      return (a + b - 1) / b;
   }

   struct Photon
   {
      float3 Power;
      float3 Position;
      float3 IncomingDirection;

      Photon() : Power(), Position(), IncomingDirection() {}
   };

   struct AreaLight
   {
      float Area;
      float3 Emission;
      float3 Normal;
      float3 Vertex0;
      float3 Vertex1;
      float3 Vertex2;
      Mat ToWorld;

      AreaLight() :
         Area( 0.0f ), Emission(), Normal(), Vertex0(), Vertex1(), Vertex2(), ToWorld() {}
      AreaLight(
         float area,
         const float3& emission,
         const float3& normal,
         const float3& v0,
         const float3& v1,
         const float3& v2,
         const Mat& m
      ) :
         Area( area ), Emission( emission ), Normal( normal ), Vertex0( v0 ), Vertex1( v1 ), Vertex2( v2 ),
         ToWorld( m ) {}
   };

   struct Box
   {
      float3 MinPoint;
      float3 MaxPoint;

      __host__ __device__
      Box() : MinPoint(), MaxPoint() {}
      __host__ __device__
      Box(const float3& min, const float3& max) : MinPoint( min ), MaxPoint( max ) {}
   };

   enum MATERIAL_TYPE { LAMBERT = 0, MIRROR, GLASS };

   struct Material
   {
      int MaterialType;
      float SpecularExponent;
      float RefractiveIndex;
      float3 Ambient;
      float3 Diffuse;
      float3 Specular;
      float3 Emission;

      Material() :
         MaterialType( 0 ), SpecularExponent( 1.0f ), RefractiveIndex( 1.0f ), Ambient(), Diffuse(), Specular(),
         Emission() {}
   };

   struct IntersectionInfo
   {
      int ObjectIndex;
      float3 Position;
      float3 Normal;
      float3 ShadingNormal;

      __host__ __device__
      IntersectionInfo() :
         ObjectIndex( -1 ), Position( make_float3( 0.0f, 0.0f, 0.0f ) ), Normal( make_float3( 0.0f, 0.0f, 0.0f ) ),
         ShadingNormal( make_float3( 0.0f, 0.0f, 0.0f ) ) {}
   };

   class PhotonMap final
   {
   public:
      PhotonMap();
      ~PhotonMap();

      void setObjects(const std::vector<std::tuple<std::string, std::string, cuda::Mat>>& objects);
      void setLights(const std::vector<std::tuple<std::string, std::string, cuda::Mat>>& lights);
      void createPhotonMap();
      void visualizePhotonMap(int width, int height);
      void render(int width, int height);

   private:
      struct CUDADevice
      {
         int ID;
         float3* VertexPtr;
         float3* NormalPtr;
         int* IndexPtr;
         int* VertexSizesPtr;
         int* IndexSizesPtr;
         Box* WorldBoundsPtr;
         Mat* ToWorldsPtr;
         Material* MaterialsPtr;
         AreaLight* AreaLightsPtr;
         Photon* GlobalPhotonsPtr;

         CUDADevice() :
            ID( -1 ), VertexPtr( nullptr ), NormalPtr( nullptr ), IndexPtr( nullptr ), VertexSizesPtr( nullptr ),
            IndexSizesPtr( nullptr ), WorldBoundsPtr( nullptr ), ToWorldsPtr( nullptr ), MaterialsPtr( nullptr ),
            AreaLightsPtr( nullptr ), GlobalPhotonsPtr( nullptr ) {}
      };

      CUDADevice Device;
      int ObjectNum;
      Mat ViewMatrix;
      Mat InverseViewMatrix;
      std::shared_ptr<KdtreeCUDA> GlobalPhotonTree;
      std::vector<float3> Vertices;
      std::vector<float3> Normals;
      std::vector<int> Indices;
      std::vector<int> VertexSizes;
      std::vector<int> IndexSizes;
      std::vector<Box> WorldBounds;
      std::vector<Mat> ToWorlds;
      std::vector<Material> Materials;
      std::vector<AreaLight> AreaLights;

      [[nodiscard]] static bool isNumber(const std::string& n)
      {
         return !n.empty() && std::find_if_not( n.begin(), n.end(), [](auto c) { return std::isdigit( c ); } ) == n.end();
      }
      void initialize();
      [[nodiscard]] static Mat getViewMatrix(const float3& eye, const float3& center, const float3& up);
      static void findNormals(
         std::vector<float3>& normals,
         const std::vector<float3>& vertices,
         const std::vector<int>& vertex_indices
      );
      void readObjectFile(Box& box, const Mat& t, const std::string& file_path);
      [[nodiscard]] static Material getMaterial(const std::string& mtl_file_path);
   };
}
#endif