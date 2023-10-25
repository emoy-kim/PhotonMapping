#include "cuda/photon_map.cuh"

#ifdef USE_CUDA
namespace cuda
{
   __host__ __device__ __forceinline__
   Mat inverse(const Mat& m)
   {
      const float coef00 = m.c2.z * m.c3.w - m.c3.z * m.c2.w;
      const float coef02 = m.c1.z * m.c3.w - m.c3.z * m.c1.w;
      const float coef03 = m.c1.z * m.c2.w - m.c2.z * m.c1.w;

      const float coef04 = m.c2.y * m.c3.w - m.c3.y * m.c2.w;
      const float coef06 = m.c1.y * m.c3.w - m.c3.y * m.c1.w;
      const float coef07 = m.c1.y * m.c2.w - m.c2.y * m.c1.w;

      const float coef08 = m.c2.y * m.c3.z - m.c3.y * m.c2.z;
      const float coef10 = m.c1.y * m.c3.z - m.c3.y * m.c1.z;
      const float coef11 = m.c1.y * m.c2.z - m.c2.y * m.c1.z;

      const float coef12 = m.c2.x * m.c3.w - m.c3.x * m.c2.w;
      const float coef14 = m.c1.x * m.c3.w - m.c3.x * m.c1.w;
      const float coef15 = m.c1.x * m.c2.w - m.c2.x * m.c1.w;

      const float coef16 = m.c2.x * m.c3.z - m.c3.x * m.c2.z;
      const float coef18 = m.c1.x * m.c3.z - m.c3.x * m.c1.z;
      const float coef19 = m.c1.x * m.c2.z - m.c2.x * m.c1.z;

      const float coef20 = m.c2.x * m.c3.y - m.c3.x * m.c2.y;
      const float coef22 = m.c1.x * m.c3.y - m.c3.x * m.c1.y;
      const float coef23 = m.c1.x * m.c2.y - m.c2.x * m.c1.y;

      const float4 fac0 = make_float4( coef00, coef00, coef02, coef03 );
      const float4 fac1 = make_float4( coef04, coef04, coef06, coef07 );
      const float4 fac2 = make_float4( coef08, coef08, coef10, coef11 );
      const float4 fac3 = make_float4( coef12, coef12, coef14, coef15 );
      const float4 fac4 = make_float4( coef16, coef16, coef18, coef19 );
      const float4 fac5 = make_float4( coef20, coef20, coef22, coef23 );

      const float4 vec0 = make_float4( m.c1.x, m.c0.x, m.c0.x, m.c0.x );
      const float4 vec1 = make_float4( m.c1.y, m.c0.y, m.c0.y, m.c0.y );
      const float4 vec2 = make_float4( m.c1.z, m.c0.z, m.c0.z, m.c0.z );
      const float4 vec3 = make_float4( m.c1.w, m.c0.w, m.c0.w, m.c0.w );

      const float4 inv0 = vec1 * fac0 - vec2 * fac1 + vec3 * fac2;
      const float4 inv1 = vec0 * fac0 - vec2 * fac3 + vec3 * fac4;
      const float4 inv2 = vec0 * fac1 - vec1 * fac3 + vec3 * fac5;
      const float4 inv3 = vec0 * fac2 - vec1 * fac4 + vec2 * fac5;

      const float4 sign_a = make_float4( 1.0f, -1.0f, 1.0f, -1.0f );
      const float4 sign_b = make_float4( -1.0f, 1.0f, -1.0f, 1.0f );
      Mat inv(inv0 * sign_a, inv1 * sign_b, inv2 * sign_a, inv3 * sign_b);
      const float one_over_det = 1.0f / dot( m.c0, make_float4( inv.c0.x, inv.c1.x, inv.c2.x, inv.c3.x ) );
      inv.c0 *= one_over_det;
      inv.c1 *= one_over_det;
      inv.c2 *= one_over_det;
      inv.c3 *= one_over_det;
      return inv;
   }

   __host__ __device__ __forceinline__
   Mat transpose(const Mat& m)
   {
      Mat t;
      t.c0.x = m.c0.x;
      t.c0.y = m.c1.x;
      t.c0.z = m.c2.x;
      t.c0.w = m.c3.x;

      t.c1.x = m.c0.y;
      t.c1.y = m.c1.y;
      t.c1.z = m.c2.y;
      t.c1.w = m.c3.y;

      t.c2.x = m.c0.z;
      t.c2.y = m.c1.z;
      t.c2.z = m.c2.z;
      t.c2.w = m.c3.z;

      t.c3.x = m.c0.w;
      t.c3.y = m.c1.w;
      t.c3.z = m.c2.w;
      t.c3.w = m.c3.w;
      return t;
   }

   __host__ __device__ __forceinline__
   float3 transform(const Mat& m, const float3& v)
   {
      return make_float3(
         m.c0.x * v.x + m.c1.x * v.y + m.c2.x * v.z + m.c3.x,
         m.c0.y * v.x + m.c1.y * v.y + m.c2.y * v.z + m.c3.y,
         m.c0.z * v.x + m.c1.z * v.y + m.c2.z * v.z + m.c3.z
      );
   }

   __host__ __device__ __forceinline__
   Mat getVectorTransform(const Mat& m)
   {
      return transpose( inverse( m ) );
   }

   __host__ __device__ __forceinline__
   float3 transformVector(const Mat& m, const float3& v)
   {
      return make_float3(
         m.c0.x * v.x + m.c1.x * v.y + m.c2.x * v.z,
         m.c0.y * v.x + m.c1.y * v.y + m.c2.y * v.z,
         m.c0.z * v.x + m.c1.z * v.y + m.c2.z * v.z
      );
   }

   __device__ __forceinline__
   float getRandomValue(curandState* state, float a, float b)
   {
      float r = curand_uniform( state );
      r = r * (b - a) + a;
      return r;
   }

   // this hemisphere is towards the y-axis, and its lower plane is the xz-plane.
   __device__
   float3 getRandomPointInUnitHemisphere(float& pdf, curandState* state)
   {
      const float phi = 2.0f * CUDART_PI_F * getRandomValue( state, 0.0f, 1.0f ); // [0, 2pi]
      const float theta = acos( getRandomValue( state, -1.0f, 1.0f ) ) * 0.5f; // [0, pi/2]
      const float cos_theta = cos( theta );
      pdf *= cos_theta * CUDART_2_OVER_PI_F * 0.5f;
      return { cos( phi ) * sin( theta ), cos_theta, sin( phi ) * sin( theta ) };
   }

   __device__
   float3 getSamplePointAroundAxis(float& pdf, const float3& v, curandState* state)
   {
      const float3 u = abs( v.y ) < 0.9f ?
         normalize( cross( v, make_float3( 0.0f, 1.0f, 0.0f ) ) ) :
         normalize( cross( v, make_float3( 0.0f, 0.0f, 1.0f ) ) );
      const float3 n = normalize( cross( u, v ) );
      const float3 p = getRandomPointInUnitHemisphere( pdf, state );
      return make_float3(
         u.x * p.x + v.x * p.y + n.x * p.z,
         u.y * p.x + v.y * p.y + n.y * p.z,
         u.z * p.x + v.z * p.y + n.z * p.z
      );
   }

   __device__
   float3 getSampleRayFromLight(
      float3& ray_origin,
      float3& ray_direction,
      curandState* state,
      const AreaLight* lights
   )
   {
      // Currently, the number of lights is 2.
      const int light_index = getRandomValue( state, 0.0f, 1.0f ) > 0.5f ? 0 : 1;
      const float3 v0 = lights[light_index].Vertex0;
      const float3 v1 = lights[light_index].Vertex1;
      const float3 v2 = lights[light_index].Vertex2;
      const float a = getRandomValue( state, 0.0f, 1.0f );
      const float b = getRandomValue( state, 0.0f, 1.0f );
      ray_origin = (1.0f - a - b) * v0 + a * v1 + b * v2;
      ray_origin = transform( lights[light_index].ToWorld, ray_origin );

      const float3 normal = lights[light_index].Normal;
      float pdf = 1.0f / (lights[light_index].Area * 2.0f);
      ray_direction = getSamplePointAroundAxis( pdf, normal, state );
      const float3 power = lights[light_index].Emission / pdf * abs( dot( ray_direction, normal ) );

      const Mat n = getVectorTransform( lights[light_index].ToWorld );
      ray_direction = transformVector( n, ray_direction );
      return power;
   }

   __device__
   bool intersectWithBox(const float3& ray_origin, const float3& ray_direction, const Box& box)
   {
      float exit = CUDART_INF_F;
      float enter = -CUDART_INF_F;
      if (abs( ray_direction.x ) > 1e-5f) {
         const float t_min = (box.MinPoint.x - ray_origin.x) / ray_direction.x;
         const float t_max = (box.MaxPoint.x - ray_origin.x) / ray_direction.x;
         const float t_enter = min( t_min, t_max );
         const float t_exit = max( t_min, t_max );
         if (t_enter > enter) enter = t_enter;
         if (t_exit < exit) exit = t_exit;
         if (enter > exit || exit < 0.0f) return false;
      }
      else if (ray_origin.x < box.MinPoint.x || box.MaxPoint.x < ray_origin.x) return false;

      if (abs( ray_direction.y ) > 1e-5f) {
         const float t_min = (box.MinPoint.y - ray_origin.y) / ray_direction.y;
         const float t_max = (box.MaxPoint.y - ray_origin.y) / ray_direction.y;
         const float t_enter = min( t_min, t_max );
         const float t_exit = max( t_min, t_max );
         if (t_enter > enter) enter = t_enter;
         if (t_exit < exit) exit = t_exit;
         if (enter > exit || exit < 0.0f) return false;
      }
      else if (ray_origin.y < box.MinPoint.y || box.MaxPoint.y < ray_origin.y) return false;

      if (abs( ray_direction.z ) > 1e-5f) {
         const float t_min = (box.MinPoint.z - ray_origin.z) / ray_direction.z;
         const float t_max = (box.MaxPoint.z - ray_origin.z) / ray_direction.z;
         const float t_enter = min( t_min, t_max );
         const float t_exit = max( t_min, t_max );
         if (t_enter > enter) enter = t_enter;
         if (t_exit < exit) exit = t_exit;
         if (enter > exit || exit < 0.0f) return false;
      }
      else if (ray_origin.z < box.MinPoint.z || box.MaxPoint.z < ray_origin.z) return false;

      //distance = enter > 0.0f ? enter : exit;
      return true;
   }

   __device__
   bool intersectWithTriangle(
      float3& tuv,
      const float3& ray_origin,
      const float3& ray_direction,
      const float3& p0,
      const float3& p1,
      const float3& p2
   )
   {
      const float3 e1 = p1 - p0;
      const float3 e2 = p2 - p0;
      const float3 q = cross( ray_direction, e2 );
      const float det = dot( e1, q );
      if (abs( det ) < 1e-5f) return false;

      const float f = 1.0f / det;
      const float3 s = ray_origin - p0;
      const float u = f * dot( s, q );
      if (u < 0.0f) return false;

      const float3 r = cross( s, e1 );
      const float v = f * dot( ray_direction, r );
      if (v < 0.0f || u + v > 1.0f) return false;

      const float t = f * dot( e2, r );
      if (t <= 0.0f) return false;

      tuv = make_float3( t, u, v );
      return true;
   }

   __device__
   bool findIntersection(
      IntersectionInfo& intersection,
      const Box* world_bounds,
      const Mat* to_worlds,
      const float3* vertices,
      const float3* normals,
      const int* indices,
      const int* vertex_sizes,
      const int* index_sizes,
      const float3& ray_origin,
      const float3& ray_direction,
      int object_num
   )
   {
      bool intersect = false;
      float distance = CUDART_INF_F;
      int offset = 0, index_offset = 0;
      for (int i = 0; i < object_num; ++i) {
         if (intersectWithBox( ray_origin, ray_direction, world_bounds[i] )) {
            const Mat vector_transform = getVectorTransform( to_worlds[i] );
            for (int j = 0; j < index_sizes[i]; j += 3) {
               const int k0 = offset + indices[index_offset + j];
               const int k1 = offset + indices[index_offset + j + 1];
               const int k2 = offset + indices[index_offset + j + 2];
               const float3 p0 = transform( to_worlds[i], vertices[k0] );
               const float3 p1 = transform( to_worlds[i], vertices[k1] );
               const float3 p2 = transform( to_worlds[i], vertices[k2] );

               float3 tuv;
               if (intersectWithTriangle( tuv, ray_origin, ray_direction, p0, p1, p2 )) {
                  if (distance > tuv.x) {
                     const float3 n0 = normalize( transformVector( vector_transform, normals[k0] ) );
                     const float3 n1 = normalize( transformVector( vector_transform, normals[k1] ) );
                     const float3 n2 = normalize( transformVector( vector_transform, normals[k2] ) );
                     distance = tuv.x;
                     intersection.ObjectIndex = i;
                     intersection.Normal = (n0 + n1 + n2) / 3.0f;
                     intersection.Position = ray_origin + tuv.x * ray_direction;
                     intersection.ShadingNormal = (1.0f - tuv.y - tuv.z) * n0 + tuv.y * n1 + tuv.z * n2;
                     intersect = true;
                  }
               }
            }
         }
         offset += vertex_sizes[i];
         index_offset += index_sizes[i];
      }
      return intersect;
   }

   __device__
   float getSchlickApproximation(float cos_theta, float n1, float n2)
   {
      const float r0 = (n1 - n2) * (n1 - n2) / ((n1 + n2) * (n1 + n2));
      const float x = 1.0f - cos_theta;
      const float x2 = x * x;
      return r0 + (1.0f - r0) * x2 * x2 * x;
   }

   __device__
   float correctShadingNormal(
      const float3& wo,
      const float3& wi,
      const float3& normal,
      const float3& shading_normal
   )
   {
      const float wo_dot_n = dot( wo, normal );
      const float wi_dot_n = dot( wi, normal );
      const float wo_dot_sn = dot( wo, shading_normal );
      const float wi_dot_sn = dot( wi, shading_normal );
      if (wo_dot_n * wo_dot_sn <= 0.0f || wi_dot_n * wi_dot_sn <= 0.0f) return 0.0f;

      const float a = abs( wo_dot_n * wi_dot_sn );
      return a < 1e-5f ? 1.0f : abs( wo_dot_sn * wi_dot_n ) / a;
   }

   __device__
   float3 getSampleBRDF(
      float& pdf,
      float3& outgoing,
      const float3& ray_direction,
      curandState* state,
      const Material* materials,
      const IntersectionInfo& intersection,
      bool from_camera
   )
   {
      pdf = 1.0f;
      float3 brdf;
      const float3 incoming = -ray_direction;
      if (materials[intersection.ObjectIndex].MaterialType == MATERIAL_TYPE::LAMBERT) {
         outgoing = getSamplePointAroundAxis( pdf, intersection.ShadingNormal, state );
         if (dot( incoming, intersection.ShadingNormal ) < 0.0f ||
             dot( outgoing, intersection.ShadingNormal ) < 0.0f) {
            brdf = make_float3( 0.0f, 0.0f, 0.0f );
         }
         else brdf = materials[intersection.ObjectIndex].Diffuse * CUDART_2_OVER_PI_F * 0.5f;
      }
      else if (materials[intersection.ObjectIndex].MaterialType == MATERIAL_TYPE::MIRROR) {
         outgoing = reflect( -incoming, intersection.ShadingNormal );
         const float d = abs( dot( outgoing, intersection.ShadingNormal ) );
         brdf = d < 1e-5f ? make_float3( 0.0f, 0.0f, 0.0f ) : make_float3( 1.0f, 1.0f, 1.0f ) / d;
      }
      else if (materials[intersection.ObjectIndex].MaterialType == MATERIAL_TYPE::GLASS) {
         float3 n;
         float in_refractive_index, out_refractive_index;
         if (dot( incoming, intersection.ShadingNormal ) > 0.0f) {
            out_refractive_index = 1.0f;
            in_refractive_index = materials[intersection.ObjectIndex].RefractiveIndex;
            n = intersection.ShadingNormal;
         }
         else {
            in_refractive_index = 1.0f;
            out_refractive_index = materials[intersection.ObjectIndex].RefractiveIndex;
            n = -intersection.ShadingNormal;
         }

         const float fresnel = getSchlickApproximation( dot( incoming, n ), out_refractive_index, in_refractive_index );
         if (getRandomValue( state, 0.0f, 1.0f ) < fresnel) {
            outgoing = reflect( -incoming, n );
            const float d = abs( dot( outgoing, n ) );
            brdf = d < 1e-5f ? make_float3( 0.0f, 0.0f, 0.0f ) : make_float3( 1.0f, 1.0f, 1.0f ) / d;
         }
         else {
            outgoing = refract( -incoming, n, out_refractive_index / in_refractive_index );
            if (outgoing.x == 0.0f && outgoing.y == 0.0f && outgoing.z == 0.0f) {
               outgoing = reflect( -incoming, n );
               const float d = abs( dot( outgoing, n ) );
               brdf = d < 1e-5f ? make_float3( 0.0f, 0.0f, 0.0f ) : make_float3( 1.0f, 1.0f, 1.0f ) / d;
            }
            else {
               const float d = abs( dot( outgoing, n ) );
               const float f = from_camera ?
                  (out_refractive_index * out_refractive_index) / (in_refractive_index * in_refractive_index) : 1.0f;
               brdf = d < 1e-5f ? make_float3( 0.0f, 0.0f, 0.0f ) : make_float3( 1.0f, 1.0f, 1.0f ) * f / d;
            }
         }
      }
      outgoing = normalize( outgoing );
      return brdf;
   }

   __global__
   void cuCreatePhotonMap(
      Photon* photons,
      const AreaLight* lights,
      const Material* materials,
      const Box* world_bounds,
      const Mat* to_worlds,
      const float3* vertices,
      const float3* normals,
      const int* indices,
      const int* vertex_sizes,
      const int* index_sizes,
      int object_num,
      uint seed
   )
   {
      int generated_num = 0;
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      const int photons_to_generate = divideUp( MaxGlobalPhotonNum, step );
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) * photons_to_generate;

      curandState state;
      curand_init( seed, index, 0, &state );

      while (true) {
         float3 ray_origin, ray_direction;
         float3 power = getSampleRayFromLight( ray_origin, ray_direction, &state, lights );
         for (int i = 0; i < MaxDepth; ++i) {
            if (power.x < 0.0f || power.y < 0.0f || power.z < 0.0f) break;

            IntersectionInfo intersection;
            if (!findIntersection(
                  intersection, world_bounds, to_worlds, vertices, normals, indices, vertex_sizes, index_sizes,
                  ray_origin, ray_direction, object_num
               )) break;

            if (materials[intersection.ObjectIndex].MaterialType == MATERIAL_TYPE::LAMBERT) {
               photons[index + generated_num].Power = power;
               photons[index + generated_num].Position = intersection.Position;
               photons[index + generated_num].IncomingDirection = -ray_direction;
               generated_num++;
               if (generated_num == photons_to_generate || index + generated_num >= MaxGlobalPhotonNum) return;
            }

            if (i > 0) {
               const float russian_roulette = min( max( power.x, max( power.y, power.z ) ), 1.0f );
               if (getRandomValue( &state, 0.0f, 1.0f ) < russian_roulette) power /= russian_roulette;
               else break;
            }

            float pdf;
            float3 outgoing;
            power *= getSampleBRDF( pdf, outgoing, ray_direction, &state, materials, intersection, false );
            power *=
               correctShadingNormal( outgoing, -ray_direction, intersection.Normal, intersection.ShadingNormal ) / pdf;
            ray_origin = intersection.Position;
            ray_direction = outgoing;
         }
      }
   }

   __global__
   void cuPrepareKdtree(float* coordinates, const Photon* photons, int size)
   {
      const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto step = static_cast<int>(blockDim.x * gridDim.x);
      for (int i = index; i < size; i += step) {
         coordinates[3 * i] = photons[i].Position.x;
         coordinates[3 * i + 1] = photons[i].Position.y;
         coordinates[3 * i + 2] = photons[i].Position.z;
      }
   }

   __device__
   void findNearestNeighbor(
      int& found_index,
      float& found_distance,
      const KdtreeNode* root,
      const Photon* photons,
      const float3& query,
      int node_index,
      int size
   )
   {
      constexpr int dim = 3;
      int depth = 0;
      int prev = -1;
      int curr = node_index;
      found_index = -1;
      found_distance = CUDART_INF_F;
      while (curr >= 0) {
         const KdtreeNode* node = &root[curr];
         const int parent = node->ParentIndex;
         if (curr >= size) {
            prev = curr;
            curr = parent;
            continue;
         }

         const bool from_child = prev >= 0 && (prev == node->LeftChildIndex || prev == node->RightChildIndex);
         if (!from_child) {
            const float3 v = query - photons[node->Index].Position;
            const float squared_distance = dot( v, v );
            if (squared_distance <= found_distance) {
               found_index = curr;
               found_distance = squared_distance;
            }
         }

         float t;
         const int axis = depth % dim;
         if (axis == 0) t = query.x - photons[node->Index].Position.x;
         else if (axis == 1) t = query.y - photons[node->Index].Position.y;
         else t = query.z - photons[node->Index].Position.z;
         const bool right_priority = t > 0;
         const int far_child = right_priority ? node->LeftChildIndex : node->RightChildIndex;
         const int close_child = right_priority ? node->RightChildIndex : node->LeftChildIndex;

         int next = -1;
         if (prev >= 0 && prev == close_child) {
            if (far_child >= 0 && (t == 0 || t * t <= found_distance)) {
               next = far_child;
               depth++;
            }
            else {
               next = parent;
               depth--;
            }
         }
         else if (prev >= 0 && prev == far_child) {
            next = parent;
            depth--;
         }
         else if (prev < 0 || prev == parent) {
            if (close_child < 0 && far_child < 0) {
               next = parent;
               depth--;
            }
            else if (close_child < 0) {
               next = far_child;
               depth++;
            }
            else {
               next = close_child;
               depth++;
            }
         }

         prev = curr;
         curr = next;
      }
   }

   __device__
   float push(int found_index[], float found_distance[], int node_index, float squared_distance)
   {
      int n = node_index;
      float d = squared_distance;
      for (int i = 0; i < NeighborNum; ++i) {
         float max_distance = found_distance[i];
         float min_distance = d;
         int max_index = found_index[i];
         int min_index = n;
         if (max_distance < min_distance) {
            max_distance = d;
            min_distance = found_distance[i];
            max_index = n;
            min_index = found_index[i];
         }
         found_index[i] = min_index;
         found_distance[i] = min_distance;
         n = max_index;
         d = max_distance;
      }
      return found_distance[NeighborNum - 1];
   }

   __device__
   void findNearestNeighbors(
      int found_index[],
      float found_distance[],
      const KdtreeNode* root,
      const Photon* photons,
      const float3& query,
      int node_index,
      int size
   )
   {
      constexpr int dim = 3;
      int depth = 0;
      int prev = -1;
      int curr = node_index;
      float max_squared_distance = CUDART_INF_F;
      while (curr >= 0) {
         const KdtreeNode* node = &root[curr];
         const int parent = node->ParentIndex;
         if (curr >= size) {
            prev = curr;
            curr = parent;
            continue;
         }

         const bool from_child = prev >= 0 && (prev == node->LeftChildIndex || prev == node->RightChildIndex);
         if (!from_child) {
            const float3 v = query - photons[node->Index].Position;
            const float squared_distance = dot( v, v );
            if (squared_distance <= max_squared_distance) {
               max_squared_distance = push( found_index, found_distance, curr, squared_distance );
            }
         }

         float t;
         const int axis = depth % dim;
         if (axis == 0) t = query.x - photons[node->Index].Position.x;
         else if (axis == 1) t = query.y - photons[node->Index].Position.y;
         else t = query.z - photons[node->Index].Position.z;
         const bool right_priority = t > 0;
         const int far_child = right_priority ? node->LeftChildIndex : node->RightChildIndex;
         const int close_child = right_priority ? node->RightChildIndex : node->LeftChildIndex;

         int next = -1;
         if (prev >= 0 && prev == close_child) {
            if (far_child >= 0 && (t == 0 || t * t <= max_squared_distance)) {
               next = far_child;
               depth++;
            }
            else {
               next = parent;
               depth--;
            }
         }
         else if (prev >= 0 && prev == far_child) {
            next = parent;
            depth--;
         }
         else if (prev < 0 || prev == parent) {
            if (close_child < 0 && far_child < 0) {
               next = parent;
               depth--;
            }
            else if (close_child < 0) {
               next = far_child;
               depth++;
            }
            else {
               next = close_child;
               depth++;
            }
         }

         prev = curr;
         curr = next;
      }
   }

   __global__
   void cuVisualizePhotonMap(
      uint8_t* image_buffer,
      const Photon* photons,
      const KdtreeNode* root,
      const Box* world_bounds,
      const Mat* to_worlds,
      const float3* vertices,
      const float3* normals,
      const int* indices,
      const int* vertex_sizes,
      const int* index_sizes,
      Mat inverse_view,
      int root_node,
      int width,
      int height,
      int object_num,
      int size
   )
   {
      const auto x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
      if (x >= width || y >= height) return;

      const int k = (y * width + x) * 3;
      const auto w = static_cast<float>(width);
      const auto h = static_cast<float>(height);
      const float u = (2.0f * static_cast<float>(x) - w) / h;
      const float v = (2.0f * static_cast<float>(y) - h) / h;
      const float3 ray_origin = make_float3( inverse_view.c3.x, inverse_view.c3.y, inverse_view.c3.z );
      const float3 ray_direction = normalize( transform( inverse_view, make_float3( u, v, -1.0f ) ) - ray_origin );

      IntersectionInfo intersection;
      if (!findIntersection(
            intersection, world_bounds, to_worlds, vertices, normals, indices, vertex_sizes, index_sizes,
            ray_origin, ray_direction, object_num
         )) {
         image_buffer[k] = image_buffer[k + 1] = image_buffer[k + 2] = 0;
         return;
      }

      int index;
      float squared_distance;
      findNearestNeighbor( index, squared_distance, root, photons, intersection.Position, root_node, size );
      if (squared_distance < 0.1f) {
         float3 power = photons[index].Power;
         power.x = min( max( power.x * 255.0f, 0.0f ), 255.0f );
         power.y = min( max( power.y * 255.0f, 0.0f ), 255.0f );
         power.z = min( max( power.z * 255.0f, 0.0f ), 255.0f );
         image_buffer[k + 2] = static_cast<uint8_t>(power.x);
         image_buffer[k + 1] = static_cast<uint8_t>(power.y);
         image_buffer[k] = static_cast<uint8_t>(power.z);
      }
      else image_buffer[k] = image_buffer[k + 1] = image_buffer[k + 2] = 0;
   }

   __device__
   bool hitLight(float3& emission, const AreaLight* lights, const float3& ray_origin, const float3& ray_direction)
   {
      float3 tuv;
      float3 v0 = lights[0].Vertex0;
      float3 v1 = lights[0].Vertex1;
      float3 v2 = lights[0].Vertex2;
      float3 p0 = transform( lights[0].ToWorld, v0 );
      float3 p1 = transform( lights[0].ToWorld, v1 );
      float3 p2 = transform( lights[0].ToWorld, v2 );
      if (intersectWithTriangle( tuv, ray_origin, ray_direction, p0, p1, p2 )) {
         emission = lights[0].Emission;
         return true;
      }

      v0 = lights[1].Vertex0;
      v1 = lights[1].Vertex1;
      v2 = lights[1].Vertex2;
      p0 = transform( lights[1].ToWorld, v0 );
      p1 = transform( lights[1].ToWorld, v1 );
      p2 = transform( lights[1].ToWorld, v2 );
      if (intersectWithTriangle( tuv, ray_origin, ray_direction, p0, p1, p2 )) {
         emission = lights[1].Emission;
         return true;
      }
      return false;
   }

   __device__
   float3 computeRadianceWithPhotonMap(
      const IntersectionInfo& intersection,
      const KdtreeNode* root,
      const Photon* photons,
      const Material* materials,
      const float3& ray_direction,
      int root_node,
      int size
   )
   {
      float3 radiance = make_float3( 0.0f, 0.0f, 0.0f );
      if (materials[intersection.ObjectIndex].MaterialType != MATERIAL_TYPE::LAMBERT) return radiance;

      int indices[NeighborNum];
      float squared_distances[NeighborNum];
      for (int i = 0; i < NeighborNum; ++i) {
         indices[i] = -1;
         squared_distances[i] = CUDART_INF_F;
      }
      findNearestNeighbor( indices[0], squared_distances[0], root, photons, intersection.Position, root_node, size );

      float max_distance = 0.0f;
      for (int i = 0; i < NeighborNum; ++i) {
         if (indices[i] < 0) break;

         if (dot( photons[indices[i]].IncomingDirection, intersection.ShadingNormal ) > 0.0f) {
            radiance +=
               photons[indices[i]].Power * materials[intersection.ObjectIndex].Diffuse * CUDART_2_OVER_PI_F * 0.5f;
         }
         max_distance = squared_distances[i];
      }
      if (indices[0] >= 0) radiance /= static_cast<float>(size) * CUDART_PI_F * max_distance;
      return radiance;
   }

   __device__
   float3 computeDirectIllumination(
      const IntersectionInfo& intersection,
      const AreaLight* lights,
      const Material* materials,
      const Box* world_bounds,
      const Mat* to_worlds,
      const float3& ray_origin,
      const float3& ray_direction,
      const float3* vertices,
      const float3* normals,
      const int* indices,
      const int* vertex_sizes,
      const int* index_sizes,
      curandState* state,
      int object_num
   )
   {
      // Currently, the number of lights is 2.
      const int light_index = getRandomValue( state, 0.0f, 1.0f ) > 0.5f ? 0 : 1;
      const float3 v0 = lights[light_index].Vertex0;
      const float3 v1 = lights[light_index].Vertex1;
      const float3 v2 = lights[light_index].Vertex2;
      const float a = getRandomValue( state, 0.0f, 1.0f );
      const float b = getRandomValue( state, 0.0f, 1.0f );
      float3 light_position = (1.0f - a - b) * v0 + a * v1 + b * v2;
      light_position = transform( lights[light_index].ToWorld, light_position );

      const float3 normal = lights[light_index].Normal;
      float pdf = 1.0f / (lights[light_index].Area * 2.0f);
      const float3 incoming = normalize( light_position - intersection.Position );
      const float l = length( light_position - intersection.Position );
      pdf *= l * l / abs( dot( -incoming, normal ) );

      float3 shadow_ray_origin = intersection.Position;
      float3 shadow_ray_direction = incoming;

      float3 radiance = make_float3( 0.0f, 0.0f, 0.0f );
      IntersectionInfo shadow_intersection;
      if (!findIntersection(
            shadow_intersection, world_bounds, to_worlds, vertices, normals, indices, vertex_sizes, index_sizes,
            shadow_ray_origin, shadow_ray_direction, object_num
         )) {
         if (materials[intersection.ObjectIndex].MaterialType == MATERIAL_TYPE::LAMBERT) {
            if (dot( -ray_direction, intersection.ShadingNormal ) > 0.0f &&
                dot( incoming, intersection.ShadingNormal ) > 0.0f) {
               radiance = lights[light_index].Emission * abs( dot( incoming, intersection.ShadingNormal ) ) *
                  materials[intersection.ObjectIndex].Diffuse * CUDART_2_OVER_PI_F * 0.5f / pdf;
            }
         }
      }
      return radiance;
   }

   __device__
   float3 computeIndirectIllumination(
      const IntersectionInfo& intersection,
      const KdtreeNode* root,
      const Photon* photons,
      const AreaLight* lights,
      const Material* materials,
      const Box* world_bounds,
      const Mat* to_worlds,
      const float3& ray_origin,
      const float3& ray_direction,
      const float3* vertices,
      const float3* normals,
      const int* indices,
      const int* vertex_sizes,
      const int* index_sizes,
      curandState* state,
      int root_node,
      int object_num,
      int size
   )
   {
      int depth = 0;
      float3 curr_ray_direction = ray_direction;
      float3 power = make_float3( 1.0f, 1.0f, 1.0f );
      float3 radiance = make_float3( 0.0f, 0.0f, 0.0f );
      IntersectionInfo curr_info = intersection;
      while (depth < MaxDepth) {
         float pdf;
         float3 gathering_ray_direction;
         float3 gathering_ray_origin = curr_info.Position;
         power *= getSampleBRDF( pdf, gathering_ray_direction, curr_ray_direction, state, materials, curr_info, true );
         power *= abs( dot( gathering_ray_direction, curr_info.ShadingNormal ) );

         IntersectionInfo gathering_intersection;
         if (!findIntersection(
               gathering_intersection, world_bounds, to_worlds, vertices, normals, indices, vertex_sizes, index_sizes,
               gathering_ray_origin, gathering_ray_direction, object_num
            )) break;

         if (materials[gathering_intersection.ObjectIndex].MaterialType == MATERIAL_TYPE::LAMBERT) {
            radiance += power / pdf * computeRadianceWithPhotonMap(
               gathering_intersection, root, photons, materials, ray_direction, root_node, size
            );
            break;
         }
         else {
            curr_ray_direction = -gathering_ray_direction;
            curr_info = gathering_intersection;
            depth++;
         }
      }
      return radiance;
   }

   __device__
   float3 getRadiance(
      const Photon* photons,
      const AreaLight* lights,
      const Material* materials,
      const KdtreeNode* root,
      const Box* world_bounds,
      const Mat* to_worlds,
      const float3& ray_origin,
      const float3& ray_direction,
      const float3* vertices,
      const float3* normals,
      const int* indices,
      const int* vertex_sizes,
      const int* index_sizes,
      curandState* state,
      int root_node,
      int object_num,
      int max_depth,
      int size
   )
   {
      int depth = 0;
      float3 curr_ray_origin = ray_origin;
      float3 curr_ray_direction = ray_direction;
      while (depth < MaxDepth) {
         IntersectionInfo intersection;
         if (!findIntersection(
               intersection, world_bounds, to_worlds, vertices, normals, indices, vertex_sizes, index_sizes,
               curr_ray_origin, curr_ray_direction, object_num
            )) return make_float3( 0.0f, 0.0f, 0.0f );

         float3 emission;
         if (hitLight( emission, lights, curr_ray_origin, curr_ray_direction )) return emission;

         if (materials[intersection.ObjectIndex].MaterialType == MATERIAL_TYPE::LAMBERT) {
            if (depth >= GatheringDepth) {
               return computeRadianceWithPhotonMap(
                  intersection, root, photons, materials, curr_ray_direction, root_node, size
               );
            }
            else {
               const float3 direct = computeDirectIllumination(
                  intersection, lights, materials, world_bounds, to_worlds, curr_ray_origin, curr_ray_direction,
                  vertices, normals, indices, vertex_sizes, index_sizes, state, object_num
               );
               //const float3 caustics = computeCausticsWithPhotonMap();
               const float3 indirect = computeIndirectIllumination(
                  intersection, root, photons, lights, materials, world_bounds, to_worlds,
                  curr_ray_origin, curr_ray_direction, vertices, normals, indices, vertex_sizes, index_sizes,
                  state, root_node, object_num, size
               );
               return direct + indirect;
            }
         }
         else {
            if (depth >= 3) {

            }
            depth++;
            break;
         }

      }
      return make_float3( 0.0f, 0.0f, 0.0f );
   }

   __global__
   void cuRenderScene(
      uint8_t* image_buffer,
      const Photon* photons,
      const KdtreeNode* root,
      const AreaLight* lights,
      const Material* materials,
      const Box* world_bounds,
      const Mat* to_worlds,
      const float3* vertices,
      const float3* normals,
      const int* indices,
      const int* vertex_sizes,
      const int* index_sizes,
      Mat inverse_view,
      int root_node,
      int width,
      int height,
      int object_num,
      int size,
      uint seed
   )
   {
      const auto x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const auto y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
      if (x >= width || y >= height) return;

      curandState state;
      curand_init( seed, x * 1973 + y * 9277, 0, &state );

      const auto w = static_cast<float>(width);
      const auto h = static_cast<float>(height);
      const float3 ray_origin = make_float3( inverse_view.c3.x, inverse_view.c3.y, inverse_view.c3.z );
      float3 color = make_float3( 0.0f, 0.0f, 0.0f );
      for (int s = 0; s < SampleNum; ++s) {
         const float u = (2.0f * (static_cast<float>(x) + getRandomValue( &state, 0.0f, 1.0f )) - w) / h;
         const float v = (2.0f * (static_cast<float>(y) + getRandomValue( &state, 0.0f, 1.0f )) - h) / h;
         const float3 ray_direction = normalize( transform( inverse_view, make_float3( u, v, -1.0f ) ) - ray_origin );
         color += getRadiance(
            photons, lights, materials, root, world_bounds, to_worlds, ray_origin, ray_direction,
            vertices, normals, indices, vertex_sizes, index_sizes, &state, root_node, object_num, 0, size
         );
      }
      color /= static_cast<float>(SampleNum);

      const int k = (y * width + x) * 3;
      image_buffer[k + 2] = static_cast<uint8_t>(min( max( color.x * 255.0f, 0.0f ), 255.0f ));
      image_buffer[k + 1] = static_cast<uint8_t>(min( max( color.y * 255.0f, 0.0f ), 255.0f ));
      image_buffer[k] = static_cast<uint8_t>(min( max( color.z * 255.0f, 0.0f ), 255.0f ));
   }

   PhotonMap::PhotonMap() : Device(), ObjectNum( 0 )
   {
      ViewMatrix = getViewMatrix(
         make_float3( 0.0f, 250.0f, 750.0f ),
         make_float3( 0.0f, 250.0f, 0.0f ),
         make_float3( 0.0f, 1.0f, 0.0f )
      );
      InverseViewMatrix = inverse( ViewMatrix );
   }

   PhotonMap::~PhotonMap()
   {
      if (Device.VertexPtr != nullptr) cudaFree( Device.VertexPtr );
      if (Device.VertexPtr != nullptr) cudaFree( Device.VertexPtr );
      if (Device.NormalPtr != nullptr) cudaFree( Device.NormalPtr );
      if (Device.IndexPtr != nullptr) cudaFree( Device.IndexPtr );
      if (Device.VertexSizesPtr != nullptr) cudaFree( Device.VertexSizesPtr );
      if (Device.IndexSizesPtr != nullptr) cudaFree( Device.IndexSizesPtr );
      if (Device.WorldBoundsPtr != nullptr) cudaFree( Device.WorldBoundsPtr );
      if (Device.ToWorldsPtr != nullptr) cudaFree( Device.ToWorldsPtr );
      if (Device.MaterialsPtr != nullptr) cudaFree( Device.MaterialsPtr );
      if (Device.AreaLightsPtr != nullptr) cudaFree( Device.AreaLightsPtr );
   }

   void PhotonMap::initialize()
   {
      assert( Vertices.size() == Normals.size() );

      ObjectNum = static_cast<int>(Materials.size());

      int device_num = 0;
      CHECK_CUDA( cudaGetDeviceCount( &device_num ) );
      if( device_num <= 0 ) throw std::runtime_error( "cuda device not found\n" );

      Device.ID = 0;
      CHECK_CUDA( cudaSetDevice( Device.ID ) );

      auto buffer_size = sizeof( float3 ) * Vertices.size();
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.VertexPtr), buffer_size ) );
      CHECK_CUDA( cudaMemcpy( Device.VertexPtr, Vertices.data(), buffer_size, cudaMemcpyHostToDevice ) );

      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.NormalPtr), buffer_size ) );
      CHECK_CUDA( cudaMemcpy( Device.NormalPtr, Normals.data(), buffer_size, cudaMemcpyHostToDevice ) );

      buffer_size = sizeof( int ) * Indices.size();
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.IndexPtr), buffer_size ) );
      CHECK_CUDA( cudaMemcpy( Device.IndexPtr, Indices.data(), buffer_size, cudaMemcpyHostToDevice ) );

      buffer_size = sizeof( int ) * VertexSizes.size();
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.VertexSizesPtr), buffer_size ) );
      CHECK_CUDA( cudaMemcpy( Device.VertexSizesPtr, VertexSizes.data(), buffer_size, cudaMemcpyHostToDevice ) );

      buffer_size = sizeof( int ) * IndexSizes.size();
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.IndexSizesPtr), buffer_size ) );
      CHECK_CUDA( cudaMemcpy( Device.IndexSizesPtr, IndexSizes.data(), buffer_size, cudaMemcpyHostToDevice ) );

      buffer_size = sizeof( Box ) * WorldBounds.size();
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.WorldBoundsPtr), buffer_size ) );
      CHECK_CUDA( cudaMemcpy( Device.WorldBoundsPtr, WorldBounds.data(), buffer_size, cudaMemcpyHostToDevice ) );

      buffer_size = sizeof( Mat ) * ToWorlds.size();
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.ToWorldsPtr), buffer_size ) );
      CHECK_CUDA( cudaMemcpy( Device.ToWorldsPtr, ToWorlds.data(), buffer_size, cudaMemcpyHostToDevice ) );

      buffer_size = sizeof( Material ) * Materials.size();
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.MaterialsPtr), buffer_size ) );
      CHECK_CUDA( cudaMemcpy( Device.MaterialsPtr, Materials.data(), buffer_size, cudaMemcpyHostToDevice ) );

      buffer_size = sizeof( AreaLight ) * AreaLights.size();
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.AreaLightsPtr), buffer_size ) );
      CHECK_CUDA( cudaMemcpy( Device.AreaLightsPtr, AreaLights.data(), buffer_size, cudaMemcpyHostToDevice ) );

      buffer_size = sizeof( Photon ) * MaxGlobalPhotonNum;
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&Device.GlobalPhotonsPtr), buffer_size ) );
   }

   void PhotonMap::createPhotonMap()
   {
      initialize();

      std::cout << ">> Create Photon Map ...\n";
      constexpr int block_num = 128;
      constexpr int thread_num = 512;
      const auto object_num = static_cast<int>(Materials.size());

      std::vector<uint> seed(1);
      std::seed_seq sequence{ std::chrono::system_clock::now().time_since_epoch().count() };
      sequence.generate( seed.begin(), seed.end() );

      cuCreatePhotonMap<<<block_num, thread_num>>>(
         Device.GlobalPhotonsPtr,
         Device.AreaLightsPtr, Device.MaterialsPtr, Device.WorldBoundsPtr, Device.ToWorldsPtr,
         Device.VertexPtr, Device.NormalPtr, Device.IndexPtr, Device.VertexSizesPtr, Device.IndexSizesPtr,
         object_num, seed[0]
      );
      CHECK_KERNEL;
      CHECK_CUDA( cudaDeviceSynchronize() );
      std::cout << ">> Created Photon Map\n";

      std::cout << ">> Build Global Photon Map ...\n";
      GlobalPhotonTree = std::make_shared<KdtreeCUDA>( MaxGlobalPhotonNum, 3 );
      float* coordinates = GlobalPhotonTree->prepareDeviceCoordinatesPtr();
      cuPrepareKdtree<<<block_num, thread_num>>>( coordinates, Device.GlobalPhotonsPtr, MaxGlobalPhotonNum );
      CHECK_KERNEL;
      GlobalPhotonTree->create();
      std::cout << ">> Built Global Photon Map\n";
   }

   Mat PhotonMap::getViewMatrix(const float3& eye, const float3& center, const float3& up)
   {
      const float3 f = normalize( center - eye );
		const float3 s = normalize( cross( f, up ) );
		const float3 u = cross( s, f );

		Mat view(1.0f);
		view.c0.x = s.x;
		view.c1.x = s.y;
		view.c2.x = s.z;
		view.c0.y = u.x;
		view.c1.y = u.y;
		view.c2.y = u.z;
		view.c0.z = -f.x;
		view.c1.z = -f.y;
		view.c2.z = -f.z;
		view.c3.x = -dot( s, eye );
		view.c3.y = -dot( u, eye );
		view.c3.z = dot( f, eye );
		return view;
   }

   void PhotonMap::visualizePhotonMap(int width, int height)
   {
      std::cout << ">> Visualize Global Photon Map ...\n";
      uint8_t* image_buffer_ptr = nullptr;
      const size_t buffer_size = sizeof( uint8_t ) * width * height * 3;
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&image_buffer_ptr), buffer_size ) );

      constexpr dim3 block(32, 32);
      const dim3 grid(divideUp( width, static_cast<int>(block.x) ), divideUp( height, static_cast<int>(block.y) ));
      cuVisualizePhotonMap<<<grid, block>>>(
         image_buffer_ptr,
         Device.GlobalPhotonsPtr, GlobalPhotonTree->getRoot(),
         Device.WorldBoundsPtr, Device.ToWorldsPtr,
         Device.VertexPtr, Device.NormalPtr, Device.IndexPtr, Device.VertexSizesPtr, Device.IndexSizesPtr,
         InverseViewMatrix, GlobalPhotonTree->getRootNode(), width, height, ObjectNum, MaxGlobalPhotonNum
      );
      CHECK_KERNEL;
      CHECK_CUDA( cudaDeviceSynchronize() );

      auto* image_buffer = new uint8_t[width * height * 3];
      CHECK_CUDA( cudaMemcpy( image_buffer, image_buffer_ptr, buffer_size, cudaMemcpyDeviceToHost ) );
      cudaFree( image_buffer_ptr );

      FIBITMAP* image = FreeImage_ConvertFromRawBits(
         image_buffer, width, height, width * 3, 24,
         FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false
      );
      FreeImage_Save( FIF_PNG, image, "../global_photons.png" );
      FreeImage_Unload( image );
      delete [] image_buffer;
      std::cout << ">> Visualized Global Photon Map\n";
   }

   void PhotonMap::render(int width, int height)
   {
      std::cout << ">> Rendering ...\n";
      uint8_t* image_buffer_ptr = nullptr;
      const size_t buffer_size = sizeof( uint8_t ) * width * height * 3;
      CHECK_CUDA( cudaMalloc( reinterpret_cast<void**>(&image_buffer_ptr), buffer_size ) );

      std::vector<uint> seed(1);
      std::seed_seq sequence{ std::chrono::system_clock::now().time_since_epoch().count() };
      sequence.generate( seed.begin(), seed.end() );

      constexpr dim3 block(16, 16);
      const dim3 grid(divideUp( width, static_cast<int>(block.x) ), divideUp( height, static_cast<int>(block.y) ));
      cuRenderScene<<<grid, block>>>(
         image_buffer_ptr,
         Device.GlobalPhotonsPtr, GlobalPhotonTree->getRoot(),
         Device.AreaLightsPtr, Device.MaterialsPtr, Device.WorldBoundsPtr, Device.ToWorldsPtr,
         Device.VertexPtr, Device.NormalPtr, Device.IndexPtr, Device.VertexSizesPtr, Device.IndexSizesPtr,
         InverseViewMatrix, GlobalPhotonTree->getRootNode(), width, height, ObjectNum, MaxGlobalPhotonNum, seed[0]
      );
      CHECK_KERNEL;

      auto* image_buffer = new uint8_t[width * height * 3];
      CHECK_CUDA( cudaMemcpy( image_buffer, image_buffer_ptr, buffer_size, cudaMemcpyDeviceToHost ) );
      cudaFree( image_buffer_ptr );

      FIBITMAP* image = FreeImage_ConvertFromRawBits(
         image_buffer, width, height, width * 3, 24,
         FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false
      );
      FreeImage_Save( FIF_PNG, image, "../scene.png" );
      FreeImage_Unload( image );
      delete [] image_buffer;
      std::cout << ">> Rendered\n";
   }

   void PhotonMap::findNormals(
      std::vector<float3>& normals,
      const std::vector<float3>& vertices,
      const std::vector<int>& vertex_indices
   )
   {
      normals.resize( vertices.size() );
      const auto size = static_cast<int>(vertex_indices.size());
      for (int i = 0; i < size; i += 3) {
         const int n0 = vertex_indices[i];
         const int n1 = vertex_indices[i + 1];
         const int n2 = vertex_indices[i + 2];
         const float3 normal = cross( vertices[n1] - vertices[n0], vertices[n2] - vertices[n0] );
         normals[n0] += normal;
         normals[n1] += normal;
         normals[n2] += normal;
      }
      for (auto& n : normals) n = normalize( n );
   }

   void PhotonMap::readObjectFile(Box& box, const Mat& t, const std::string& file_path)
   {
      std::ifstream file(file_path);

      assert( file.is_open() );

      constexpr auto m = std::numeric_limits<float>::max();
      constexpr auto n = std::numeric_limits<float>::lowest();
      box.MinPoint = make_float3( m, m, m );
      box.MaxPoint = make_float3( n, n, n );

      bool found_normals = false;
      std::vector<float3> vertex_buffer, normal_buffer;
      std::vector<int> vertex_indices, normal_indices;
      while (!file.eof()) {
         std::string word;
         file >> word;

         if (word == "v") {
            float3 vertex;
            file >> vertex.x >> vertex.y >> vertex.z;
            vertex_buffer.emplace_back( vertex );

            const float3 v = transform( t, vertex );
            box.MinPoint.x = std::min( v.x, box.MinPoint.x );
            box.MinPoint.y = std::min( v.y, box.MinPoint.y );
            box.MinPoint.z = std::min( v.z, box.MinPoint.z );
            box.MaxPoint.x = std::max( v.x, box.MaxPoint.x );
            box.MaxPoint.y = std::max( v.y, box.MaxPoint.y );
            box.MaxPoint.z = std::max( v.z, box.MaxPoint.z );
         }
         else if (word == "vn") {
            float3 normal;
            file >> normal.x >> normal.y >> normal.z;
            normal_buffer.emplace_back( normal );
            found_normals = true;
         }
         else if (word == "f") {
            std::string face;
            const std::regex delimiter("[/]");
            for (int i = 0; i < 3; ++i) {
               file >> face;
               const std::sregex_token_iterator it(face.begin(), face.end(), delimiter, -1);
               const std::vector<std::string> vtn(it, std::sregex_token_iterator());
               vertex_indices.emplace_back( std::stoi( vtn[0] ) - 1 );
               if (found_normals && isNumber( vtn[2] )) {
                  normal_indices.emplace_back( std::stoi( vtn[2] ) - 1 );
                  found_normals = false;
               }
            }
         }
         else std::getline( file, word );
      }

      if (!found_normals) findNormals( normal_buffer, vertex_buffer, vertex_indices );

      VertexSizes.emplace_back( vertex_buffer.size() );
      IndexSizes.emplace_back( vertex_indices.size() );
      Vertices.insert(
         Vertices.end(),
         std::make_move_iterator( vertex_buffer.begin() ),
         std::make_move_iterator( vertex_buffer.end() )
      );
      Normals.insert(
         Normals.end(),
         std::make_move_iterator( normal_buffer.begin() ),
         std::make_move_iterator( normal_buffer.end() )
      );
      Indices.insert(
         Indices.end(),
         std::make_move_iterator( vertex_indices.begin() ),
         std::make_move_iterator( vertex_indices.end() )
      );
   }

   Material PhotonMap::getMaterial(const std::string& mtl_file_path)
   {
      std::ifstream file(mtl_file_path);

      assert( file.is_open() );

      Material material;
      while (!file.eof()) {
         std::string line;
         std::getline( file, line );

         const std::regex space_delimiter("[ ]");
         const std::sregex_token_iterator line_it(line.begin(), line.end(), space_delimiter, -1);
         const std::vector<std::string> parsed(line_it, std::sregex_token_iterator());
         if (parsed.empty()) continue;

         if (parsed[0] == "Ka") {
            material.Ambient.x = std::stof( parsed[1] );
            material.Ambient.y = std::stof( parsed[2] );
            material.Ambient.z = std::stof( parsed[3] );
         }
         else if (parsed[0] == "Kd") {
            material.Diffuse.x = std::stof( parsed[1] );
            material.Diffuse.y = std::stof( parsed[2] );
            material.Diffuse.z = std::stof( parsed[3] );
         }
         else if (parsed[0] == "Ks") {
            material.Specular.x = std::stof( parsed[1] );
            material.Specular.y = std::stof( parsed[2] );
            material.Specular.z = std::stof( parsed[3] );
         }
         else if (parsed[0] == "Ke") {
            material.Emission.x = std::stof( parsed[1] );
            material.Emission.y = std::stof( parsed[2] );
            material.Emission.z = std::stof( parsed[3] );
         }
         else if (parsed[0] == "Ns") material.SpecularExponent = std::stof( parsed[1] );
         else if (parsed[0] == "Ni") material.RefractiveIndex = std::stof( parsed[1] );
         else if (parsed[0] == "illum") {
            switch (std::stoi( parsed[1] )) {
               case 5: material.MaterialType = MATERIAL_TYPE::MIRROR; break;
               case 7: material.MaterialType = MATERIAL_TYPE::GLASS; break;
               default: material.MaterialType = MATERIAL_TYPE::LAMBERT; break;
            }
         }
      }
      return material;
   }

   void PhotonMap::setObjects(const std::vector<std::tuple<std::string, std::string, cuda::Mat>>& objects)
   {
      Box box;
      for (size_t i = 0; i < objects.size(); ++i) {
         Materials.emplace_back( getMaterial( std::get<1>( objects[i] ) ) );
         ToWorlds.emplace_back( std::get<2>( objects[i] ) );
         readObjectFile( box, ToWorlds[i], std::get<0>( objects[i] ) );
         WorldBounds.emplace_back( box.MinPoint, box.MaxPoint );
      }
   }

   void PhotonMap::setLights(const std::vector<std::tuple<std::string, std::string, cuda::Mat>>& lights)
   {
      Box box;
      for (const auto& light : lights) {
         const auto offset = static_cast<int>(Vertices.size());
         const auto index_offset = static_cast<int>(Indices.size());
         Materials.emplace_back( getMaterial( std::get<1>( light ) ) );
         ToWorlds.emplace_back( std::get<2>( light ) );
         readObjectFile( box, ToWorlds.back(), std::get<0>( light ) );
         WorldBounds.emplace_back( box.MinPoint, box.MaxPoint );

         const auto& m = ToWorlds.back();
         for (int j = index_offset; j < static_cast<int>(Indices.size()); j += 3) {
            const int n0 = offset + Indices[j];
            const int n1 = offset + Indices[j + 1];
            const int n2 = offset + Indices[j + 2];
            const float3 normal = cross( Vertices[n1] - Vertices[n0], Vertices[n2] - Vertices[n0] );
            AreaLights.emplace_back(
               length( normal ) * 0.5f,
               Materials.back().Emission,
               Normals[n0],
               Vertices[n0], Vertices[n1], Vertices[n2],
               m
            );
         }
      }
   }
}
#endif