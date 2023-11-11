#include "cuda/photon_map.cuh"
#include "renderer.h"

void testCUDA()
{
   const glm::mat4 to_world(1.0f);
   cuda::Mat cornell_box_scale;
   cornell_box_scale.c0 = make_float4( to_world[0][0], to_world[0][1], to_world[0][2], to_world[0][3] );
   cornell_box_scale.c1 = make_float4( to_world[1][0], to_world[1][1], to_world[1][2], to_world[1][3] );
   cornell_box_scale.c2 = make_float4( to_world[2][0], to_world[2][1], to_world[2][2], to_world[2][3] );
   cornell_box_scale.c3 = make_float4( to_world[3][0], to_world[3][1], to_world[3][2], to_world[3][3] );

   const std::string sample_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples";
   const std::vector<std::tuple<std::string, std::string, cuda::Mat>> objects = {
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/floor.obj"),
         std::string(sample_directory_path + "/CornellBox/floor.mtl"),
         cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/ceiling.obj"),
         std::string(sample_directory_path + "/CornellBox/ceiling.mtl"),
         cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/back_wall.obj"),
         std::string(sample_directory_path + "/CornellBox/back_wall.mtl"),
         cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/left_wall.obj"),
         std::string(sample_directory_path + "/CornellBox/left_wall.mtl"),
         cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/right_wall.obj"),
         std::string(sample_directory_path + "/CornellBox/right_wall.mtl"),
         cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/left_sphere.obj"),
         std::string(sample_directory_path + "/CornellBox/left_sphere.mtl"),
         cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/right_sphere.obj"),
         std::string(sample_directory_path + "/CornellBox/right_sphere.mtl"),
         cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/water_surface.obj"),
         std::string(sample_directory_path + "/CornellBox/water.mtl"),
         cornell_box_scale
      )
   };

   const std::vector<std::tuple<std::string, std::string, cuda::Mat>> lights = {
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/light.obj"),
         std::string(sample_directory_path + "/CornellBox/light.mtl"),
         cornell_box_scale
      )
   };

   cuda::PhotonMap photon_map;
   photon_map.setObjects( objects );
   photon_map.setLights( lights );
   photon_map.createPhotonMap();
   //photon_map.visualizeGlobalPhotonMap( 800, 800 );
   //photon_map.visualizeCausticPhotonMap( 800, 800 );
   photon_map.render( 800, 800 );
}

int main()
{
   testCUDA();

   //RendererGL renderer;
   //renderer.play();
   return 0;
}