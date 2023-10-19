#include "cuda/photon_map.cuh"
#include "renderer.h"

void testCUDA()
{
   const glm::mat4 cornell_box_scale = glm::scale( glm::mat4(1.0f), glm::vec3(300.0f) );
   const std::string sample_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples";
   const std::vector<std::tuple<std::string, std::string, glm::mat4>> objects = {
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
         std::string(sample_directory_path + "/CornellBox/water.obj"),
         std::string(sample_directory_path + "/CornellBox/water.mtl"),
         cornell_box_scale
      )
   };

   const std::vector<std::tuple<std::string, std::string, glm::mat4>> lights = {
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
   //photon_map.visualizePhotonMap( 1024, 1024 );
}

int main()
{
   testCUDA();

   //RendererGL renderer;
   //renderer.play();
   return 0;
}