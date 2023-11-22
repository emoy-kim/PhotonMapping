/*
 * Author: Jeesun Kim
 * E-mail: emoy.kim_AT_gmail.com
 *
 */

#include "project_constants.h"
#include "cuda/photon_map.cuh"

int main()
{
   cuda::Mat cornell_box_scale(1.0f);
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
         std::string(sample_directory_path + "/CornellBox/light_ceiling.obj"),
         std::string(sample_directory_path + "/CornellBox/light_ceiling.mtl"),
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
   return 0;
}