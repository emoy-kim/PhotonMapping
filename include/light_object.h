#pragma once

#include "object.h"

class LightGL final : public ObjectGL
{
public:
   LightGL();
   ~LightGL() override = default;

   void toggleLightSwitch() { TurnLightOn = !TurnLightOn; }
   void setObjectWithTransform(
      GLenum draw_mode,
      const TYPE& type,
      const glm::mat4& transform,
      const std::string& obj_file_path,
      const std::string& mtl_file_path
   ) override;
   [[nodiscard]] float getSpotlightCutoffAngle() const { return SpotlightCutoffAngle; }
   [[nodiscard]] float getSpotlightFeather() const { return SpotlightFeather; }
   [[nodiscard]] float getFallOffRadius() const { return FallOffRadius; }
   [[nodiscard]] glm::vec3 getNormal() const { return SpotlightDirection; }
   [[nodiscard]] glm::vec4 getCentroid() const;
   [[nodiscard]] const std::vector<float>& getAreas() const { return Areas; }
   [[nodiscard]] const std::vector<std::array<glm::vec3, 3>>& getTriangles() const { return Triangles; }

private:
   bool TurnLightOn;
   float SpotlightCutoffAngle;
   float SpotlightFeather;
   float FallOffRadius;
   glm::vec3 SpotlightDirection;
   std::vector<float> Areas;
   std::vector<std::array<glm::vec3, 3>> Triangles;
};