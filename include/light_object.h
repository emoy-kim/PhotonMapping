#pragma once

#include "object.h"

class LightGL final : public ObjectGL
{
public:
   struct AreaLightInfo
   {
      alignas(4) float Area;
      alignas(16) glm::vec3 Emission;
      alignas(16) glm::vec3 Normal;
      alignas(16) glm::vec3 Vertices[4];

      AreaLightInfo() : Area( 0.0f ), Emission(), Normal(), Vertices{} {}
   };

   LightGL();
   ~LightGL() override = default;

   void toggleLightSwitch() { TurnLightOn = !TurnLightOn; }
   void transferUniformsToShader(const ShaderGL* shader, int index) const;
   void setObjectWithTransform(
      GLenum draw_mode,
      const TYPE& type,
      const glm::mat4& transform,
      const std::string& obj_file_path,
      const std::string& mtl_file_path
   ) override;

private:
   bool TurnLightOn;
   float SpotlightCutoffAngle;
   float SpotlightFeather;
   float FallOffRadius;
   glm::vec3 SpotlightDirection;
   AreaLightInfo AreaLight;
};