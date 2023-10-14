#pragma once

#include "shader.h"

class BuildPhotonMapShaderGL final : public ShaderGL
{
public:
   enum UNIFORM {
      Seed = 0,
      MaxGlobalPhotonNum,
      MaxDepth,
      ObjectNum,
      ObjectBRDFs,
      ObjectMaterialTypes = 14,
      WorldMatrices = 24
   };

   BuildPhotonMapShaderGL() = default;
   ~BuildPhotonMapShaderGL() override = default;

   BuildPhotonMapShaderGL(const BuildPhotonMapShaderGL&) = delete;
   BuildPhotonMapShaderGL(const BuildPhotonMapShaderGL&&) = delete;
   BuildPhotonMapShaderGL& operator=(const BuildPhotonMapShaderGL&) = delete;
   BuildPhotonMapShaderGL& operator=(const BuildPhotonMapShaderGL&&) = delete;
};