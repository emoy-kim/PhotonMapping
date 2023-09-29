#pragma once

#include "shader.h"

class BuildPhotonMapShaderGL final : public ShaderGL
{
public:
   BuildPhotonMapShaderGL() = default;
   ~BuildPhotonMapShaderGL() override = default;

   BuildPhotonMapShaderGL(const BuildPhotonMapShaderGL&) = delete;
   BuildPhotonMapShaderGL(const BuildPhotonMapShaderGL&&) = delete;
   BuildPhotonMapShaderGL& operator=(const BuildPhotonMapShaderGL&) = delete;
   BuildPhotonMapShaderGL& operator=(const BuildPhotonMapShaderGL&&) = delete;

   void setUniformLocations() override;
};