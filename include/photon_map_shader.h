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

class VisualizePhotonMapShaderGL final : public ShaderGL
{
public:
   enum UNIFORM {
      NodeIndex = 0,
      Size,
      Dim,
      ObjectNum,
      InverseViewMatrix,
      WorldMatrices
   };

   VisualizePhotonMapShaderGL() = default;
   ~VisualizePhotonMapShaderGL() override = default;

   VisualizePhotonMapShaderGL(const VisualizePhotonMapShaderGL&) = delete;
   VisualizePhotonMapShaderGL(const VisualizePhotonMapShaderGL&&) = delete;
   VisualizePhotonMapShaderGL& operator=(const VisualizePhotonMapShaderGL&) = delete;
   VisualizePhotonMapShaderGL& operator=(const VisualizePhotonMapShaderGL&&) = delete;
};