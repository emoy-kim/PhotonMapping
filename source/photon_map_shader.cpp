#include "photon_map_shader.h"

void BuildPhotonMapShaderGL::setUniformLocations()
{
   addUniformLocation( "Seed" );
   addUniformLocation( "MaxGlobalPhotonNum" );
   addUniformLocation( "MaxDepth" );
   addUniformLocation( "ObjectNum" );
}