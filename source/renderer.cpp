#include "renderer.h"

RendererGL::RendererGL() :
   Window( nullptr ), Pause( false ), NeedToUpdate( true ), FrameWidth( 1024 ), FrameHeight( 1024 ),
   ShadowMapSize( 1024 ), DepthFBO( 0 ), DepthTextureArrayID( 0 ), ClickedPoint( -1, -1 ), ActiveCamera( nullptr ),
   MainCamera( std::make_unique<CameraGL>() ), PCFSceneShader( std::make_unique<ShaderGL>() ),
   LightViewDepthShader( std::make_unique<ShaderGL>() ), Lights( std::make_unique<LightGL>() ),
   PhotonMap( std::make_unique<PhotonMapGL>() ), KdtreeBuilder()
{
   Renderer = this;

   initialize();
   printOpenGLInformation();
}

RendererGL::~RendererGL()
{
   if (DepthTextureArrayID != 0) glDeleteTextures( 1, &DepthTextureArrayID );
   if (DepthFBO != 0) glDeleteFramebuffers( 1, &DepthFBO );
}

void RendererGL::printOpenGLInformation()
{
   std::cout << "****************************************************************\n";
   std::cout << " - GLFW version supported: " << glfwGetVersionString() << "\n";
   std::cout << " - OpenGL renderer: " << glGetString( GL_RENDERER ) << "\n";
   std::cout << " - OpenGL version supported: " << glGetString( GL_VERSION ) << "\n";
   std::cout << " - OpenGL shader version supported: " << glGetString( GL_SHADING_LANGUAGE_VERSION ) << "\n";
   std::cout << "****************************************************************\n\n";
}

void RendererGL::initialize()
{
   if (!glfwInit()) {
      std::cout << "Cannot Initialize OpenGL...\n";
      return;
   }
   glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
   glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 6 );
   glfwWindowHint( GLFW_DOUBLEBUFFER, GLFW_TRUE );
   glfwWindowHint( GLFW_RESIZABLE, GLFW_FALSE );
   glfwWindowHint( GLFW_VISIBLE, GLFW_FALSE );
   glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

   Window = glfwCreateWindow( FrameWidth, FrameHeight, "Cinematic Relighting", nullptr, nullptr );
   glfwMakeContextCurrent( Window );

   if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
      std::cout << "Failed to initialize GLAD" << std::endl;
      return;
   }

   registerCallbacks();

   glEnable( GL_CULL_FACE );
   glEnable( GL_DEPTH_TEST );
   glClearColor( 0.094, 0.07f, 0.17f, 1.0f );

   MainCamera->updatePerspectiveCamera( FrameWidth, FrameHeight );
}

void RendererGL::writeFrame() const
{
   const int size = FrameWidth * FrameHeight * 3;
   auto* buffer = new uint8_t[size];
   glBindFramebuffer( GL_FRAMEBUFFER, 0 );
   glNamedFramebufferReadBuffer( 0, GL_COLOR_ATTACHMENT0 );
   glReadPixels( 0, 0, FrameWidth, FrameHeight, GL_BGR, GL_UNSIGNED_BYTE, buffer );
   FIBITMAP* image = FreeImage_ConvertFromRawBits(
      buffer, FrameWidth, FrameHeight, FrameWidth * 3, 24,
      FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false
   );
   FreeImage_Save( FIF_PNG, image, "../result.png" );
   FreeImage_Unload( image );
   delete [] buffer;
}

void RendererGL::writeDepthTextureArray() const
{
   const int size = ShadowMapSize * ShadowMapSize;
   auto* buffer = new uint8_t[size];
   auto* raw_buffer = new GLfloat[size];
   glBindFramebuffer( GL_FRAMEBUFFER, DepthFBO );
   for (int s = 0; s < Lights->getTotalLightNum(); ++s) {
      glNamedFramebufferReadBuffer( DepthFBO, GL_DEPTH_ATTACHMENT );
      glNamedFramebufferTextureLayer( DepthFBO, GL_DEPTH_ATTACHMENT, DepthTextureArrayID, 0, s );
      glReadPixels( 0, 0, ShadowMapSize, ShadowMapSize, GL_DEPTH_COMPONENT, GL_FLOAT, raw_buffer );

      for (int i = 0; i < size; ++i) {
         buffer[i] = static_cast<uint8_t>(LightCameras[s]->linearizeDepthValue( raw_buffer[i] ) * 255.0f);
      }

      FIBITMAP* image = FreeImage_ConvertFromRawBits(
         buffer, ShadowMapSize, ShadowMapSize, ShadowMapSize, 8,
         FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false
      );
      FreeImage_Save( FIF_PNG, image, std::string("../depth" + std::to_string( s ) + ".png").c_str() );
      FreeImage_Unload( image );
   }
   delete [] raw_buffer;
   delete [] buffer;
}

void RendererGL::cleanup(GLFWwindow* window)
{
   glfwSetWindowShouldClose( window, GLFW_TRUE );
}

void RendererGL::keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
   if (action != GLFW_PRESS) return;

   switch (key) {
      case GLFW_KEY_0:
         Renderer->ActiveCamera = Renderer->MainCamera.get();
         std::cout << ">> Main Camera Selected\n";
         break;
      case GLFW_KEY_1:
         Renderer->ActiveCamera = Renderer->LightCameras[0].get();
         std::cout << ">> Light-1 Selected\n";
         break;
      case GLFW_KEY_C:
         Renderer->writeFrame();
         std::cout << ">> Framebuffer Captured\n";
         break;
      case GLFW_KEY_D:
         Renderer->writeDepthTextureArray();
         std::cout << ">> Depth Array Captured\n";
         break;
      case GLFW_KEY_L:
         Renderer->NeedToUpdate = true;
         Renderer->Lights->toggleLightSwitch();
         std::cout << ">> Light Turned " << (Renderer->Lights->isLightOn() ? "On!\n" : "Off!\n");
         break;
      case GLFW_KEY_P: {
         const glm::vec3 pos = Renderer->MainCamera->getCameraPosition();
         std::cout << ">> Camera Position: " << pos.x << ", " << pos.y << ", " << pos.z << "\n";
      } break;
      case GLFW_KEY_SPACE:
         Renderer->Pause = !Renderer->Pause;
         break;
      case GLFW_KEY_Q:
      case GLFW_KEY_ESCAPE:
         cleanup( window );
         break;
      default:
         return;
   }
}

void RendererGL::cursor(GLFWwindow* window, double xpos, double ypos)
{
   if (Renderer->Pause) return;

   if (Renderer->ActiveCamera->getMovingState()) {
      const auto x = static_cast<int>(std::round( xpos ));
      const auto y = static_cast<int>(std::round( ypos ));
      const int dx = x - Renderer->ClickedPoint.x;
      const int dy = y - Renderer->ClickedPoint.y;
      Renderer->ActiveCamera->moveForward( -dy );
      Renderer->ActiveCamera->rotateAroundWorldY( -dx );

      if (glfwGetMouseButton( window, GLFW_MOUSE_BUTTON_RIGHT ) == GLFW_PRESS) {
         Renderer->ActiveCamera->pitch( -dy );
      }

      Renderer->ClickedPoint.x = x;
      Renderer->ClickedPoint.y = y;
      Renderer->NeedToUpdate = true;
   }
}

void RendererGL::mouse(GLFWwindow* window, int button, int action, int mods)
{
   if (Renderer->Pause) return;

   if (button == GLFW_MOUSE_BUTTON_LEFT) {
      const bool moving_state = action == GLFW_PRESS;
      if (moving_state) {
         double x, y;
         glfwGetCursorPos( window, &x, &y );
         Renderer->ClickedPoint.x = static_cast<int>(std::round( x ));
         Renderer->ClickedPoint.y = static_cast<int>(std::round( y ));
      }
      Renderer->ActiveCamera->setMovingState( moving_state );
   }
}

void RendererGL::registerCallbacks() const
{
   glfwSetWindowCloseCallback( Window, cleanup );
   glfwSetKeyCallback( Window, keyboard );
   glfwSetCursorPosCallback( Window, cursor );
   glfwSetMouseButtonCallback( Window, mouse );
}

void RendererGL::setObjects()
{
   const glm::mat4 to_tiger_object =
      glm::translate( glm::mat4(1.0f), glm::vec3(0.0f, 50.0f, 0.0f) ) *
      glm::scale( glm::mat4(1.0f), glm::vec3(0.3f) );
   const glm::mat4 cornell_box_scale = glm::scale( glm::mat4(1.0f), glm::vec3(300.0f) );
   const std::string sample_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples";
   const std::vector<object_t> objects = {
      /*std::make_tuple(
         std::string(sample_directory_path + "/Tiger/tiger.obj"), ObjectGL::TYPE::ARBITRARY,
         glm::vec4(1.0f),
         glm::translate( glm::mat4(1.0f), glm::vec3(200.0f, 0.0f, 0.0f) ) * to_tiger_object
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/Tiger/tiger.obj"), ObjectGL::TYPE::ARBITRARY,
         glm::vec4(1.0f),
         glm::translate( glm::mat4(1.0f), glm::vec3(-150.0f, 0.0f, 0.0f) ) * to_tiger_object
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/Tiger/tiger.obj"), ObjectGL::TYPE::ARBITRARY,
         glm::vec4(1.0f),
         glm::translate( glm::mat4(1.0f), glm::vec3(50.0f, 0.0f, -100.0f) ) * to_tiger_object
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/Tiger/tiger.obj"), ObjectGL::TYPE::ARBITRARY,
         glm::vec4(1.0f),
         glm::translate( glm::mat4(1.0f), glm::vec3(50.0f, 30.0f, 200.0f) ) *
         glm::rotate( glm::mat4(1.0f), glm::radians( -30.0f ), glm::vec3(1.0f, 0.0f, 0.0f) ) * to_tiger_object
      ),*/
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/floor.obj"), ObjectGL::TYPE::PLANE,
         glm::vec4(1.0f), cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/ceiling.obj"), ObjectGL::TYPE::PLANE,
         glm::vec4(1.0f), cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/back_wall.obj"), ObjectGL::TYPE::PLANE,
         glm::vec4(1.0f), cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/left_wall.obj"), ObjectGL::TYPE::PLANE,
         glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/right_wall.obj"), ObjectGL::TYPE::PLANE,
         glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/left_sphere.obj"), ObjectGL::TYPE::SPHERE,
         glm::vec4(1.0f), cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/right_sphere.obj"), ObjectGL::TYPE::SPHERE,
         glm::vec4(1.0f), cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/water.obj"), ObjectGL::TYPE::ARBITRARY,
         glm::vec4(1.0f), cornell_box_scale
      )
   };
   PhotonMap->setObjects( objects );
}

void RendererGL::setLightViewFrameBuffers()
{
   glCreateTextures( GL_TEXTURE_2D_ARRAY, 1, &DepthTextureArrayID );
   glTextureStorage3D( DepthTextureArrayID, 1, GL_DEPTH_COMPONENT32F, ShadowMapSize, ShadowMapSize, Lights->getTotalLightNum() );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE );
   glTextureParameteri( DepthTextureArrayID, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL );

   glCreateFramebuffers( 1, &DepthFBO );
   for (int i = 0; i < Lights->getTotalLightNum(); ++i) {
      glNamedFramebufferTextureLayer( DepthFBO, GL_DEPTH_ATTACHMENT, DepthTextureArrayID, 0, i );
   }

   if (glCheckNamedFramebufferStatus( DepthFBO, GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE) {
      std::cerr << "DepthFBO Setup Error\n";
   }
}

void RendererGL::setLights()
{
   glm::vec4 light_position(0.0f, 477.0f, 0.0f, 1.0f);
   glm::vec4 ambient_color(0.78f, 0.78f, 0.78f, 1.0f);
   glm::vec4 diffuse_color(0.78f, 0.78f, 0.78f, 1.0f);
   glm::vec4 specular_color(0.0f, 0.0f, 0.0f, 1.0f);
   const glm::vec3 reference_position(0.0f, 0.0f, 0.0f);
   Lights->addLight(
      light_position, ambient_color, diffuse_color, specular_color,
      reference_position - glm::vec3(light_position),
      180.0f,
      0.5f,
      1000.0f
   );

   const std::vector<glm::vec3> reference_points = {
      glm::vec3(Lights->getLightPosition( 0 )) + Lights->getSpotlightDirection( 0 )
   };

   const int light_num = Lights->getTotalLightNum();
   LightViewMatrices.resize( light_num );
   LightViewProjectionMatrices.resize( light_num );
   LightCameras.resize( light_num );
   for (int i = 0; i < light_num; ++i) {
      LightCameras[i] = std::make_unique<CameraGL>();
      LightCameras[i]->updatePerspectiveCamera( ShadowMapSize, ShadowMapSize );
      LightCameras[i]->updateNearFarPlanes( 100.0f, 1000.0f );
      LightCameras[i]->updateCameraView(
         glm::vec3(Lights->getLightPosition( i )),
         reference_points[i],
         glm::vec3(0.0f, 1.0f, 0.0f)
      );
   }
   ActiveCamera = LightCameras[0].get();

   setLightViewFrameBuffers();
}

void RendererGL::setShaders() const
{
   const std::string shader_directory_path = std::string(CMAKE_SOURCE_DIR) + "/shaders";
   LightViewDepthShader->setShader(
      std::string(shader_directory_path + "/shadow/light_view_depth_generator.vert").c_str(),
      std::string(shader_directory_path + "/shadow/light_view_depth_generator.frag").c_str()
   );
   PCFSceneShader->setShader(
      std::string(shader_directory_path + "/shadow/scene_shader.vert").c_str(),
      std::string(shader_directory_path + "/shadow/scene_shader.frag").c_str()
   );
   LightViewDepthShader->setLightViewUniformLocations();
   PCFSceneShader->setSceneUniformLocations( Lights->getTotalLightNum() );

   setKdtreeShaders();
}

void RendererGL::drawObjects(ShaderGL* shader, CameraGL* camera) const
{
   const auto& objects = PhotonMap->getObjects();
   const auto& to_worlds = PhotonMap->getWorldMatrices();
   for (size_t i = 0; i < objects.size(); ++i) {
      glBindVertexArray( objects[i]->getVAO() );
      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, objects[i]->getIBO() );
      objects[i]->transferUniformsToShader( shader );
      shader->transferBasicTransformationUniforms( to_worlds[i], camera );
      glDrawElements( objects[i]->getDrawMode(), objects[i]->getIndexNum(), GL_UNSIGNED_INT, nullptr );
   }
}

void RendererGL::drawDepthMapFromLightView()
{
   glViewport( 0, 0, ShadowMapSize, ShadowMapSize );
   glBindFramebuffer( GL_FRAMEBUFFER, DepthFBO );
   glUseProgram( LightViewDepthShader->getShaderProgram() );
   glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );

   for (int i = 0; i < Lights->getTotalLightNum(); ++i) {
      glNamedFramebufferTextureLayer( DepthFBO, GL_DEPTH_ATTACHMENT, DepthTextureArrayID, 0, i );

      constexpr GLfloat one = 1.0f;
      glClearNamedFramebufferfv( DepthFBO, GL_DEPTH, 0, &one );

      drawObjects( LightViewDepthShader.get(), LightCameras[i].get() );

      LightViewMatrices[i] = LightCameras[i]->getViewMatrix();
      LightViewProjectionMatrices[i] = LightCameras[i]->getProjectionMatrix() * LightCameras[i]->getViewMatrix();
   }
   glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
}

void RendererGL::drawSceneWithShadow() const
{
   glViewport( 0, 0, FrameWidth, FrameHeight );
   glBindFramebuffer( GL_FRAMEBUFFER, 0 );
   glUseProgram( PCFSceneShader->getShaderProgram() );

   Lights->transferUniformsToShader( PCFSceneShader.get() );

   PCFSceneShader->uniform4fv( "ShadowColor", { 0.24, 0.16f, 0.13f, 1.0f } );
   PCFSceneShader->uniformMat4fv( "LightViewMatrix", Lights->getTotalLightNum(), LightViewMatrices.data() );
   PCFSceneShader->uniformMat4fv( "LightViewProjectionMatrix", Lights->getTotalLightNum(), LightViewProjectionMatrices.data() );

   glBindTextureUnit( 1, DepthTextureArrayID );

   PCFSceneShader->uniform1i( "UseTexture", 0 );
   drawObjects( PCFSceneShader.get(), MainCamera.get() );
}

void RendererGL::render()
{
   if (NeedToUpdate) {
      glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
      drawDepthMapFromLightView();
      drawSceneWithShadow();
      NeedToUpdate = false;
   }
}

void RendererGL::play()
{
   if (glfwWindowShouldClose( Window )) initialize();

   setObjects();
   setLights();
   setShaders();
   //buildKdtree();

   glfwShowWindow( Window );
   while (!glfwWindowShouldClose( Window )) {
      if (!Pause) render();

      glfwSwapBuffers( Window );
      glfwPollEvents();
   }
   glfwDestroyWindow( Window );
}