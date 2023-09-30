#include "renderer.h"

RendererGL::RendererGL() :
   Window( nullptr ), Pause( false ), NeedToUpdate( true ), FrameWidth( 1024 ), FrameHeight( 1024 ),
   ClickedPoint( -1, -1 ), MainCamera( std::make_unique<CameraGL>() ), SceneShader( std::make_unique<ShaderGL>() ),
   PhotonMap( std::make_unique<PhotonMapGL>() ), KdtreeBuilder(), PhotonMapBuilder()
{
   Renderer = this;

   initialize();
   printOpenGLInformation();
}

void RendererGL::printOpenGLInformation()
{
   std::cout << "**********************************************************************************\n";
   std::cout << " - GLFW version supported: " << glfwGetVersionString() << "\n";
   std::cout << " - OpenGL renderer: " << glGetString( GL_RENDERER ) << "\n";
   std::cout << " - OpenGL version supported: " << glGetString( GL_VERSION ) << "\n";
   std::cout << " - OpenGL shader version supported: " << glGetString( GL_SHADING_LANGUAGE_VERSION ) << "\n";

   int work_group_count = 0;
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_group_count );
   std::cout << " - OpenGL maximum number of work groups: " <<  work_group_count << ", ";
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_group_count );
   std::cout << work_group_count << ", ";
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_group_count );
   std::cout << work_group_count << "\n";

   int work_group_size = 0;
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_group_size );
   std::cout << " - OpenGL maximum work group size: " <<  work_group_size << ", ";
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_group_size );
   std::cout << work_group_size << ", ";
   glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_group_size );
   std::cout << work_group_size << "\n";
   std::cout << "**********************************************************************************\n\n";
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

void RendererGL::cleanup(GLFWwindow* window)
{
   glfwSetWindowShouldClose( window, GLFW_TRUE );
}

void RendererGL::keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
   if (action != GLFW_PRESS) return;

   switch (key) {
      case GLFW_KEY_C:
         Renderer->writeFrame();
         std::cout << ">> Framebuffer Captured\n";
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

   if (Renderer->MainCamera->getMovingState()) {
      const auto x = static_cast<int>(std::round( xpos ));
      const auto y = static_cast<int>(std::round( ypos ));
      const int dx = x - Renderer->ClickedPoint.x;
      const int dy = y - Renderer->ClickedPoint.y;
      Renderer->MainCamera->moveForward( -dy );
      Renderer->MainCamera->rotateAroundWorldY( -dx );

      if (glfwGetMouseButton( window, GLFW_MOUSE_BUTTON_RIGHT ) == GLFW_PRESS) {
         Renderer->MainCamera->pitch( -dy );
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
      Renderer->MainCamera->setMovingState( moving_state );
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
         std::string(sample_directory_path + "/Tiger/tiger.obj"), {}, ObjectGL::TYPE::ARBITRARY,
         glm::vec4(1.0f),
         glm::translate( glm::mat4(1.0f), glm::vec3(200.0f, 0.0f, 0.0f) ) * to_tiger_object
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/Tiger/tiger.obj"), {}, ObjectGL::TYPE::ARBITRARY,
         glm::vec4(1.0f),
         glm::translate( glm::mat4(1.0f), glm::vec3(-150.0f, 0.0f, 0.0f) ) * to_tiger_object
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/Tiger/tiger.obj"), {}, ObjectGL::TYPE::ARBITRARY,
         glm::vec4(1.0f),
         glm::translate( glm::mat4(1.0f), glm::vec3(50.0f, 0.0f, -100.0f) ) * to_tiger_object
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/Tiger/tiger.obj"), {}, ObjectGL::TYPE::ARBITRARY,
         glm::vec4(1.0f),
         glm::translate( glm::mat4(1.0f), glm::vec3(50.0f, 30.0f, 200.0f) ) *
         glm::rotate( glm::mat4(1.0f), glm::radians( -30.0f ), glm::vec3(1.0f, 0.0f, 0.0f) ) * to_tiger_object
      ),*/
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/floor.obj"),
         std::string(sample_directory_path + "/CornellBox/floor.mtl"),
         ObjectGL::TYPE::PLANE, cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/ceiling.obj"),
         std::string(sample_directory_path + "/CornellBox/ceiling.mtl"),
         ObjectGL::TYPE::PLANE, cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/back_wall.obj"),
         std::string(sample_directory_path + "/CornellBox/back_wall.mtl"),
         ObjectGL::TYPE::PLANE, cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/left_wall.obj"),
         std::string(sample_directory_path + "/CornellBox/left_wall.mtl"),
         ObjectGL::TYPE::PLANE, cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/right_wall.obj"),
         std::string(sample_directory_path + "/CornellBox/right_wall.mtl"),
         ObjectGL::TYPE::PLANE, cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/left_sphere.obj"),
         std::string(sample_directory_path + "/CornellBox/left_sphere.mtl"),
         ObjectGL::TYPE::SPHERE, cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/right_sphere.obj"),
         std::string(sample_directory_path + "/CornellBox/right_sphere.mtl"),
         ObjectGL::TYPE::SPHERE, cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/water.obj"),
         std::string(sample_directory_path + "/CornellBox/water.mtl"),
         ObjectGL::TYPE::ARBITRARY, cornell_box_scale
      ),
      std::make_tuple(
         std::string(sample_directory_path + "/CornellBox/light.obj"),
         std::string(sample_directory_path + "/CornellBox/light.mtl"),
         ObjectGL::TYPE::LIGHT, cornell_box_scale
      )
   };
   PhotonMap->setObjects( objects );
}

void RendererGL::setShaders() const
{
   const std::string shader_directory_path = std::string(CMAKE_SOURCE_DIR) + "/shaders";
   SceneShader->setShader(
      std::string(shader_directory_path + "/scene_shader.vert").c_str(),
      std::string(shader_directory_path + "/scene_shader.frag").c_str()
   );
   SceneShader->setSceneUniformLocations( PhotonMap->getLightNum() );
   setKdtreeShaders();
   setPhotonMapShaders();
}

void RendererGL::drawScene() const
{
   glViewport( 0, 0, FrameWidth, FrameHeight );
   glBindFramebuffer( GL_FRAMEBUFFER, 0 );
   glUseProgram( SceneShader->getShaderProgram() );

   const LightGL* light = PhotonMap->getLight( 0 );
   light->transferUniformsToShader( SceneShader.get(), 0 );

   SceneShader->uniform1i( "UseLight", 1 );
   SceneShader->uniform1i( "UseTexture", 0 );
   SceneShader->uniform1i( "LightNum", 1 );
   const auto& objects = PhotonMap->getObjects();
   const auto& to_worlds = PhotonMap->getWorldMatrices();
   for (size_t i = 0; i < objects.size(); ++i) {
      glBindVertexArray( objects[i]->getVAO() );
      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, objects[i]->getIBO() );
      objects[i]->transferUniformsToShader( SceneShader.get() );
      SceneShader->transferBasicTransformationUniforms( to_worlds[i], MainCamera.get() );
      glDrawElements( objects[i]->getDrawMode(), objects[i]->getIndexNum(), GL_UNSIGNED_INT, nullptr );
   }
}

void RendererGL::render()
{
   if (NeedToUpdate) {
      glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
      drawScene();
      NeedToUpdate = false;
   }
}

void RendererGL::play()
{
   if (glfwWindowShouldClose( Window )) initialize();

   setObjects();
   setShaders();
   createPhotonMap();
   //buildKdtree();

   glfwShowWindow( Window );
   while (!glfwWindowShouldClose( Window )) {
      if (!Pause) render();

      glfwSwapBuffers( Window );
      glfwPollEvents();
   }
   glfwDestroyWindow( Window );
}