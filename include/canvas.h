#pragma once

#include "base.h"

class CanvasGL final
{
public:
   CanvasGL() : CanvasID( 0 ), COLOR0TextureID( 0 ) {}
   ~CanvasGL() { deleteAllTextures(); }

   [[nodiscard]] GLuint getCanvasID() const { return CanvasID; }
   [[nodiscard]] GLuint getColor0TextureID() const { return COLOR0TextureID; }
   void setCanvas(int width, int height, GLenum format)
   {
      deleteAllTextures();

      glCreateTextures( GL_TEXTURE_2D, 1, &COLOR0TextureID );
      glTextureStorage2D( COLOR0TextureID, 1, format, width, height );
      glTextureParameteri( COLOR0TextureID, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
      glTextureParameteri( COLOR0TextureID, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
      glTextureParameteri( COLOR0TextureID, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
      glTextureParameteri( COLOR0TextureID, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );

      glCreateFramebuffers( 1, &CanvasID );
      glNamedFramebufferTexture( CanvasID, GL_COLOR_ATTACHMENT0, COLOR0TextureID, 0 );

      glCheckNamedFramebufferStatus( CanvasID, GL_FRAMEBUFFER );
   }
   void clearColor() const
   {
      constexpr std::array<GLfloat, 4> clear_color = { 0.0f, 0.0f, 0.0f, 0.0f };
      glClearNamedFramebufferfv( CanvasID, GL_COLOR, 0, &clear_color[0] );
   }
   void clearColor(const std::array<GLfloat, 4>& color) const
   {
      glClearNamedFramebufferfv( CanvasID, GL_COLOR, 0, &color[0] );
   }
   void clearColor(const glm::vec4& color) const
   {
      clearColor( std::array<GLfloat, 4>{ color.r, color.g, color.b, color.a } );
   }

private:
   GLuint CanvasID;
   GLuint COLOR0TextureID;

   void deleteAllTextures()
   {
      if (COLOR0TextureID != 0) {
         glDeleteTextures( 1, &COLOR0TextureID );
         COLOR0TextureID = 0;
      }
      if (CanvasID != 0) {
         glDeleteFramebuffers( 1, &CanvasID );
         CanvasID = 0;
      }
   }
};