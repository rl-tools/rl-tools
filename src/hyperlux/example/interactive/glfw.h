#pragma once

#include "input.h"

#include <GLFW/glfw3.h>

#include <cstdint>

namespace rl_tools::hyperlux::interactive {
    struct Glfw {
        GLFWwindow* window = nullptr;
        GLuint texture = 0;
        unsigned int width = 0;
        unsigned int height = 0;
        InputState* input = nullptr;
        double last_mouse_x = 0, last_mouse_y = 0;
        bool first_mouse = true;
        bool cursor_captured = true;
    };
    namespace glfw {
        inline void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
            InputState& input = *static_cast<Glfw*>(glfwGetWindowUserPointer(window))->input;
            if (action == GLFW_PRESS) {
                if (key == GLFW_KEY_ESCAPE) {
                    input.quit = true;
                    return;
                }
                if (key == GLFW_KEY_C) {
                    input.capture_pose = true;
                    return;
                }
                if (key == GLFW_KEY_TAB) {
                    input.cycle_view = true;
                    return;
                }
            }
            bool pressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
            if (key == GLFW_KEY_W || key == GLFW_KEY_UP) {
                input.forward = pressed;
            }
            if (key == GLFW_KEY_S || key == GLFW_KEY_DOWN) {
                input.backward = pressed;
            }
            if (key == GLFW_KEY_A || key == GLFW_KEY_LEFT) {
                input.left = pressed;
            }
            if (key == GLFW_KEY_D || key == GLFW_KEY_RIGHT) {
                input.right = pressed;
            }
            if (key == GLFW_KEY_SPACE) {
                input.up = pressed;
            }
            if (key == GLFW_KEY_LEFT_SHIFT) {
                input.down = pressed;
            }
        }

        inline void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
            Glfw& glfw = *static_cast<Glfw*>(glfwGetWindowUserPointer(window));
            if (!glfw.cursor_captured) {
                return;
            }
            if (glfw.first_mouse) {
                glfw.last_mouse_x = xpos;
                glfw.last_mouse_y = ypos;
                glfw.first_mouse = false;
                return;
            }
            look(*glfw.input, static_cast<float>(xpos - glfw.last_mouse_x), static_cast<float>(ypos - glfw.last_mouse_y));
            glfw.last_mouse_x = xpos;
            glfw.last_mouse_y = ypos;
        }

        inline void mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/) {
            Glfw& glfw = *static_cast<Glfw*>(glfwGetWindowUserPointer(window));
            if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
                if (glfw.cursor_captured) {
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                    glfw.cursor_captured = false;
                } else {
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    glfw.cursor_captured = true;
                    glfw.first_mouse = true;
                }
            }
        }
    }
}

namespace rl_tools {
    template <typename DEVICE>
    bool init(DEVICE& device, hyperlux::interactive::Glfw& glfw, unsigned int width, unsigned int height) {
        if (!glfwInit()) {
            log(device, device.logger, devices::logging::Verbosity::error, "Failed to initialize GLFW");
            return false;
        }
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        glfw.window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), "RLtools Interactive Viewer", nullptr, nullptr);
        if (!glfw.window) {
            log(device, device.logger, devices::logging::Verbosity::error, "Failed to create GLFW window");
            glfwTerminate();
            return false;
        }
        glfw.width = width;
        glfw.height = height;
        glfwSetWindowUserPointer(glfw.window, &glfw);
        glfwMakeContextCurrent(glfw.window);
        glfwSwapInterval(0);
        glfwSetKeyCallback(glfw.window, hyperlux::interactive::glfw::key_callback);
        glfwSetCursorPosCallback(glfw.window, hyperlux::interactive::glfw::cursor_position_callback);
        glfwSetMouseButtonCallback(glfw.window, hyperlux::interactive::glfw::mouse_button_callback);
        glfwSetInputMode(glfw.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        glGenTextures(1, &glfw.texture);
        glBindTexture(GL_TEXTURE_2D, glfw.texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, static_cast<GLsizei>(width), static_cast<GLsizei>(height), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        return true;
    }

    template <typename DEVICE>
    void poll(DEVICE& /*device*/, hyperlux::interactive::Glfw& glfw, hyperlux::interactive::InputState& input) {
        glfw.input = &input;
        glfwPollEvents();
        if (glfwWindowShouldClose(glfw.window)) {
            input.quit = true;
        }
    }

    template <typename DEVICE>
    void present(DEVICE& /*device*/, hyperlux::interactive::Glfw& glfw, const uint32_t* pixels) {
        glBindTexture(GL_TEXTURE_2D, glfw.texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(glfw.width), static_cast<GLsizei>(glfw.height), GL_RGBA, GL_UNSIGNED_BYTE, pixels);

        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 1); glVertex2f(-1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1, -1);
        glTexCoord2f(1, 0); glVertex2f( 1,  1);
        glTexCoord2f(0, 0); glVertex2f(-1,  1);
        glEnd();

        glfwSwapBuffers(glfw.window);
    }

    template <typename DEVICE>
    void free(DEVICE& /*device*/, hyperlux::interactive::Glfw& glfw) {
        glDeleteTextures(1, &glfw.texture);
        glfwDestroyWindow(glfw.window);
        glfwTerminate();
    }
}
