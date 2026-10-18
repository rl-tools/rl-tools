#include "ui_frontends.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#include <pthread.h>
#else
#include <GL/gl.h>
#endif
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux::spy::ui::glfw {
    struct Window {
        GLFWwindow* handle = nullptr;
        ImGuiContext* imgui = nullptr;
        GLuint atlas = 0;
        GLuint spectator = 0;
        std::vector<std::string> labels;
        std::size_t selected = 0;
        int atlas_width = 0;
        int atlas_height = 0;
        int columns = 0;
        int rows = 0;
        int spectator_width = 0;
        int spectator_height = 0;
        bool glfw_backend = false;
        bool opengl_backend = false;
        bool mouse_captured = false;
        double mouse_x = 0;
        double mouse_y = 0;
    };

    namespace {
        void require_main_thread() {
#ifdef __APPLE__
            if(!pthread_main_np()) {
                throw std::runtime_error("HyperLux spy must run on the main thread");
            }
#elif defined(__linux__)
            if(syscall(SYS_gettid) != getpid()) {
                throw std::runtime_error("HyperLux spy must run on the main thread");
            }
#endif
        }

        struct ContextScope {
            GLFWwindow* glfw = glfwGetCurrentContext();
            ImGuiContext* imgui = ImGui::GetCurrentContext();
            bool graphics;

            explicit ContextScope(Window* window, bool use_graphics = true): graphics(use_graphics) {
                if(graphics) {
                    glfwMakeContextCurrent(window->handle);
                }
                ImGui::SetCurrentContext(window->imgui);
            }

            ~ContextScope() {
                ImGui::SetCurrentContext(imgui);
                if(graphics) {
                    glfwMakeContextCurrent(glfw);
                }
            }
        };

        template <auto CALLBACK, typename... ARGS>
        void forward_event(GLFWwindow* handle, ARGS... args) {
            auto* window = static_cast<Window*>(glfwGetWindowUserPointer(handle));
            ContextScope context(window, false);
            CALLBACK(handle, args...);
        }

        void release_mouse(Window* window) {
            if(window->mouse_captured) {
                glfwSetInputMode(window->handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                window->mouse_captured = false;
            }
        }

        void release_graphics(Window* window) {
            glfwSetWindowFocusCallback(window->handle, nullptr);
            glfwSetCursorEnterCallback(window->handle, nullptr);
            glfwSetCursorPosCallback(window->handle, nullptr);
            glfwSetMouseButtonCallback(window->handle, nullptr);
            glfwSetScrollCallback(window->handle, nullptr);
            glfwSetKeyCallback(window->handle, nullptr);
            glfwSetCharCallback(window->handle, nullptr);
            release_mouse(window);
            if(window->opengl_backend) {
                ImGui_ImplOpenGL3_Shutdown();
            }
            if(window->glfw_backend) {
                ImGui_ImplGlfw_Shutdown();
            }
            if(window->imgui != nullptr) {
                ImGui::DestroyContext(window->imgui);
                window->imgui = nullptr;
            }
            glDeleteTextures(1, &window->atlas);
            glDeleteTextures(1, &window->spectator);
        }

        GLuint create_texture(int width, int height) {
            std::vector<std::uint32_t> pixels(static_cast<std::size_t>(width) * height, 0xff171717u);
            GLuint texture = 0;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
            glBindTexture(GL_TEXTURE_2D, 0);
            if(glGetError() != GL_NO_ERROR) {
                glDeleteTextures(1, &texture);
                throw std::runtime_error("HyperLux spy could not allocate its display textures");
            }
            return texture;
        }

        void upload(Window* window, GLuint texture, int width, int height, const std::uint32_t* pixels) {
            ContextScope context(window);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        ImVec2 image_size(int width, int height) {
            const auto available = ImGui::GetContentRegionAvail();
            const float scale = std::max(0.01f, std::min(available.x / width, available.y / height));
            return ImVec2(width * scale, height * scale);
        }

        void frame_age(const char* label, double age) {
            if(std::isfinite(age) && age >= 0) {
                ImGui::Text("%s: %.0f ms old", label, age * 1000);
            }
            else {
                ImGui::Text("%s: waiting for a frame", label);
            }
        }

        void draw(Window* window, Input& input, double atlas_age, double spectator_age, const char* status) {
            const auto previous_selection = window->selected;
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
            ImGui::Begin("Spy", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
            if(!window->labels.empty()) {
                const float left_width = ImGui::GetContentRegionAvail().x * 0.45f;
                ImGui::BeginChild("Cameras", ImVec2(left_width, 0));
                frame_age("Cameras", atlas_age);
                ImGui::TextUnformatted(window->labels[window->selected].c_str());
                const auto atlas_size = image_size(window->atlas_width, window->atlas_height);
                const auto atlas_origin = ImGui::GetCursorScreenPos();
                ImGui::Image((ImTextureID)(std::intptr_t)window->atlas, atlas_size);
                if(!window->mouse_captured && ImGui::IsItemHovered()) {
                    const auto cursor = ImGui::GetIO().MousePos;
                    const auto column = std::min(window->columns - 1, static_cast<int>((cursor.x - atlas_origin.x) * window->columns / atlas_size.x));
                    const auto row = std::min(window->rows - 1, static_cast<int>((cursor.y - atlas_origin.y) * window->rows / atlas_size.y));
                    const auto camera = static_cast<std::size_t>(row) * window->columns + column;
                    if(camera < window->labels.size()) {
                        ImGui::BeginTooltip();
                        ImGui::TextUnformatted(window->labels[camera].c_str());
                        ImGui::EndTooltip();
                        if(ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                            window->selected = camera;
                        }
                    }
                }
                const float tile_width = atlas_size.x / window->columns;
                const float tile_height = atlas_size.y / window->rows;
                const ImVec2 tile_origin(atlas_origin.x + (window->selected % window->columns) * tile_width, atlas_origin.y + (window->selected / window->columns) * tile_height);
                ImGui::GetWindowDrawList()->AddRect(tile_origin, ImVec2(tile_origin.x + tile_width, tile_origin.y + tile_height), IM_COL32(255, 194, 62, 255), 0.0f, 2.0f);
                ImGui::EndChild();
                ImGui::SameLine();
            }
            ImGui::BeginChild("Spectator", ImVec2(0, 0));
            if(window->selected != previous_selection) {
                spectator_age = -1;
            }
            frame_age("Spectator", spectator_age);
            input.reset_camera = ImGui::Button(window->labels.empty() ? "Reset camera" : "Start from selected camera") && !window->mouse_captured;
            ImGui::TextUnformatted("Click view: mouse look | WASD: move | Q/E: down/up | Shift: faster | Esc: release");
            if(status != nullptr && status[0] != '\0') {
                ImGui::TextWrapped("%s", status);
            }
            const auto spectator_size = image_size(window->spectator_width, window->spectator_height);
            if(spectator_age >= 0) {
                ImGui::Image((ImTextureID)(std::intptr_t)window->spectator, spectator_size);
            }
            else {
                ImGui::Dummy(spectator_size);
            }
            if(!window->mouse_captured && ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                glfwSetInputMode(window->handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                glfwGetCursorPos(window->handle, &window->mouse_x, &window->mouse_y);
                window->mouse_captured = true;
            }
            ImGui::EndChild();
            ImGui::End();
            ImGui::Render();
            int width = 0;
            int height = 0;
            glfwGetFramebufferSize(window->handle, &width, &height);
            glViewport(0, 0, width, height);
            glClearColor(0.06f, 0.06f, 0.06f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window->handle);
        }
    }

    Window* create(const char* title, const std::vector<std::string>& labels, int atlas_width, int atlas_height, int columns, int rows, int spectator_width, int spectator_height) {
        require_main_thread();
        if(!glfwInit()) {
            throw std::runtime_error("HyperLux spy could not initialize GLFW; a desktop display is required");
        }
        auto window = std::make_unique<Window>();
        window->labels = labels;
        window->atlas_width = atlas_width;
        window->atlas_height = atlas_height;
        window->columns = columns;
        window->rows = rows;
        window->spectator_width = spectator_width;
        window->spectator_height = spectator_height;
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        window->handle = glfwCreateWindow(labels.empty() ? 960 : 1440, 800, title, nullptr, nullptr);
        glfwDefaultWindowHints();
        if(window->handle == nullptr) {
            throw std::runtime_error("HyperLux spy could not create an OpenGL 3.2 window");
        }
        try {
            ContextScope context(window.get());
            IMGUI_CHECKVERSION();
            window->imgui = ImGui::CreateContext();
            ImGui::GetIO().IniFilename = nullptr;
            ImGui::GetIO().LogFilename = nullptr;
            ImGui::StyleColorsDark();
            glfwSwapInterval(0);
            if(!ImGui_ImplGlfw_InitForOpenGL(window->handle, false)) {
                throw std::runtime_error("HyperLux spy could not initialize its GLFW GUI backend");
            }
            window->glfw_backend = true;
            glfwSetWindowUserPointer(window->handle, window.get());
            glfwSetWindowFocusCallback(window->handle, forward_event<ImGui_ImplGlfw_WindowFocusCallback>);
            glfwSetCursorEnterCallback(window->handle, forward_event<ImGui_ImplGlfw_CursorEnterCallback>);
#if IMGUI_VERSION_NUM >= 18700
            glfwSetCursorPosCallback(window->handle, forward_event<ImGui_ImplGlfw_CursorPosCallback>);
#endif
            glfwSetMouseButtonCallback(window->handle, forward_event<ImGui_ImplGlfw_MouseButtonCallback>);
            glfwSetScrollCallback(window->handle, forward_event<ImGui_ImplGlfw_ScrollCallback>);
            glfwSetKeyCallback(window->handle, forward_event<ImGui_ImplGlfw_KeyCallback>);
            glfwSetCharCallback(window->handle, forward_event<ImGui_ImplGlfw_CharCallback>);
            if(!ImGui_ImplOpenGL3_Init("#version 150")) {
                throw std::runtime_error("HyperLux spy could not initialize its OpenGL GUI backend");
            }
            window->opengl_backend = true;
            if(!labels.empty()) {
                window->atlas = create_texture(atlas_width, atlas_height);
            }
            window->spectator = create_texture(spectator_width, spectator_height);
        }
        catch(...) {
            {
                ContextScope context(window.get());
                release_graphics(window.get());
            }
            glfwDestroyWindow(window->handle);
            throw;
        }
        return window.release();
    }

    Input update(Window* window, double atlas_age, double spectator_age, const char* status) {
        glfwPollEvents();
        ContextScope context(window, false);
        Input input;
        input.camera = window->selected;
        input.closed = glfwWindowShouldClose(window->handle);
        if(input.closed) {
            release_mouse(window);
            return input;
        }
        if(glfwGetKey(window->handle, GLFW_KEY_ESCAPE) == GLFW_PRESS || !glfwGetWindowAttrib(window->handle, GLFW_FOCUSED)) {
            release_mouse(window);
        }
        if(window->mouse_captured) {
            double x = 0;
            double y = 0;
            glfwGetCursorPos(window->handle, &x, &y);
            input.yaw_delta = static_cast<float>(window->mouse_x - x) * 0.002f;
            input.pitch_delta = static_cast<float>(window->mouse_y - y) * 0.002f;
            window->mouse_x = x;
            window->mouse_y = y;
            const float speed = glfwGetKey(window->handle, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 3.0f : 1.0f;
            input.forward = speed * ((glfwGetKey(window->handle, GLFW_KEY_W) == GLFW_PRESS) - (glfwGetKey(window->handle, GLFW_KEY_S) == GLFW_PRESS));
            input.right = speed * ((glfwGetKey(window->handle, GLFW_KEY_D) == GLFW_PRESS) - (glfwGetKey(window->handle, GLFW_KEY_A) == GLFW_PRESS));
            input.up = speed * ((glfwGetKey(window->handle, GLFW_KEY_E) == GLFW_PRESS) - (glfwGetKey(window->handle, GLFW_KEY_Q) == GLFW_PRESS));
        }
        ContextScope graphics(window);
        draw(window, input, atlas_age, spectator_age, status);
        input.camera = window->selected;
        return input;
    }

    void upload_atlas(Window* window, const std::uint32_t* pixels) {
        upload(window, window->atlas, window->atlas_width, window->atlas_height, pixels);
    }

    void upload_spectator(Window* window, const std::uint32_t* pixels) {
        upload(window, window->spectator, window->spectator_width, window->spectator_height, pixels);
    }

    void destroy(Window* window) {
        if(window == nullptr) {
            return;
        }
        {
            ContextScope context(window);
            release_graphics(window);
        }
        glfwDestroyWindow(window->handle);
        delete window;
        // GLFW has no initialization ownership query; terminating it would also destroy the host application's windows.
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
