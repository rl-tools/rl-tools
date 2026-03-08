#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 0

#include <rl_tools/operations/cpu_mux.h>

#include "environment/environment.h"
#include "environment/operations_cpu.h"

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>

namespace rlt = rl_tools;

struct InputState {
    bool forward = false, backward = false, left = false, right = false;
    bool up = false, down = false;
    double mouse_x = 0, mouse_y = 0;
    double last_mouse_x = 0, last_mouse_y = 0;
    bool first_mouse = true;
    bool cursor_captured = true;
    float yaw = 0;
    float pitch = 0;
};

static InputState g_input;

static void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return;
    }
    bool pressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
    if (key == GLFW_KEY_W) g_input.forward = pressed;
    if (key == GLFW_KEY_S) g_input.backward = pressed;
    if (key == GLFW_KEY_A) g_input.left = pressed;
    if (key == GLFW_KEY_D) g_input.right = pressed;
    if (key == GLFW_KEY_SPACE) g_input.up = pressed;
    if (key == GLFW_KEY_LEFT_SHIFT) g_input.down = pressed;
}

static void cursor_pos_callback(GLFWwindow* /*window*/, double xpos, double ypos) {
    if (!g_input.cursor_captured) return;
    if (g_input.first_mouse) {
        g_input.last_mouse_x = xpos;
        g_input.last_mouse_y = ypos;
        g_input.first_mouse = false;
        return;
    }
    float dx = static_cast<float>(xpos - g_input.last_mouse_x);
    float dy = static_cast<float>(ypos - g_input.last_mouse_y);
    g_input.last_mouse_x = xpos;
    g_input.last_mouse_y = ypos;

    constexpr float sensitivity = 0.002f;
    g_input.yaw += dx * sensitivity;
    g_input.pitch -= dy * sensitivity;
    constexpr float max_pitch = static_cast<float>(M_PI) * 0.49f;
    if (g_input.pitch > max_pitch) g_input.pitch = max_pitch;
    if (g_input.pitch < -max_pitch) g_input.pitch = -max_pitch;
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        if (g_input.cursor_captured) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            g_input.cursor_captured = false;
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            g_input.cursor_captured = true;
            g_input.first_mouse = true;
        }
    }
}

int main(int argc, char** argv) {
    using T = float;
    using TI = unsigned int;
    constexpr TI CAM_WIDTH = 640;
    constexpr TI CAM_HEIGHT = 480;
    constexpr TI NUM_ENVS = 1;
    using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, NUM_ENVS, CAM_WIDTH, CAM_HEIGHT, 64>;
    using DEVICE = rlt::devices::DEVICE_FACTORY<>;

    DEVICE device;
    rlt::init(device);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scene.glb>" << std::endl;
        return 1;
    }
    rlt::rl::environments::raytracing_example::Environment<SPEC> env;
    env.scene_path = argv[1];

    rlt::malloc(device, env);
    rlt::init(device, env);

    rlt::rl::environments::raytracing_example::State<SPEC> state{};
    if (env.num_indoor_initial_states > 0) {
        state = env.indoor_initial_states[0];
    }

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(CAM_WIDTH, CAM_HEIGHT, "RLtools Interactive Viewer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    g_input.yaw = state.yaw;

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, CAM_WIDTH, CAM_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    std::vector<uint32_t> pixels(CAM_WIDTH * CAM_HEIGHT);

    constexpr float MOVE_SPEED = 3.0f;
    auto last_time = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window)) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        if (dt > 0.1f) dt = 0.1f;

        glfwPollEvents();

        float cos_yaw = std::cos(g_input.yaw);
        float sin_yaw = std::sin(g_input.yaw);
        float dx = 0, dz = 0, dy = 0;
        if (g_input.forward)  { dx += cos_yaw; dz += sin_yaw; }
        if (g_input.backward) { dx -= cos_yaw; dz -= sin_yaw; }
        if (g_input.left)     { dx += sin_yaw; dz -= cos_yaw; }
        if (g_input.right)    { dx -= sin_yaw; dz += cos_yaw; }
        if (g_input.up)       { dy += 1; }
        if (g_input.down)     { dy -= 1; }
        float move_len = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (move_len > 0) {
            float speed = MOVE_SPEED * dt / move_len;
            state.position[0] += dx * speed;
            state.position[1] += dy * speed;
            state.position[2] += dz * speed;
        }
        state.yaw = g_input.yaw;

        owl::vec3f eye(state.position[0], state.position[1] + env.eye_height, state.position[2]);
        owl::vec3f look_at(
            eye.x + std::cos(g_input.yaw) * std::cos(g_input.pitch),
            eye.y + std::sin(g_input.pitch),
            eye.z + std::sin(g_input.yaw) * std::cos(g_input.pitch)
        );
        T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
        rlt::CameraData camera = rlt::make_camera_data(eye, look_at, owl::vec3f(0.f, 1.f, 0.f), SPEC::RAYTRACING_SPEC::COS_FOVY, aspect);

        rlt::set_cameras(device, *env.renderer, &camera, static_cast<TI>(1));
        rlt::render_rgb_only(device, *env.renderer);
        rlt::read_frame_buffer(device, *env.renderer, pixels.data(), static_cast<TI>(pixels.size()));

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, CAM_WIDTH, CAM_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 1); glVertex2f(-1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1, -1);
        glTexCoord2f(1, 0); glVertex2f( 1,  1);
        glTexCoord2f(0, 0); glVertex2f(-1,  1);
        glEnd();

        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();

    rlt::free(device, env);
    return 0;
}
