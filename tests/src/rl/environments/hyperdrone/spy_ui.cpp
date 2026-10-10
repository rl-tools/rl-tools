#include <rl_tools/rl/environments/hyperdrone/spy/ui.h>
#include <metra/metra.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>

#ifdef RL_TOOLS_SPY_TEST_X11
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#endif

namespace {
    namespace ui = rl_tools::rl::environments::hyperdrone::spy::ui;

    ui::Input frame(ui::Window* window) {
        ui::update(window, 0.05, 0.05, "GUI integration test");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return ui::update(window, 0.05, 0.05, "GUI integration test");
    }

#ifdef RL_TOOLS_SPY_TEST_X11
    ::Window find_spy_window(Display* display, ::Window parent_window) {
        ::Window root = 0;
        ::Window parent = 0;
        ::Window* children = nullptr;
        unsigned int count = 0;
        XQueryTree(display, parent_window, &root, &parent, &children, &count);
        ::Window found = 0;
        for(unsigned int i = 0; i < count; i++) {
            char* title = nullptr;
            if(XFetchName(display, children[i], &title) && title != nullptr) {
                if(std::strcmp(title, "HyperDrone spy") == 0) {
                    found = children[i];
                }
                XFree(title);
            }
            if(found == 0) {
                found = find_spy_window(display, children[i]);
            }
            if(found != 0) {
                break;
            }
        }
        XFree(children);
        return found;
    }

    void button(Display* display, ::Window window, bool pressed, int x, int y) {
        XEvent event{};
        event.xbutton.type = pressed ? ButtonPress : ButtonRelease;
        event.xbutton.display = display;
        event.xbutton.window = window;
        event.xbutton.root = DefaultRootWindow(display);
        event.xbutton.button = Button1;
        event.xbutton.same_screen = True;
        event.xbutton.x = x;
        event.xbutton.y = y;
        XSendEvent(display, window, False, pressed ? ButtonPressMask : ButtonReleaseMask, &event);
        XSync(display, False);
    }

    void key(Display* display, ::Window window, KeySym symbol, bool pressed) {
        XEvent event{};
        event.xkey.type = pressed ? KeyPress : KeyRelease;
        event.xkey.display = display;
        event.xkey.window = window;
        event.xkey.root = DefaultRootWindow(display);
        event.xkey.keycode = XKeysymToKeycode(display, symbol);
        event.xkey.same_screen = True;
        XSendEvent(display, window, False, pressed ? KeyPressMask : KeyReleaseMask, &event);
        XSync(display, False);
    }

    ui::Input click(Display* display, ::Window native, ui::Window* window, int x, int y) {
        XWarpPointer(display, None, native, 0, 0, 0, 0, x, y);
        XSync(display, False);
        frame(window);
        button(display, native, true, x, y);
        const auto input = frame(window);
        button(display, native, false, x, y);
        frame(window);
        return input;
    }
#endif
}

TEST(HyperDroneSpyUI, RealWindowUploadsInputAndHostContext) {
#ifdef RL_TOOLS_SPY_TEST_X11
    if(std::getenv("DISPLAY") == nullptr) {
        GTEST_SKIP() << "An X11 display is required; run under xvfb-run for a virtual display";
    }
#ifdef GLFW_PLATFORM
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif
#elif defined(__linux__)
    if(std::getenv("DISPLAY") == nullptr && std::getenv("WAYLAND_DISPLAY") == nullptr) {
        GTEST_SKIP() << "A desktop display is required; run under xvfb-run for a virtual display";
    }
#endif
    ASSERT_TRUE(glfwInit());
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> host(glfwCreateWindow(64, 64, "Host context", nullptr, nullptr), glfwDestroyWindow);
    ASSERT_NE(host, nullptr);
    glfwMakeContextCurrent(host.get());
    std::unique_ptr<ImGuiContext, decltype(&ImGui::DestroyContext)> host_imgui(ImGui::CreateContext(), ImGui::DestroyContext);
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);

    const auto started = std::chrono::steady_clock::now();
    std::unique_ptr<ui::Window, decltype(&ui::destroy)> window(ui::create(2, 1, 2, 2, 8, 8, 4, 2, 8, 6), ui::destroy);
    EXPECT_EQ(glfwGetCurrentContext(), host.get());
    EXPECT_EQ(ImGui::GetCurrentContext(), host_imgui.get());
    EXPECT_THROW(ui::create(1, 1, 1, 1, 8, 8, 1, 1, 8, 6), std::runtime_error);

    std::array<std::uint32_t, 64> atlas;
    atlas.fill(0xff0000ffu);
    std::array<std::uint32_t, 48> spectator;
    spectator.fill(0xffff0000u);
    ui::upload_atlas(window.get(), atlas.data());
    ui::upload_spectator(window.get(), spectator.data());
    EXPECT_FALSE(frame(window.get()).closed);
    EXPECT_EQ(glfwGetCurrentContext(), host.get());
    EXPECT_EQ(ImGui::GetCurrentContext(), host_imgui.get());
    std::array<float, 4> clear_color{};
    glGetFloatv(GL_COLOR_CLEAR_VALUE, clear_color.data());
    EXPECT_EQ(clear_color, (std::array<float, 4>{0.25f, 0.5f, 0.75f, 1.0f}));

#ifdef RL_TOOLS_SPY_TEST_X11
    std::unique_ptr<Display, decltype(&XCloseDisplay)> display(XOpenDisplay(nullptr), XCloseDisplay);
    ASSERT_NE(display, nullptr);
    const auto native = find_spy_window(display.get(), DefaultRootWindow(display.get()));
    ASSERT_NE(native, 0u);
    XSetInputFocus(display.get(), native, RevertToParent, CurrentTime);
    XSync(display.get(), False);
    frame(window.get());

    XWindowAttributes attributes{};
    ASSERT_TRUE(XGetWindowAttributes(display.get(), native, &attributes));
    XImage* pixels = XGetImage(display.get(), native, 0, 0, attributes.width, attributes.height, AllPlanes, ZPixmap);
    ASSERT_NE(pixels, nullptr);
    size_t red_pixels = 0;
    size_t blue_pixels = 0;
    for(int y = 0; y < pixels->height; y++) {
        for(int x = 0; x < pixels->width; x++) {
            const auto pixel = XGetPixel(pixels, x, y);
            red_pixels += (pixel & (pixels->red_mask | pixels->green_mask | pixels->blue_mask)) == pixels->red_mask;
            blue_pixels += (pixel & (pixels->red_mask | pixels->green_mask | pixels->blue_mask)) == pixels->blue_mask;
        }
    }
    XDestroyImage(pixels);
    EXPECT_GT(red_pixels, 100000u);
    EXPECT_GT(blue_pixels, 100000u);

    EXPECT_EQ(click(display.get(), native, window.get(), 600, 530).camera, 7u);
    click(display.get(), native, window.get(), 900, 400);
    XWarpPointer(display.get(), None, native, 0, 0, 0, 0, 780, 440);
    XSync(display.get(), False);
    float yaw_delta = 0;
    float pitch_delta = 0;
    for(int i = 0; i < 4; i++) {
        const auto movement = ui::update(window.get(), 0.05, 0.05, "GUI integration test");
        yaw_delta += movement.yaw_delta;
        pitch_delta += movement.pitch_delta;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_LT(yaw_delta, 0);
    EXPECT_LT(pitch_delta, 0);
    key(display.get(), native, XK_w, true);
    EXPECT_EQ(frame(window.get()).forward, 1.0f);
    key(display.get(), native, XK_Shift_L, true);
    key(display.get(), native, XK_d, true);
    key(display.get(), native, XK_e, true);
    const auto accelerated = frame(window.get());
    EXPECT_EQ(accelerated.forward, 3.0f);
    EXPECT_EQ(accelerated.right, 3.0f);
    EXPECT_EQ(accelerated.up, 3.0f);
    key(display.get(), native, XK_Escape, true);
    EXPECT_EQ(frame(window.get()).forward, 0.0f);
    for(const auto symbol: {XK_w, XK_Shift_L, XK_d, XK_e, XK_Escape}) {
        key(display.get(), native, symbol, false);
    }
    frame(window.get());
    click(display.get(), native, window.get(), 900, 400);
    XSetInputFocus(display.get(), DefaultRootWindow(display.get()), RevertToParent, CurrentTime);
    XSync(display.get(), False);
    frame(window.get());
    XSetInputFocus(display.get(), native, RevertToParent, CurrentTime);
    XSync(display.get(), False);
    key(display.get(), native, XK_w, true);
    EXPECT_EQ(frame(window.get()).forward, 0.0f);
    key(display.get(), native, XK_w, false);

    XEvent close{};
    close.xclient.type = ClientMessage;
    close.xclient.window = native;
    close.xclient.message_type = XInternAtom(display.get(), "WM_PROTOCOLS", False);
    close.xclient.format = 32;
    close.xclient.data.l[0] = XInternAtom(display.get(), "WM_DELETE_WINDOW", False);
    XSendEvent(display.get(), native, False, NoEventMask, &close);
    XSync(display.get(), False);
    EXPECT_TRUE(frame(window.get()).closed);
#endif

    window.reset();
    EXPECT_EQ(glfwGetCurrentContext(), host.get());
    EXPECT_EQ(ImGui::GetCurrentContext(), host_imgui.get());
    EXPECT_EQ(glfwWindowShouldClose(host.get()), GLFW_FALSE);
    window.reset(ui::create(1, 1, 1, 1, 8, 8, 1, 1, 8, 6));
    EXPECT_FALSE(frame(window.get()).closed);
    window.reset();
    EXPECT_EQ(glfwGetCurrentContext(), host.get());
    EXPECT_EQ(ImGui::GetCurrentContext(), host_imgui.get());
    EXPECT_EQ(glGetError(), static_cast<GLenum>(GL_NO_ERROR));
    metra::log("hyperdrone/spy/ui/smoke_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
}
