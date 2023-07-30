#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_CAR_UI_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_CAR_UI_H

#include "car.h"

#include <gtk/gtk.h>
#include <thread>
#include <chrono>

namespace backprop_tools::rl::environments::car{
    namespace ui{
        template<typename T_T, typename T_TI, T_TI T_SIZE, T_TI T_PLAYBACK_SPEED>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
        static constexpr TI SIZE = T_SIZE;
            static constexpr T PLAYBACK_SPEED = T_PLAYBACK_SPEED/100.0;
            static constexpr T ACTION_INDICATOR_SIZE = SIZE * 1.0/10.0;
        };
    }

    template<typename T_SPEC>
    struct UI {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        std::chrono::time_point<std::chrono::high_resolution_clock> last_render_time;
        rl::environments::car::State<typename SPEC::T, typename SPEC::TI> state;
        rl::environments::car::Parameters<typename SPEC::T> parameters;
        MatrixDynamic<matrix::Specification<T, TI, 1, 2>> action;
        GtkWidget *window;
        GtkWidget *canvas;
    };
}


namespace backprop_tools::rl::environments::car::ui{
    template <typename T>
    void R(T alpha, T result[2][2]) {
        result[0][0] = cos(alpha);
        result[0][1] = -sin(alpha);
        result[1][0] = sin(alpha);
        result[1][1] = cos(alpha);
    }
    template <typename T, typename TI>
    void global_position(T x_offset, T y_offset, T result[2], T x, T y, T alpha, const rl::environments::car::State<T, TI>& s, T scale) {
        T rotation[2][2];
        R(alpha, rotation);
        result[0] = x_offset + s.x * scale + (rotation[0][0] * x + rotation[0][1] * y) * scale;
        result[1] = y_offset - s.y * scale - (rotation[1][0] * x + rotation[1][1] * y) * scale;
    }
    template <typename T, typename TI>
    void draw_wheel(cairo_t* cr, T x_offset, T y_offset, T x, T y, T size, T width, T delta, const rl::environments::car::State<T, TI>& s, T scale){
        T start[2] = { -size / 2 * cos(delta), -size / 2 * sin(delta) };
        T finish[2] = { size / 2 * cos(delta), size / 2 * sin(delta) };

        T position[2];
        global_position(x_offset, y_offset, position, x + start[0], y + start[1], s.mu, s, scale);
        cairo_move_to(cr, position[0], position[1]);
        global_position(x_offset, y_offset, position, x + finish[0], y + finish[1], s.mu, s, scale);
        cairo_line_to(cr, position[0], position[1]);
        cairo_set_source_rgb(cr, 1, 0, 0);
        cairo_set_line_width(cr, width*scale);
        cairo_stroke(cr);
    }

    template <typename SPEC>
    static gboolean draw_callback(GtkWidget *c, cairo_t *cr, gpointer data){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        UI<SPEC>& ui = *(UI<SPEC>*)data;
        auto& p = ui.parameters;
        auto& s = ui.state;
        auto& a = ui.action;
        T h = gtk_widget_get_allocated_height(c);
        T w = gtk_widget_get_allocated_width(c);
        T x_offset = w / 2;
        T y_offset = h / 2;

        cairo_rectangle(cr, 0, 0, w, h);
        cairo_set_source_rgb(cr, 1, 1, 1);
        cairo_fill(cr);

        T scale = 500;
        T L = p.lr + p.lf;
        T W = L / 2.5;

        T position[2];
        global_position<T, TI>(x_offset, y_offset, position, -p.lr, 0, s.mu, s, scale);
        cairo_move_to(cr, position[0], position[1]);
        global_position<T, TI>(x_offset, y_offset, position, p.lf, 0, s.mu, s, scale);
        cairo_line_to(cr, position[0], position[1]);
        cairo_set_source_rgb(cr, 0, 0, 0);
        cairo_set_line_width(cr, W*scale);
        cairo_stroke(cr);

        T wheel_size = L / 2;
        T wheel_width = L / 10;

        draw_wheel<T, TI>(cr, x_offset, y_offset, -p.lr, W/2 + wheel_width/2, wheel_size, wheel_width, 0, s, scale);
        draw_wheel<T, TI>(cr, x_offset, y_offset, -p.lr, -W/2 - wheel_width/2, wheel_size, wheel_width, 0, s, scale);
        draw_wheel<T, TI>(cr, x_offset, y_offset, p.lf, W/2 + wheel_width/2, wheel_size, wheel_width, get(a, 0, 1), s, scale);
        draw_wheel<T, TI>(cr, x_offset, y_offset, p.lf, -W/2 - wheel_width/2, wheel_size, wheel_width, get(a, 0, 1), s, scale);
        return FALSE;
    }
}


namespace backprop_tools{
    template <typename DEVICE, typename ENV_SPEC, typename SPEC>
    void render(DEVICE& device, const rl::environments::Car<ENV_SPEC>& env, rl::environments::car::UI<SPEC>& ui){
        auto now = std::chrono::high_resolution_clock::now();
        auto interval = (typename DEVICE::index_t)(1000.0 * env.parameters.dt / SPEC::PLAYBACK_SPEED);
        auto next_render_time = ui.last_render_time + std::chrono::milliseconds(interval);
        if(now < next_render_time){
            auto diff = next_render_time - now;
            std::this_thread::sleep_for(diff);
        }

        using T = typename SPEC::T;
        while(gtk_events_pending()){
            gtk_main_iteration_do(FALSE);
        }
        gtk_widget_queue_draw(ui.canvas);


        now = std::chrono::high_resolution_clock::now();
        ui.last_render_time = now;
    }
    template <typename DEVICE, typename ENV_SPEC, typename SPEC>
    void init(DEVICE& device, const rl::environments::Car<ENV_SPEC>& env, rl::environments::car::UI<SPEC>& ui){
        malloc(device, ui.action);
        ui.parameters = env.parameters;
        gtk_init(nullptr, nullptr);
        ui.window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        g_signal_connect(ui.window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
        ui.canvas = gtk_drawing_area_new();
        gtk_container_add(GTK_CONTAINER(ui.window), ui.canvas);
        g_signal_connect(G_OBJECT(ui.canvas), "draw", G_CALLBACK(rl::environments::car::ui::draw_callback<SPEC>), &ui);
        gtk_widget_show_all(ui.window);
        render(device, env, ui);
        ui.last_render_time = std::chrono::high_resolution_clock::now();
    }
    template <typename DEVICE, typename ENV_SPEC, typename SPEC, typename T, typename TI>
    void set_state(DEVICE& device, const rl::environments::Car<ENV_SPEC>& env, rl::environments::car::UI<SPEC>& ui, const rl::environments::car::State<T, TI>& state){
        ui.state = state;
    }
    template <typename DEVICE, typename ENV_SPEC, typename SPEC, typename ACTION_SPEC>
    void set_action(DEVICE& device, const rl::environments::Car<ENV_SPEC>& env, rl::environments::car::UI<SPEC>& ui, const Matrix<ACTION_SPEC>& action){
        using ENVIRONMENT = rl::environments::Car<ENV_SPEC>;
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == ENVIRONMENT::ACTION_DIM);
        copy(device, device, ui.action, action);
    }
}

#endif
