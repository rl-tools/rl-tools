#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_ACROBOT_UI_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_ACROBOT_UI_H

#include "acrobot.h"

#include <gtk/gtk.h>
#include <thread>
#include <chrono>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::acrobot{
    namespace ui{
        template<typename T_T, typename T_TI, typename T_ENVIRONMENT, T_TI T_SIZE, T_TI T_PLAYBACK_SPEED, bool T_BLOCK = true>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            using ENVIRONMENT = T_ENVIRONMENT;
            static constexpr TI SIZE = T_SIZE;
            static constexpr T PLAYBACK_SPEED = T_PLAYBACK_SPEED/100.0;
            static constexpr T ACTION_INDICATOR_SIZE = SIZE * 1.0/10.0;
            static constexpr T UI_SCALE = 500;
            static constexpr bool BLOCK = T_BLOCK;
        };
    }

    template<typename T_SPEC>
    struct UI {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        std::chrono::time_point<std::chrono::high_resolution_clock> last_render_time;
        typename SPEC::ENVIRONMENT::State state;
        typename SPEC::ENVIRONMENT::PARAMETERS parameters;
        MatrixDynamic<matrix::Specification<T, TI, 1, SPEC::ENVIRONMENT::ACTION_DIM>> action;
        GtkWidget *window;
        GtkWidget *canvas;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::acrobot::ui{
    template <typename T>
    void R(T alpha, T result[2][2]) {
        result[0][0] = cos(alpha);
        result[0][1] = -sin(alpha);
        result[1][0] = sin(alpha);
        result[1][1] = cos(alpha);
    }

    template <typename SPEC>
    static gboolean draw_callback(GtkWidget *c, cairo_t *cr, gpointer data){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        UI<SPEC>& ui = *(UI<SPEC>*)data;
//        auto& p = ui.parameters;
        auto& s = ui.state;
        auto& a = ui.action;
        T h = gtk_widget_get_allocated_height(c);
        T w = gtk_widget_get_allocated_width(c);
        T x_offset = w / 2;
        T y_offset = h / 2;

        cairo_rectangle(cr, 0, 0, w, h);
        cairo_set_source_rgb(cr, 1, 1, 1);
        cairo_fill(cr);
        T second_joint_x =  sin(s.theta_0) * SPEC::ENVIRONMENT::PARAMETERS::LINK_LENGTH_1;
        T second_joint_y = -cos(s.theta_0) * SPEC::ENVIRONMENT::PARAMETERS::LINK_LENGTH_1;
        T end_x = second_joint_x + sin(s.theta_0 + s.theta_1) * SPEC::ENVIRONMENT::PARAMETERS::LINK_LENGTH_2;
        T end_y = second_joint_y - cos(s.theta_0 + s.theta_1) * SPEC::ENVIRONMENT::PARAMETERS::LINK_LENGTH_2;


        T scale = SPEC::SIZE / 3.0;


        cairo_set_source_rgb(cr, 0xb8/255.0, 0xb8/255.0, 0xb8/255.0);
        cairo_move_to(cr, x_offset, y_offset);
        cairo_line_to(cr, x_offset + second_joint_x*scale, y_offset - second_joint_y*scale);
        cairo_set_line_width(cr, scale * 0.1);
        cairo_stroke(cr);
        cairo_move_to(cr, x_offset + second_joint_x*scale, y_offset - second_joint_y*scale);
        cairo_line_to(cr, x_offset + end_x*scale, y_offset - end_y*scale);
        cairo_set_line_width(cr, scale * 0.1);
        cairo_stroke(cr);
        cairo_arc(cr, x_offset, y_offset, scale * 0.1, 0, 2*M_PI);
        cairo_arc(cr, x_offset + second_joint_x * scale, y_offset - second_joint_y * scale, scale * 0.1, 0, 2*M_PI);
        cairo_set_source_rgb(cr, 0x7D/255.0, 0xB9/255.0, 0xB6/255.0);
        cairo_fill(cr);
        return FALSE;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename ENV_SPEC, typename SPEC>
    void render(DEVICE& device, const rl::environments::Acrobot<ENV_SPEC>& env, rl::environments::acrobot::UI<SPEC>& ui){
        auto now = std::chrono::high_resolution_clock::now();
        auto interval = (typename DEVICE::index_t)(1000.0 * ENV_SPEC::PARAMETERS::DT / SPEC::PLAYBACK_SPEED);
        auto next_render_time = ui.last_render_time + std::chrono::milliseconds(interval);
        if(SPEC::BLOCK && now < next_render_time){
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
    void init(DEVICE& device, const rl::environments::Acrobot<ENV_SPEC>& env, rl::environments::acrobot::UI<SPEC>& ui){
        malloc(device, ui.action);
//        ui.parameters = env.parameters;
        gtk_init(nullptr, nullptr);
        ui.window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        gtk_window_set_default_size(GTK_WINDOW(ui.window), 1000, 1000);
        g_signal_connect(ui.window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
        ui.canvas = gtk_drawing_area_new();
        gtk_container_add(GTK_CONTAINER(ui.window), ui.canvas);
        g_signal_connect(G_OBJECT(ui.canvas), "draw", G_CALLBACK(rl::environments::acrobot::ui::draw_callback<SPEC>), &ui);
        gtk_widget_show_all(ui.window);
        render(device, env, ui);
        ui.last_render_time = std::chrono::high_resolution_clock::now();
    }
    template <typename DEVICE, typename ENV_SPEC, typename SPEC, typename T, typename TI>
    void set_state(DEVICE& device, const rl::environments::Acrobot<ENV_SPEC>& env, rl::environments::acrobot::UI<SPEC>& ui, const rl::environments::acrobot::State<T, TI>& state){
        ui.state = state;
    }
    template <typename DEVICE, typename ENV_SPEC, typename SPEC, typename ACTION_SPEC>
    void set_action(DEVICE& device, const rl::environments::Acrobot<ENV_SPEC>& env, rl::environments::acrobot::UI<SPEC>& ui, const Matrix<ACTION_SPEC>& action){
        using ENVIRONMENT = rl::environments::Acrobot<ENV_SPEC>;
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == ENVIRONMENT::ACTION_DIM);
        copy(device, device, action, ui.action);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
