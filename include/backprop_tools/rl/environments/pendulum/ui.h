#include <QApplication>
#include <QPushButton>
#include <QPainter>
#include <QTimer>
#include <thread>
#include <atomic>
//#include <gtk/gtk.h>
//#include <cairo.h>
//#include <glib.h>
#include <iostream>
#include <mutex>
#include "pendulum.h"


namespace backprop_tools::rl::environments::pendulum {
    template <typename T>
    class PendulumWidget : public QWidget
    {
    public:
        T angle = 0;
    protected:
        void paintEvent(QPaintEvent *event)
        {
            QPainter painter(this);
            painter.setPen(QPen(Qt::black, 12, Qt::SolidLine, Qt::RoundCap));
            T radius = 100;
            T start_x = radius * 1.2;
            T start_y = radius * 1.2;
            T end_x = start_x + radius * math::sin(angle + M_PI);
            T end_y = start_y + radius * math::cos(angle + M_PI);
            painter.drawLine(start_x, start_y, end_x, end_y);
        }
    };


    template<typename T, auto FPS=60>
    struct UI {
        char* argv[1] = {"pendulum_render"};
        typename DEVICE::index_t argc = 1;
        QApplication* app = nullptr;
        std::thread t;
        std::atomic<bool> running = true;
        std::atomic<T> angle = 0;
        PendulumWidget<T>* window = nullptr;
        UI(){
            t = std::thread(&UI::run, this);
        };
        void run() {
            this->app = new QApplication(this->argc, this->argv);
            this->window = new PendulumWidget<T>();
            this->window->resize(320, 240);
            this->window->show();
            this->window->setWindowTitle("backprop_tools::rl::environments::pendulum");
//            QPushButton hello("Hello world!");
//            hello.setParent(&window);
            QTimer timer;
            QObject::connect(&timer, &QTimer::timeout, [&]() {
                this->timer_callback();
            });
            timer.start(1000/FPS);
            this->app->exec();
            delete this->window;
            delete this->app;
        }
        void timer_callback() {
            this->window->angle = this->angle;
            this->window->update();
//            std::cout << "timer_callback" << std::endl;
            if(!this->running){
                this->window->close();
            }
        }
        ~UI() {
            this->running = false;
            this->t.join();
        }
    };
}


namespace backprop_tools{
    template <typename T>
    void set_state(rl::environments::pendulum::UI<T>& ui, const rl::environments::pendulum::State<T>& state){
        ui.angle = state.theta;
    }
}
