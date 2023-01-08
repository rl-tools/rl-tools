namespace layer_in_c::utils{
    template <typename DEV_SPEC, typename T>
    void assert_exit(const devices::Dummy<DEV_SPEC>& dev, bool condition, T message){
        if(!condition){
            logging::text(typename DEV_SPEC::LOGGING(), message);
        }
    }
}
