#include "variations.h"
#include "assessment.h"


template <typename DEVICE, typename ENVIRONMENT, typename ACTOR>
void full_assessment(DEVICE& device, ACTOR& actor, typename ENVIRONMENT::PARAMETERS nominal_parameters, bool log = false){
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 10);

    using VARIATION_SPEC = variations::Specification<typename DEVICE::index_t, ENVIRONMENT, decltype(rng)>;
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/init.variation_0/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/init.variation_0/steps", std::get<1>(stats));
        }
        else{
            std::cout << "init.variation_0 (simple): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/init.variation_1/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/init.variation_1/steps", std::get<1>(stats));
        }
        else{
            std::cout << "init.variation_1 (orientation small): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/init.variation_2/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/init.variation_2/steps", std::get<1>(stats));
        }
        else{
            std::cout << "init.variation_2 (orientation big): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_3<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/init.variation_3/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/init.variation_3/steps", std::get<1>(stats));
        }
        else{
            std::cout << "init.variation_3 (orientation bigger): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_4<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/init.variation_4/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/init.variation_4/steps", std::get<1>(stats));
        }
        else{
            std::cout << "init.variation_4 (orientation biggest): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_5<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/init.variation_5/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/init.variation_5/steps", std::get<1>(stats));
        }
        else{
            std::cout << "init.variation_5 (orientation all around): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::position::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.position.variation_0:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.position.variation_0:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.position.variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::position::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.position.variation_1:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.position.variation_1:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.position.variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::position::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.position.variation_2:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.position.variation_2:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.position.variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::orientation::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.orientation.variation_0:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.orientation.variation_0:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.orientation.variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::orientation::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.orientation.variation_1:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.orientation.variation_1:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.orientation.variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::orientation::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.orientation.variation_2:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.orientation.variation_2:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.orientation.variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::linear_velocity::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.linear_velocity.variation_0:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.linear_velocity.variation_0:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.linear_velocity.variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::linear_velocity::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.linear_velocity.variation_1:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.linear_velocity.variation_1:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.linear_velocity.variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::linear_velocity::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.linear_velocity.variation_2:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.linear_velocity.variation_2:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.linear_velocity.variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::angular_velocity::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.angular_velocity.variation_0:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.angular_velocity.variation_0:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.angular_velocity.variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::angular_velocity::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.angular_velocity.variation_1:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.angular_velocity.variation_1:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.angular_velocity.variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::angular_velocity::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.angular_velocity.variation_2:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/observation_noise.angular_velocity.variation_2:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "observation_noise.angular_velocity.variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::action_noise::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/action_noise.variation_0:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/action_noise.variation_0:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "action_noise.variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::action_noise::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/action_noise.variation_1:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/action_noise.variation_1:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "action_noise.variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::action_noise::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/action_noise.variation_2:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/action_noise.variation_2:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "action_noise.variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::inertia::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::inertia::variation_0:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::inertia::variation_0:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::inertia::variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::inertia::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::inertia::variation_1:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::inertia::variation_1:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::inertia::variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::inertia::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::inertia::variation_2:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::inertia::variation_2:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::inertia::variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::mass::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::mass::variation_0:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::mass::variation_0:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::mass::variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::mass::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::mass::variation_1:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::mass::variation_1:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::mass::variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::mass::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::mass::variation_2:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::mass::variation_2:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::mass::variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::max_rpm::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::max_rpm::variation_0:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::max_rpm::variation_0:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::max_rpm::variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::max_rpm::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::max_rpm::variation_1:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::max_rpm::variation_1:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::max_rpm::variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::max_rpm::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::max_rpm::variation_2:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::max_rpm::variation_2:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::max_rpm::variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::rpm_time_constant::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::rpm_time_constant::variation_0:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::rpm_time_constant::variation_0:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::rpm_time_constant::variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::rpm_time_constant::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::rpm_time_constant::variation_1:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::rpm_time_constant::variation_1:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::rpm_time_constant::variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::rpm_time_constant::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        if(log){
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::rpm_time_constant::variation_2:/return", std::get<0>(stats));
            bpt::add_scalar(device, device.logger, "assessment/variations::dynamics::rpm_time_constant::variation_2:/steps", std::get<1>(stats));
        }
        else{
            std::cout << "variations::dynamics::rpm_time_constant::variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
        }
    }
}
