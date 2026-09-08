#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_PERSIST_H
#include "ant.h"
#include <cstring>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::environments::mujoco::Ant<SPEC>& environment, GROUP& group){
        auto* model = environment.model;
        auto* data = environment.data;
        utils::assert_exit(device, model->nplugin == 0, "Ant persistence requires a model without simulator plugins");
        const int model_size = mj_sizeModel(model);
        std::vector<char> model_data(model_size);
        mj_saveModel(model, nullptr, model_data.data(), model_size);
        save_binary(device, &model_size, 1, group, "model_size");
        save_binary(device, model_data.data(), model_size, group, "model");
        save_binary(device, environment.init_q, SPEC::STATE_DIM_Q, group, "initial_position");
        save_binary(device, environment.init_q_dot, SPEC::STATE_DIM_Q_DOT, group, "initial_velocity");
        save_binary(device, &environment.last_reward, 1, group, "last_reward");
        save_binary(device, &environment.last_terminated, 1, group, "last_terminated");
        save_binary(device, &data->time, 1, group, "time");
        save_binary(device, data->qpos, model->nq * 1, group, "qpos");
        save_binary(device, data->qvel, model->nv * 1, group, "qvel");
        save_binary(device, data->act, model->na * 1, group, "act");
        save_binary(device, data->qacc_warmstart, model->nv * 1, group, "qacc_warmstart");
        save_binary(device, data->ctrl, model->nu * 1, group, "ctrl");
        save_binary(device, data->qfrc_applied, model->nv * 1, group, "qfrc_applied");
        save_binary(device, data->xfrc_applied, model->nbody * 6, group, "xfrc_applied");
        save_binary(device, data->mocap_pos, model->nmocap * 3, group, "mocap_pos");
        save_binary(device, data->mocap_quat, model->nmocap * 4, group, "mocap_quat");
        save_binary(device, data->userdata, model->nuserdata * 1, group, "userdata");
        save_binary(device, data->xpos, model->nbody * 3, group, "xpos");
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::environments::mujoco::Ant<SPEC>& environment, GROUP& group){
        int model_size = 0;
        if(!load_binary(device, &model_size, 1, group, "model_size") || model_size <= 0) return false;
        std::vector<char> model_data(model_size);
        if(!load_binary(device, model_data.data(), model_size, group, "model")) return false;
        mjVFS vfs;
        mj_defaultVFS(&vfs);
        if(mj_makeEmptyFileVFS(&vfs, "checkpoint.mjb", model_size) != 0){
            mj_deleteVFS(&vfs);
            return false;
        }
        std::memcpy(vfs.filedata[mj_findFileVFS(&vfs, "checkpoint.mjb")], model_data.data(), model_size);
        auto* model = mj_loadModel("checkpoint.mjb", &vfs);
        mj_deleteVFS(&vfs);
        if(model == nullptr) return false;
        if(model->nq != SPEC::STATE_DIM_Q || model->nv != SPEC::STATE_DIM_Q_DOT || model->nu != SPEC::ACTION_DIM || model->nplugin != 0){
            mj_deleteModel(model);
            return false;
        }
        auto* data = mj_makeData(model);
        if(data == nullptr){ mj_deleteModel(model); return false; }
        mj_deleteData(environment.data);
        mj_deleteModel(environment.model);
        environment.model = model;
        environment.data = data;
        environment.torso_id = mj_name2id(model, mjOBJ_XBODY, "torso");
        bool success = load_binary(device, environment.init_q, SPEC::STATE_DIM_Q, group, "initial_position");
        success &= load_binary(device, environment.init_q_dot, SPEC::STATE_DIM_Q_DOT, group, "initial_velocity");
        success &= load_binary(device, &environment.last_reward, 1, group, "last_reward");
        success &= load_binary(device, &environment.last_terminated, 1, group, "last_terminated");
        success &= load_binary(device, &data->time, 1, group, "time");
        success &= load_binary(device, data->qpos, model->nq * 1, group, "qpos");
        success &= load_binary(device, data->qvel, model->nv * 1, group, "qvel");
        success &= load_binary(device, data->act, model->na * 1, group, "act");
        success &= load_binary(device, data->ctrl, model->nu * 1, group, "ctrl");
        success &= load_binary(device, data->qfrc_applied, model->nv * 1, group, "qfrc_applied");
        success &= load_binary(device, data->xfrc_applied, model->nbody * 6, group, "xfrc_applied");
        success &= load_binary(device, data->mocap_pos, model->nmocap * 3, group, "mocap_pos");
        success &= load_binary(device, data->mocap_quat, model->nmocap * 4, group, "mocap_quat");
        success &= load_binary(device, data->userdata, model->nuserdata * 1, group, "userdata");
        mj_forward(model, data);
        success &= load_binary(device, data->qacc_warmstart, model->nv, group, "qacc_warmstart");
        success &= load_binary(device, data->xpos, model->nbody * 3, group, "xpos");
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
