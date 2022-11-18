
template <typename T, int STATE_DIM, int ACTION_DIM, int CAPACITY>
struct ReplayBuffer {
    T states[CAPACITY][STATE_DIM];
    T actions[CAPACITY][ACTION_DIM];
    T rewards[CAPACITY];
    T next_states[CAPACITY][STATE_DIM];
    bool terminated[CAPACITY];
    bool truncated[CAPACITY];
    uint32_t position = 0;
    bool full;
};


template <typename T, int STATE_DIM, int ACTION_DIM, int CAPACITY>
void add(ReplayBuffer<T, STATE_DIM, ACTION_DIM, CAPACITY>& buffer, const T state[STATE_DIM], const T action[ACTION_DIM], const T reward, const T next_state[STATE_DIM], const bool terminated, const bool truncated) {
    for(int i = 0; i < STATE_DIM; i++) {
        buffer.states[buffer.position][i] = state[i];
        buffer.next_states[buffer.position][i] = next_state[i];
    }
    for(int i = 0; i < ACTION_DIM; i++) {
        buffer.actions[buffer.position][i] = action[i];
    }
    buffer.rewards[buffer.position] = reward;
    buffer.terminated[buffer.position] = terminated;
    buffer.truncated[buffer.position] = truncated;
    buffer.position = (buffer.position + 1) % CAPACITY;
    if(buffer.position == 0 && !buffer.full) {
        buffer.full = true;
    }
}