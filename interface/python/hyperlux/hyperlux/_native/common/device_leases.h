#pragma once
// Device arrays a stream still reads or writes, oldest batch first, each behind an event recorded
// after the work that uses it. The leases are the imported arrays themselves, not their Python
// producers: a DLPack import owns its storage independently of the producer, which may hand its
// only reference to the consumer. nanobind-free so nvcc-compiled owners can hold one.
#include <cuda_runtime.h>
#include "cuda_device.h"

#include <deque>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace hyperlux {
    struct DeviceLeases {
        struct Batch {
            cudaEvent_t ready;
            std::vector<std::shared_ptr<void>> arrays;
        };
        std::deque<Batch> in_flight;
        std::vector<cudaEvent_t> spare_events;

        DeviceLeases() = default;
        DeviceLeases(const DeviceLeases&) = delete;
        DeviceLeases& operator=(const DeviceLeases&) = delete;
        // waits for the work the leases guard, so an owner needs no stream synchronization of its own first
        ~DeviceLeases(){
            for(const Batch& batch : in_flight){
                cudaEventSynchronize(batch.ready);
                cudaEventDestroy(batch.ready);
            }
            for(cudaEvent_t event : spare_events){
                cudaEventDestroy(event);
            }
        }

        // how many of the oldest batches the stream is done with
        size_t release(){
            size_t released = 0;
            while(!in_flight.empty() && cudaEventQuery(in_flight.front().ready) == cudaSuccess){
                spare_events.push_back(in_flight.front().ready);
                in_flight.pop_front();
                released++;
            }
            return released;
        }

        // keeps the arrays alive until the work enqueued on stream (of the owner's GPU `device`, where
        // 0 is its default stream) so far completes; when the event cannot be recorded, waits for
        // that work so dropping them is safe
        void retain(int device, cudaStream_t stream, std::initializer_list<std::shared_ptr<void>> arrays){
            const CurrentCudaDevice current(device);
            cudaEvent_t event = nullptr;
            try {
                if(spare_events.empty()){
                    if(cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess){
                        throw std::runtime_error("hyperlux: cudaEventCreate failed");
                    }
                }
                else {
                    event = spare_events.back();
                    spare_events.pop_back();
                }
                if(cudaEventRecord(event, stream) != cudaSuccess){
                    throw std::runtime_error("hyperlux: cudaEventRecord failed");
                }
                in_flight.push_back({event, arrays});
            }
            catch (...) {
                cudaStreamSynchronize(stream);
                if(event != nullptr){
                    cudaEventDestroy(event);
                }
                throw;
            }
            release();
        }
    };
}
