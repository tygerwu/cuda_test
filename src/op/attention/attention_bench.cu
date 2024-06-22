#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "utils.h"
#include "utils.cuh"
#include "cuda_runtime.h"
#include "attention_api.cuh"

using AttentionFunc = 
    std::function<void(const __half* q,const __half* k,const __half* v,__half* o,
                       int batch,int head_num,int qo_seq_len,int kv_seq_len,int head_dim,float softmax_scale,
                       int q_batch_stride,int kv_batch_stride,int o_batch_stride,
                       cudaStream_t stream)>;

extern int32_t profile_loop;

class AttentionBench : public ::testing::Test{
    using T = __half;

    public:
        void Bench(AttentionFunc func){
            std::vector<float> times;

            Prepare();

            auto hQ = RandomFloats<T>(qo_size,-1,1);
            auto hK = RandomFloats<T>(kv_size,-1,1);
            auto hV = RandomFloats<T>(kv_size,-1,1);
            auto hO = std::vector<T>(qo_size);

            int warms = 5;
            for(int i=0; i<profile_loop; i++){
                T *dQ,*dK,*dV,*dO;
                CUDA_ERROR_CHECK(cudaMalloc(&dQ,qo_bytes));
                CUDA_ERROR_CHECK(cudaMalloc(&dK,kv_bytes));
                CUDA_ERROR_CHECK(cudaMalloc(&dV,kv_bytes));
                CUDA_ERROR_CHECK(cudaMalloc(&dO,qo_bytes));

                CUDA_ERROR_CHECK(cudaMemcpy(dQ,hQ.data(),qo_bytes,cudaMemcpyHostToDevice));
                CUDA_ERROR_CHECK(cudaMemcpy(dK,hQ.data(),kv_bytes,cudaMemcpyHostToDevice));
                CUDA_ERROR_CHECK(cudaMemcpy(dV,hQ.data(),kv_bytes,cudaMemcpyHostToDevice));


                cudaStream_t stream;
                cudaStreamCreate(&stream);

                cudaEvent_t start,stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                cudaEventRecord(start,stream);
                func(dQ,dK,dV,dO,
                     batch,head_num,qo_seq_len,kv_seq_len,head_dim,softmax_scale,
                     q_batch_stride,kv_batch_stride,o_batch_stride,
                     stream);
                auto err = cudaGetLastError();
                if(err != cudaSuccess){
                    printf("err = %d, str = %s\n",err,cudaGetErrorString(err));
                    EXPECT_EQ(0,1);
                }

                CUDA_ERROR_CHECK(cudaEventRecord(stop,stream));
                CUDA_ERROR_CHECK(cudaEventSynchronize(stop));

                float time_ms = 0;
                CUDA_ERROR_CHECK(cudaEventElapsedTime(&time_ms,start,stop));
                if(i >= warms){
                    times.push_back(time_ms);
                }
                
                cudaStreamSynchronize(stream);
                cudaDeviceSynchronize();
                CUDA_ERROR_CHECK(cudaMemcpy(hO.data(),dO,qo_bytes,cudaMemcpyDeviceToHost));

                cudaFree(dQ);
                cudaFree(dK);
                cudaFree(dV);
                cudaFree(dO);

                CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
            }
            std::cout << "Average Time:" << Average(times) << std::endl;
        }

    private:
        void Prepare(){
            hidden_size = head_dim * head_num;
            qo_size = batch * qo_seq_len * hidden_size;
            kv_size = batch * kv_seq_len * hidden_size;

            qo_bytes = qo_size * sizeof(T);
            kv_bytes = kv_size * sizeof(T);
        }

    public:
        int batch,head_num,head_dim,qo_seq_len,kv_seq_len;
        float softmax_scale;
        int loops = 10;

        int q_batch_stride;
        int kv_batch_stride;
        int o_batch_stride;
    private:
        int hidden_size;
        int qo_size,kv_size;
        int qo_bytes,kv_bytes;
        
};


TEST_F(AttentionBench,test){
    batch = 1;
    head_num = 2;
    head_dim = 128;
    qo_seq_len = 16384;
    kv_seq_len = 16384;
    softmax_scale = 1;
    q_batch_stride = head_num * head_dim * qo_seq_len;
    kv_batch_stride = head_num * head_dim * kv_seq_len;
    o_batch_stride = q_batch_stride;
    loops = 1;
    // Op::PrintAttentionV1Info(head_dim);
    // Bench(Op::AttentionV1);

    
    Op::PrintAttentionV1Info(head_dim);
    Op::PrintAttentionV2Info(head_dim);
    Bench(Op::AttentionV2);
    Bench(Op::AttentionV1);
    Bench(Op::AttentionV2);
    Bench(Op::AttentionV1);
}