
#pragma once
#include "cuda_fp16.h"
#include "cuda_runtime.h"
namespace Op{

void AttentionV1(const __half* q,const __half* k,const __half* v,__half* o,
                 int batch,int head_num,int qo_seq_len,int kv_seq_len,int head_dim,float softmax_scale,
                 int q_batch_stride,int kv_batch_stride,int o_batch_stride,
                 cudaStream_t stream);
void PrintAttentionV1Info(int HD);


void AttentionV2(const __half* q,const __half* k,const __half* v,__half* o,
                 int batch,int head_num,int qo_seq_len,int kv_seq_len,int head_dim,float softmax_scale,
                 int q_batch_stride,int kv_batch_stride,int o_batch_stride,
                 cudaStream_t stream);
void PrintAttentionV2Info(int HD);


}