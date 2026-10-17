// Exact matrix-stage feasibility benchmark. No mining or network submission.
#include "plan.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#include "mma16.cuh"

namespace {
constexpr int R = 128, H = 8, W = 32;
thread_local std::string last_error;
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void ck(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(int(e)));
}
struct DeviceBuffer {
    void* p = nullptr;
    explicit DeviceBuffer(size_t bytes) { ck(cudaMalloc(&p, std::max(bytes, size_t(4)))); }
    ~DeviceBuffer() { if (p) cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template<class T> T* as() { return static_cast<T*>(p); }
};
struct Handles {
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    Handles() {
        try {
            ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            ck(cublasCreate(&blas));
            ck(cublasSetStream(blas, stream));
            ck(cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH));
            ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
        } catch (...) { release(); throw; }
    }
    void release() {
        if (end) cudaEventDestroy(end);
        if (begin) cudaEventDestroy(begin);
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }
    ~Handles() { release(); }
};

__global__ void materialize(const int8_t* v, const uint32_t* pairs,
                            int8_t* b, int n, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n*k) return;
    int t=x%R, j=(x/R)%n, c=x/(R*n), s=c*R+t;
    b[x] = int(v[j*R+pairs[2*s]]) - int(v[j*R+pairs[2*s+1]]);
}
__global__ void pack_a(const int8_t* a, int8_t* packed, int m, int k) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    if (x>=m*k) return;
    int t=x%R, i=(x/R)%m, c=x/(R*m);
    packed[x]=a[i*k+c*R+t];
}
__global__ void contrasts(const int8_t* v, const uint32_t* vertices,
                          const uint32_t* anchors, const uint32_t* dims,
                          int8_t* b, int n, int k) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    if (x>=n*k) return;
    int t=x%R, j=(x/R)%n, c=x/(R*n);
    b[x]= t<int(dims[c]) ? int(v[j*R+vertices[c*R+t]])-int(v[j*R+anchors[c*R+t]]) : 0;
}

// One row of G per block. Keep the dense INT8 part and a sparse exact residual.
// All 128 threads participate, including inactive padded coefficients.
__global__ void group_a(const int8_t* a, const uint32_t* offsets,
                       const int32_t* edges, const uint32_t* dims,
                       int8_t* clipped, uint32_t* counts,
                       uint32_t* indices, int16_t* residuals, int m, int k) {
    __shared__ unsigned int count;
    int t=threadIdx.x, row=blockIdx.x, i=row%m, c=row/m;
    if (t==0) count=0;
    __syncthreads();
    int sum=0;
    if (t<int(dims[c])) {
        for (uint32_t p=offsets[c*(R+1)+t];p<offsets[c*(R+1)+t+1];++p) {
            int edge=edges[p], s=(edge>0 ? edge : -edge)-1;
            int value=a[i*k+c*R+s];
            sum += edge>0 ? value : -value;
        }
    }
    int lo=sum < -128 ? -128 : (sum>127 ? 127 : sum);
    clipped[row*R+t]=int8_t(lo);
    if (sum!=lo) {
        unsigned slot=atomicAdd(&count,1u);
        indices[row*R+slot]=t;
        residuals[row*R+slot]=int16_t(sum-lo);
    }
    __syncthreads();
    if (t==0) counts[row]=count;
}
__global__ void correct(const int8_t* b, const uint32_t* counts,
                        const uint32_t* indices, const int16_t* residuals,
                        int32_t* delta, int m, int n, int chunks) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    if (x>=chunks*m*n) return;
    int j=x%n, row=x/n, c=row/m, correction=0;
    for (uint32_t s=0;s<counts[row];++s)
        correction += int(residuals[row*R+s])*int(b[(c*n+j)*R+indices[row*R+s]]);
    delta[x] += correction;
}

// Identical reduction/output work for both methods. Capture prefixes only for
// correctness; timing keeps only the 16-word state for each 8x32 ticket.
template<bool Capture>
__global__ void finish(const int32_t* delta, int32_t* prefixes,
                       uint32_t* states, int m, int n, int chunks) {
    __shared__ uint32_t warp_xor[8];
    int tid=threadIdx.x, lane=tid%32, warp=tid/32;
    int i=blockIdx.y*H+warp, j=blockIdx.x*W+lane;
    int ticket=blockIdx.y*(n/W)+blockIdx.x;
    int32_t acc=0;
    if (tid<16) states[ticket*16+tid]=0;
    __syncthreads();
    for (int c=0;c<chunks;++c) {
        int x=(c*m+i)*n+j;
        acc += delta[x];
        if (Capture) prefixes[x]=acc;
        uint32_t v=uint32_t(acc);
        for (int shift=16;shift>0;shift/=2) v^=__shfl_xor_sync(0xffffffffu,v,shift);
        if (lane==0) warp_xor[warp]=v;
        __syncthreads();
        if (tid==0) {
            uint32_t combined=0;
            for (int w=0;w<8;++w) combined^=warp_xor[w];
            uint32_t old=states[ticket*16+c%16];
            states[ticket*16+c%16]=((old<<13)|(old>>19))^combined;
        }
        __syncthreads();
    }
}

using pfm::Plan;
using pfm::make_plan;
struct Context {
    int m,n,k,candidates,chunks;
    Handles h;
    std::vector<uint32_t> host_pairs;
    DeviceBuffer a,v,pairs,bt,contr,offsets,edges,vertices,anchors,dims;
    DeviceBuffer clipped,counts,indices,residuals,delta,prefixes,states;
    int prepared=-1;
    Context(int m_,int n_,int k_,int candidates_,const int8_t* host_a,
            const int8_t* host_v,const uint32_t* host_p):
        m(m_),n(n_),k(k_),candidates(candidates_),chunks(k/R),
        host_pairs(host_p,host_p+2*k),
        a(size_t(candidates)*m*k),v(size_t(n)*R),pairs(size_t(2)*k*4),
        bt(size_t(n)*k),contr(size_t(n)*k),offsets(size_t(chunks)*(R+1)*4),
        edges(size_t(2)*k*4),vertices(size_t(k)*4),anchors(size_t(k)*4),dims(size_t(chunks)*4),
        clipped(size_t(m)*k),counts(size_t(chunks)*m*4),indices(size_t(m)*k*4),
        residuals(size_t(m)*k*2),delta(size_t(chunks)*m*n*4),
        prefixes(size_t(chunks)*m*n*4),states(size_t(m/H)*(n/W)*16*4) {
        ck(cudaMemcpy(a.p,host_a,size_t(candidates)*m*k,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(v.p,host_v,size_t(n)*R,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(pairs.p,host_p,size_t(2)*k*4,cudaMemcpyHostToDevice));
    }
    template<class T> void upload(DeviceBuffer& b,const std::vector<T>& values) {
        // Synchronous copies keep the temporary host plan alive through copying.
        if (!values.empty()) ck(cudaMemcpy(b.p,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
    }
    void prepare(int method) {
        if (method==0 || method==2 || method==3) {
            materialize<<<(n*k+255)/256,256,0,h.stream>>>(v.as<int8_t>(),pairs.as<uint32_t>(),bt.as<int8_t>(),n,k);
        } else {
            Plan p=make_plan(host_pairs,k);
            upload(offsets,p.offsets); upload(edges,p.edges); upload(vertices,p.vertices);
            upload(anchors,p.anchors); upload(dims,p.dims);
            contrasts<<<(n*k+255)/256,256,0,h.stream>>>(v.as<int8_t>(),vertices.as<uint32_t>(),anchors.as<uint32_t>(),dims.as<uint32_t>(),contr.as<int8_t>(),n,k);
        }
        ck(cudaGetLastError()); prepared=method;
    }
    template<bool Grouped,bool Inline,bool Trim>
    void fused(const int8_t* input,bool capture) {
        dim3 grid((n+127)/128,(m+15)/16);
        auto b=(Grouped ? contr.as<int8_t>() : bt.as<int8_t>());
        if (capture)
            pfm_mma16::prefix<Grouped,Inline,Trim,true><<<grid,128,0,h.stream>>>(input,b,offsets.as<uint32_t>(),edges.as<int32_t>(),dims.as<uint32_t>(),counts.as<uint32_t>(),indices.as<uint32_t>(),residuals.as<int16_t>(),prefixes.as<int32_t>(),states.as<uint32_t>(),m,n,k);
        else
            pfm_mma16::prefix<Grouped,Inline,Trim,false><<<grid,128,0,h.stream>>>(input,b,offsets.as<uint32_t>(),edges.as<int32_t>(),dims.as<uint32_t>(),counts.as<uint32_t>(),indices.as<uint32_t>(),residuals.as<int16_t>(),prefixes.as<int32_t>(),states.as<uint32_t>(),m,n,k);
    }
    void run(int method,int candidate,bool capture) {
        if (prepared!=method) throw std::runtime_error("prepare B for the selected method first");
        if (candidate<0 || candidate>=candidates) throw std::runtime_error("candidate out of range");
        auto input=a.as<int8_t>()+size_t(candidate)*m*k;
        if (method>=3) {
            if (method==3) fused<false,false,false>(input,capture);
            else if (method==4) {
                group_a<<<chunks*m,R,0,h.stream>>>(input,offsets.as<uint32_t>(),edges.as<int32_t>(),dims.as<uint32_t>(),clipped.as<int8_t>(),counts.as<uint32_t>(),indices.as<uint32_t>(),residuals.as<int16_t>(),m,k);
                fused<true,false,true>(clipped.as<int8_t>(),capture);
            } else if (method==5) fused<true,true,true>(input,capture);
            else fused<true,true,false>(input,capture);
            ck(cudaGetLastError()); return;
        }
        int8_t* b=bt.as<int8_t>();
        int lda=k;
        long long stride_a=R;
        if (method==1) {
            group_a<<<chunks*m,R,0,h.stream>>>(input,offsets.as<uint32_t>(),edges.as<int32_t>(),dims.as<uint32_t>(),clipped.as<int8_t>(),counts.as<uint32_t>(),indices.as<uint32_t>(),residuals.as<int16_t>(),m,k);
            input=clipped.as<int8_t>(); b=contr.as<int8_t>(); lda=R; stride_a=static_cast<long long>(m)*R;
        }
        if (method==2) {
            pack_a<<<(m*k+255)/256,256,0,h.stream>>>(input,clipped.as<int8_t>(),m,k);
            input=clipped.as<int8_t>(); lda=R; stride_a=static_cast<long long>(m)*R;
        }
        // Row-major output is column-major C^T = B * A^T. Each rank-128
        // chunk is computed independently, then finish accumulates the prefixes.
        const int32_t alpha=1,beta=0;
        ck(cublasGemmStridedBatchedEx(h.blas,CUBLAS_OP_T,CUBLAS_OP_N,
            n,m,R,&alpha,b,CUDA_R_8I,R,static_cast<long long>(n)*R,
            input,CUDA_R_8I,lda,stride_a,&beta,delta.p,CUDA_R_32I,n,
            static_cast<long long>(m)*n,chunks,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        if (method==1)
            correct<<<(chunks*m*n+255)/256,256,0,h.stream>>>(b,counts.as<uint32_t>(),indices.as<uint32_t>(),residuals.as<int16_t>(),delta.as<int32_t>(),m,n,chunks);
        dim3 tiles(n/W,m/H);
        if (capture) finish<true><<<tiles,256,0,h.stream>>>(delta.as<int32_t>(),prefixes.as<int32_t>(),states.as<uint32_t>(),m,n,chunks);
        else finish<false><<<tiles,256,0,h.stream>>>(delta.as<int32_t>(),prefixes.as<int32_t>(),states.as<uint32_t>(),m,n,chunks);
        ck(cudaGetLastError());
    }
};
template<class F> int guarded(F f) {
    try { f(); last_error.clear(); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
} // namespace

extern "C" {
const char* pfm_error() { return last_error.c_str(); }
void* pfm_create(int m,int n,int k,int candidates,const int8_t* a,
                 const int8_t* v,const uint32_t* pairs) {
    Context* ctx=nullptr;
    guarded([&] {
        if (m<=0 || n<=0 || m%H || n%W || k!=2048 || candidates<=0 ||
            size_t(m)*n*(k/R)>size_t(INT32_MAX)) throw std::runtime_error("unsupported shape");
        if (!a || !v || !pairs) throw std::runtime_error("null input");
        for (int s=0;s<2*k;++s) if (pairs[s]>=R) throw std::runtime_error("invalid selection vertex");
        // V in [-32,31] makes every signed difference exactly representable.
        for (int s=0;s<n*R;++s) if (v[s]<-32 || v[s]>31) throw std::runtime_error("V outside supported noise range");
        ctx=new Context(m,n,k,candidates,a,v,pairs);
    });
    return ctx;
}
void pfm_destroy(void* p) { delete static_cast<Context*>(p); }
int pfm_prepare(void* p,int method) {
    return guarded([&] {
        if (method<0 || method>6) throw std::runtime_error("invalid method");
        auto& c=*static_cast<Context*>(p); c.prepare(method); ck(cudaStreamSynchronize(c.h.stream));
    });
}
int pfm_run(void* p,int method,int candidate,int capture) {
    return guarded([&] { auto& c=*static_cast<Context*>(p); if (method<0 || method>6) throw std::runtime_error("invalid method"); c.run(method,candidate,capture!=0); ck(cudaStreamSynchronize(c.h.stream)); });
}
int pfm_read(void* p,int32_t* prefixes,uint32_t* states) {
    return guarded([&] {
        auto& c=*static_cast<Context*>(p);
        if (prefixes) ck(cudaMemcpy(prefixes,c.prefixes.p,size_t(c.chunks)*c.m*c.n*4,cudaMemcpyDeviceToHost));
        if (states) ck(cudaMemcpy(states,c.states.p,size_t(c.m/H)*(c.n/W)*16*4,cudaMemcpyDeviceToHost));
    });
}
// mode=0: B preparation including CPU planning/uploads; mode=1: all candidates.
// The CUDA event span can include host scheduling gaps; it is not active GPU time.
int pfm_measure(void* p,int method,int mode,double* wall_us,double* stream_us) {
    return guarded([&] {
        if (method<0 || method>6 || mode<0 || mode>1) throw std::runtime_error("invalid measurement mode");
        auto& c=*static_cast<Context*>(p);
        ck(cudaStreamSynchronize(c.h.stream));
        auto start=std::chrono::steady_clock::now();
        ck(cudaEventRecord(c.h.begin,c.h.stream));
        if (mode==0) c.prepare(method);
        else for (int candidate=0;candidate<c.candidates;++candidate) c.run(method,candidate,false);
        ck(cudaEventRecord(c.h.end,c.h.stream)); ck(cudaEventSynchronize(c.h.end));
        *wall_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        float ms=0; ck(cudaEventElapsedTime(&ms,c.h.begin,c.h.end)); *stream_us=double(ms)*1000;
    });
}
int pfm_device(char* name,int capacity,int* major,int* minor,int* runtime,int* blas_version) {
    return guarded([&] {
        cudaDeviceProp prop; int device=0; ck(cudaGetDevice(&device)); ck(cudaGetDeviceProperties(&prop,device));
        std::snprintf(name,capacity,"%s",prop.name); *major=prop.major; *minor=prop.minor;
        ck(cudaRuntimeGetVersion(runtime)); Handles h; ck(cublasGetVersion(h.blas,blas_version));
    });
}
}
