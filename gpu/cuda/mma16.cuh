#pragma once
// INT8 m16n8k16 fragments follow NVIDIA PTX ISA section 9.7.16.5.9.
// One block covers 16x128 outputs; four warps each cover 16x32 outputs.
// Shared row pitch 144 avoids the 128-byte pitch's fragment-load bank conflicts.
namespace pfm_mma16 {
constexpr int Rank=128, Rows=16, Cols=128, Pitch=144;

__device__ __forceinline__ void mma(int (&d)[4],uint32_t a0,uint32_t a1,uint32_t b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(b));
}

// Grouped+Inline groups A inside the shared-memory load, then fuses residual
// corrections, all prefix accumulations and ticket-state reductions. Prepared
// grouping uses the existing global grouping pass as an additional control.
template<bool Grouped,bool Inline,bool Trim,bool Capture,bool Correct=true>
__global__ void prefix(const int8_t* a,const int8_t* packed_b,
                       const uint32_t* offsets,const int32_t* edges,
                       const uint32_t* dims,const uint32_t* global_counts,
                       const uint32_t* global_indices,const int16_t* global_residuals,
                       int32_t* prefixes,uint32_t* states,int m,int n,int k) {
    __shared__ __align__(16) int8_t sa[Rows*Pitch];
    __shared__ __align__(16) int8_t sb[Cols*Pitch];
    __shared__ __align__(16) int8_t raw[Rows*Pitch];
    __shared__ unsigned int counts[Rows];
    __shared__ uint8_t indices[Rows*Rank];
    __shared__ int16_t residuals[Rows*Rank];
    int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    int group=lane/4,quad=lane%4;
    int row_base=blockIdx.y*Rows,col_base=blockIdx.x*Cols;
    int ticket_col=col_base/32+warp;
    int top_ticket=(row_base/8)*(n/32)+ticket_col;
    int bottom_ticket=top_ticket+n/32;
    bool valid_ticket=ticket_col<n/32;
    if (lane<16 && valid_ticket) {
        states[top_ticket*16+lane]=0;
        if (row_base+8<m) states[bottom_ticket*16+lane]=0;
    }
    int acc[4][4] = {};
    for (int c=0;c<k/Rank;++c) {
        if (Grouped && Inline) {
            if (tid<Rows) counts[tid]=0;
            for (int x=tid;x<Rows*(Rank/16);x+=128) {
                int row=x/(Rank/16),t=(x%(Rank/16))*16,i=row_base+row;
                uint4 value=i<m ? *reinterpret_cast<const uint4*>(&a[i*k+c*Rank+t]) : make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(&raw[row*Pitch+t])=value;
            }
            __syncthreads();
            for (int x=tid;x<Rows*Rank;x+=128) {
                int row=x/Rank,t=x%Rank,sum=0;
                if (t<int(dims[c])) {
                    for (uint32_t p=offsets[c*(Rank+1)+t];p<offsets[c*(Rank+1)+t+1];++p) {
                        int edge=edges[p],s=(edge>0?edge:-edge)-1;
                        int value=raw[row*Pitch+s]; sum+=edge>0?value:-value;
                    }
                }
                int lo=sum < -128 ? -128 : (sum>127 ? 127 : sum);
                sa[row*Pitch+t]=int8_t(lo);
                if (sum!=lo) {
                    unsigned slot=atomicAdd(&counts[row],1u);
                    indices[row*Rank+slot]=uint8_t(t);
                    residuals[row*Rank+slot]=int16_t(sum-lo);
                }
            }
        } else {
            for (int x=tid;x<Rows*(Rank/16);x+=128) {
                int row=x/(Rank/16),t=(x%(Rank/16))*16,i=row_base+row;
                const int8_t* source=Grouped ? &a[(c*m+i)*Rank+t] : &a[i*k+c*Rank+t];
                uint4 value=i<m ? *reinterpret_cast<const uint4*>(source) : make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(&sa[row*Pitch+t])=value;
            }
        }
        for (int x=tid;x<Cols*(Rank/16);x+=128) {
            int col=x/(Rank/16),t=(x%(Rank/16))*16,j=col_base+col;
            uint4 value=j<n ? *reinterpret_cast<const uint4*>(&packed_b[(c*n+j)*Rank+t]) : make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(&sb[col*Pitch+t])=value;
        }
        __syncthreads();
        int width=Grouped && Trim ? ((int(dims[c])+15)/16)*16 : Rank;
        for (int t=0;t<width;t+=16) {
            int kt=t+quad*4;
            uint32_t a0=*reinterpret_cast<const uint32_t*>(&sa[group*Pitch+kt]);
            uint32_t a1=*reinterpret_cast<const uint32_t*>(&sa[(group+8)*Pitch+kt]);
            #pragma unroll
            for (int f=0;f<4;++f) {
                uint32_t b=*reinterpret_cast<const uint32_t*>(&sb[(warp*32+f*8+group)*Pitch+kt]);
                mma(acc[f],a0,a1,b);
            }
        }
        // Correct=false is a deliberately invalid diagnostic, never a method
        // eligible for the reported grouped/direct performance comparison.
        if (Grouped && Correct) {
            #pragma unroll
            for (int half=0;half<2;++half) {
                int row=group+half*8,i=row_base+row,global_row=c*m+i;
                unsigned count=i>=m ? 0 : (Inline ? counts[row] : global_counts[global_row]);
                for (unsigned s=0;s<count;++s) {
                    unsigned index=Inline ? indices[row*Rank+s] : global_indices[global_row*Rank+s];
                    int value=Inline ? residuals[row*Rank+s] : global_residuals[global_row*Rank+s];
                    #pragma unroll
                    for (int f=0;f<4;++f) {
                        int col=warp*32+f*8+quad*2;
                        acc[f][half*2]+=value*int(sb[col*Pitch+index]);
                        acc[f][half*2+1]+=value*int(sb[(col+1)*Pitch+index]);
                    }
                }
            }
        }
        uint32_t top=0,bottom=0;
        #pragma unroll
        for (int f=0;f<4;++f) {
            int col=col_base+warp*32+f*8+quad*2;
            if (Capture) {
                if (col<n) {
                    prefixes[(c*m+row_base+group)*n+col]=acc[f][0];
                    prefixes[(c*m+row_base+group)*n+col+1]=acc[f][1];
                    if (row_base+group+8<m) {
                        prefixes[(c*m+row_base+group+8)*n+col]=acc[f][2];
                        prefixes[(c*m+row_base+group+8)*n+col+1]=acc[f][3];
                    }
                }
            }
            top^=uint32_t(acc[f][0])^uint32_t(acc[f][1]);
            bottom^=uint32_t(acc[f][2])^uint32_t(acc[f][3]);
        }
        #pragma unroll
        for (int shift=16;shift>0;shift/=2) {
            top^=__shfl_xor_sync(0xffffffffu,top,shift);
            bottom^=__shfl_xor_sync(0xffffffffu,bottom,shift);
        }
        if (lane==0 && valid_ticket) {
            uint32_t old=states[top_ticket*16+c%16];
            states[top_ticket*16+c%16]=((old<<13)|(old>>19))^top;
            if (row_base+8<m) {
                old=states[bottom_ticket*16+c%16];
                states[bottom_ticket*16+c%16]=((old<<13)|(old>>19))^bottom;
            }
        }
        __syncthreads();
    }
}
} // namespace pfm_mma16
