#include <metal_stdlib>
using namespace metal;
struct Shape { uint m,n,k,r; };
constant uint H=8, W=32, R=128;

// Both job paths start with A', V and the signed B selections.
kernel void materialize_b(device const char* v [[buffer(0)]],
                          device const uint* pairs [[buffer(1)]],
                          device char* bt [[buffer(2)]],
                          constant Shape& z [[buffer(3)]],
                          uint x [[thread_position_in_grid]]) {
    if(x>=z.n*z.k) return;
    uint j=x/z.k,s=x%z.k;
    bt[((s/R)*z.n+j)*R+s%R]=v[j*R+pairs[2*s]]-v[j*R+pairs[2*s+1]];
}

kernel void group_a(device const char* a [[buffer(0)]],
                    device const uint* offsets [[buffer(1)]],
                    device const int* edges [[buffer(2)]],
                    device const uint* dims [[buffer(3)]],
                    device short* g [[buffer(4)]],
                    constant Shape& z [[buffer(5)]],
                    uint x [[thread_position_in_grid]]) {
    if(x>=(z.k/R)*z.m*R) return;
    uint t=x%R, i=(x/R)%z.m, c=x/(R*z.m);
    int sum=0;
    if(t<dims[c]) {
        for(uint p=offsets[c*(R+1)+t];p<offsets[c*(R+1)+t+1];p++) {
            int edge=edges[p];
            int value=int(a[i*z.k+c*R+uint(abs(edge)-1)]);
            sum+=edge>0 ? value : -value;
        }
    }
    g[x]=short(sum);
}

kernel void contrasts(device const char* v [[buffer(0)]],
                       device const uint* vertices [[buffer(1)]],
                       device const uint* anchors [[buffer(2)]],
                       device const uint* dims [[buffer(3)]],
                       device char* b [[buffer(4)]],
                       constant Shape& z [[buffer(5)]],
                       uint x [[thread_position_in_grid]]) {
    if(x>=(z.k/R)*z.n*R) return;
    uint t=x%R, j=(x/R)%z.n, c=x/(R*z.n);
    b[x]= t<dims[c] ? char(int(v[j*R+vertices[c*R+t]])-int(v[j*R+anchors[c*R+t]])) : 0;
}

// One 256-thread group computes an 8x32 tile and retains its accumulators.
// Cooperative loads read contiguous input rows, then transpose B in shared memory.
kernel void direct_prefix(device const char* a [[buffer(0)]],
                          device const char* bt [[buffer(1)]],
                          device int* out [[buffer(2)]],
                          constant Shape& z [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 tile [[threadgroup_position_in_grid]]) {
    threadgroup char sa[8*128];
    threadgroup char sb[128*32];
    uint row=tid/32,col=tid%32;
    uint i=tile.y*8+row,j=tile.x*32+col;
    int acc=0;
    for(uint c=0;c<z.k/R;c++) {
        for(uint x=tid;x<8*R;x+=256) sa[x]=a[(tile.y*8+x/R)*z.k+c*R+x%R];
        for(uint x=tid;x<32*R;x+=256) sb[(x%R)*32+x/R]=bt[(c*z.n+tile.x*32+x/R)*R+x%R];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint t=0;t<R;t++) acc+=int(sa[row*R+t])*int(sb[t*32+col]);
        out[(c*z.m+i)*z.n+j]=acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void grouped_prefix(device const short* g [[buffer(0)]],
                           device const char* b [[buffer(1)]],
                           device const uint* dims [[buffer(2)]],
                           device int* out [[buffer(3)]],
                           constant Shape& z [[buffer(4)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint2 tile [[threadgroup_position_in_grid]]) {
    threadgroup short sa[8*128];
    threadgroup char sb[128*32];
    uint row=tid/32,col=tid%32;
    uint i=tile.y*8+row,j=tile.x*32+col;
    int acc=0;
    for(uint c=0;c<z.k/R;c++) {
        uint d=dims[c];
        for(uint x=tid;x<8*R;x+=256) sa[x]=g[(c*z.m+tile.y*8+x/R)*R+x%R];
        for(uint x=tid;x<32*R;x+=256) sb[(x%R)*32+x/R]=b[(c*z.n+tile.x*32+x/R)*R+x%R];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint t=0;t<d;t++) acc+=int(sa[row*R+t])*int(sb[t*32+col]);
        out[(c*z.m+i)*z.n+j]=acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Direct control with the same chunk/row layout as the grouped A operand.
// Packing is performed once per candidate and is included in cached timings.
kernel void pack_a(device const char* a [[buffer(0)]],
                   device char* packed [[buffer(1)]],
                   constant Shape& z [[buffer(2)]],
                   uint x [[thread_position_in_grid]]) {
    if(x>=z.m*z.k) return;
    uint t=x%R,i=(x/R)%z.m,c=x/(R*z.m);
    packed[x]=a[i*z.k+c*R+t];
}

kernel void direct_packed_prefix(device const char* a [[buffer(0)]],
                          device const char* bt [[buffer(1)]],
                          device int* out [[buffer(2)]],
                          constant Shape& z [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint2 tile [[threadgroup_position_in_grid]]) {
    threadgroup char sa[8*128];
    threadgroup char sb[128*32];
    uint row=tid/32,col=tid%32;
    uint i=tile.y*8+row,j=tile.x*32+col;
    int acc=0;
    for(uint c=0;c<z.k/R;c++) {
        for(uint x=tid;x<8*R;x+=256) sa[x]=a[(c*z.m+tile.y*8+x/R)*R+x%R];
        for(uint x=tid;x<32*R;x+=256) sb[(x%R)*32+x/R]=bt[(c*z.n+tile.x*32+x/R)*R+x%R];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint t=0;t<R;t++) acc+=int(sa[row*R+t])*int(sb[t*32+col]);
        out[(c*z.m+i)*z.n+j]=acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
