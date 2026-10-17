#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace pfm {
constexpr int R = 128;
struct Plan {
    std::vector<uint32_t> offsets, vertices, anchors, dims;
    std::vector<int32_t> edges;
};
int root(std::vector<int>& parents,int x) {
    while (parents[x]!=x) { parents[x]=parents[parents[x]]; x=parents[x]; }
    return x;
}
Plan make_plan(const std::vector<uint32_t>& pairs,int k,bool degree_anchor=false) {
    Plan p;
    p.offsets.resize((k/R)*(R+1)); p.vertices.resize(k); p.anchors.resize(k);
    for (int c=0;c<k/R;++c) {
        std::vector<int> parents(R);
        std::vector<std::vector<int32_t>> incident(R);
        for (int t=0;t<R;++t) parents[t]=t;
        for (int s=0;s<R;++s) {
            int a=pairs[2*(c*R+s)], b=pairs[2*(c*R+s)+1];
            int x=root(parents,a),y=root(parents,b);
            if (x!=y) parents[std::max(x,y)]=std::min(x,y);
            incident[a].push_back(s+1); incident[b].push_back(-s-1);
        }
        // This choice depends only on the fixed B graph, so it is reusable
        // across fresh A candidates. Ties prefer the smaller vertex index.
        std::vector<int> best(R,-1);
        if (degree_anchor) for (int v=0;v<R;++v) {
            int component=root(parents,v),old=best[component];
            if (old<0 || incident[v].size()>incident[old].size()) best[component]=v;
        }
        int t=0;
        for (int v=0;v<R;++v) {
            int component=root(parents,v);
            int a=degree_anchor ? best[component] : component;
            if (v==a) continue;
            p.vertices[c*R+t]=v; p.anchors[c*R+t]=a;
            p.offsets[c*(R+1)+t]=p.edges.size();
            p.edges.insert(p.edges.end(),incident[v].begin(),incident[v].end()); ++t;
        }
        p.dims.push_back(t);
        for (;t<=R;++t) p.offsets[c*(R+1)+t]=p.edges.size();
    }
    return p;
}
} // namespace pfm
