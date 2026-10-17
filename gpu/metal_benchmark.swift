import Foundation
import Metal

struct Fixture: Decodable {
    let name: String
    let m,n,k,r,h,w,id,tickets: Int
    let nonzero_a: Bool
    let a,bt,v,pairs,vertices,anchors,d,prefixes,states,hashes,seed_a: String
}
struct Manifest: Decodable { let fixtures:[Fixture] }
struct Shape { var m,n,k,r: UInt32 }
struct Plan { var offsets:[UInt32], edges:[Int32], vertices:[UInt32], anchors:[UInt32], dims:[UInt32] }
struct Times { let wall,gpu,host: Double }

func root(_ parents: inout [Int],_ initial:Int)->Int {
    var x=initial
    while parents[x] != x { let next=parents[parents[x]]; parents[x]=next; x=next }
    return x
}
func makePlan(_ pairs:[UInt32],_ f:Fixture)->Plan {
    var p=Plan(offsets:Array(repeating:0,count:(f.k/f.r)*(f.r+1)),edges:[],vertices:Array(repeating:0,count:f.k),anchors:Array(repeating:0,count:f.k),dims:[])
    for c in 0..<(f.k/f.r) {
        var parents=Array(0..<f.r)
        var incident=Array(repeating:[Int32](),count:f.r)
        for s in 0..<f.r {
            let a=Int(pairs[2*(c*f.r+s)]),b=Int(pairs[2*(c*f.r+s)+1])
            let x=root(&parents,a),y=root(&parents,b)
            if x != y {parents[max(x,y)]=min(x,y)}
            incident[a].append(Int32(s+1));incident[b].append(-Int32(s+1))
        }
        var t=0
        for vertex in 0..<f.r {
            let anchor=root(&parents,vertex)
            if vertex==anchor {continue}
            p.vertices[c*f.r+t]=UInt32(vertex);p.anchors[c*f.r+t]=UInt32(anchor)
            p.offsets[c*(f.r+1)+t]=UInt32(p.edges.count)
            p.edges.append(contentsOf:incident[vertex]);t+=1
        }
        p.dims.append(UInt32(t))
        for tail in t...f.r {p.offsets[c*(f.r+1)+tail]=UInt32(p.edges.count)}
    }
    return p
}
func words<T:FixedWidthInteger>(_ data:Data,_ type:T.Type)->[T] {
    precondition(data.count%MemoryLayout<T>.stride==0)
    return data.withUnsafeBytes { bytes in
        (0..<(data.count/MemoryLayout<T>.stride)).map { T(littleEndian:bytes.loadUnaligned(fromByteOffset:$0*MemoryLayout<T>.stride,as:T.self)) }
    }
}
func elapsed(_ start:UInt64)->Double {Double(DispatchTime.now().uptimeNanoseconds-start)/1000.0}
func median(_ x:[Double])->Double {x.sorted()[x.count/2]}

guard CommandLine.arguments.count==4 || (CommandLine.arguments.count==5 && CommandLine.arguments[4]=="--cached") else {fatalError("Usage: metal-benchmark FIXTURE_DIR SHADER_PATH REPORT_JSON [--cached]")}
let cachedMode=CommandLine.arguments.count==5
let fixtureDir=URL(fileURLWithPath:CommandLine.arguments[1])
let manifest=try JSONDecoder().decode(Manifest.self,from:Data(contentsOf:fixtureDir.appendingPathComponent("manifest.json")))
guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else {fatalError("Metal GPU unavailable")}
let shader=try String(contentsOfFile:CommandLine.arguments[2],encoding:.utf8)
let options=MTLCompileOptions() // All shader arithmetic is integer.
let library=try device.makeLibrary(source:shader,options:options)
var pipelines=[String:MTLComputePipelineState]()
for name in ["materialize_b","group_a","contrasts","direct_prefix","grouped_prefix","pack_a","direct_packed_prefix"] {
    let state=try device.makeComputePipelineState(function:library.makeFunction(name:name)!)
    precondition(state.maxTotalThreadsPerThreadgroup>=256)
    pipelines[name]=state
}
func data(_ name:String)throws->Data {try Data(contentsOf:fixtureDir.appendingPathComponent(name))}
func buffer(_ contents:Data)->MTLBuffer {contents.withUnsafeBytes{device.makeBuffer(bytes:$0.baseAddress!,length:$0.count,options:.storageModeShared)!}}
func blank(_ bytes:Int)->MTLBuffer {device.makeBuffer(length:max(bytes,4),options:.storageModeShared)!}
func copy<T>(_ values:[T],to buffer:MTLBuffer) {
    values.withUnsafeBytes {bytes in
        precondition(bytes.count<=buffer.length)
        if bytes.count>0 {buffer.contents().copyMemory(from:bytes.baseAddress!,byteCount:bytes.count)}
    }
}
func encode(_ name:String,_ command:MTLCommandBuffer,_ buffers:[MTLBuffer],_ shape:inout Shape,_ count:Int?=nil) {
    let e=command.makeComputeCommandEncoder()!
    e.setComputePipelineState(pipelines[name]!)
    for (i,b) in buffers.enumerated(){e.setBuffer(b,offset:0,index:i)}
    e.setBytes(&shape,length:MemoryLayout<Shape>.stride,index:buffers.count)
    if let count=count {
        e.dispatchThreads(MTLSize(width:count,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
    } else {
        e.dispatchThreadgroups(MTLSize(width:Int(shape.n)/32,height:Int(shape.m)/8,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
    }
    e.endEncoding()
}
func complete(_ command:MTLCommandBuffer) {
    command.commit();command.waitUntilCompleted()
    if let error=command.error {fatalError("GPU command failed: \(error)")}
    precondition(command.status == .completed)
}
func validate(_ f:Fixture,_ out:MTLBuffer,_ expected:[Int32],_ expectedStates:[UInt32],_ method:String) {
        let p=out.contents().bindMemory(to:Int32.self,capacity:expected.count)
        var states=Array(repeating:UInt32(0),count:f.tickets*16)
        for c in 0..<(f.k/f.r) {for i in 0..<f.m {for j in 0..<f.n {
            let index=(c*f.m+i)*f.n+j
            precondition(p[index]==expected[index],"\(f.name) \(method) prefix mismatch at \(index)")
        }}}
        for c in 0..<(f.k/f.r) {for ti in 0..<(f.m/8) {for tj in 0..<(f.n/32) {
            var x:UInt32=0
            for i in ti*8..<(ti+1)*8 {for j in tj*32..<(tj+1)*32 {x ^= UInt32(bitPattern:p[(c*f.m+i)*f.n+j])}}
            let index=(ti*(f.n/32)+tj)*16+c%16
            let old=states[index];states[index]=((old<<13)|(old>>19))^x
        }}}
        precondition(states==expectedStates,"\(f.name) \(method) tile state mismatch")
    }

var reports=[[String:Any]]()
var checkedEntries=0,checkedTickets=0
print("Metal device: \(device.name)")
for f in manifest.fixtures {
    precondition(f.r==128 && f.k==2048 && f.h==8 && f.w==32 && f.m%8==0 && f.n%32==0)
    var shape=Shape(m:UInt32(f.m),n:UInt32(f.n),k:UInt32(f.k),r:UInt32(f.r))
    let a=buffer(try data(f.a)),bt=buffer(try data(f.bt)),v=buffer(try data(f.v)),pairData=try data(f.pairs)
    let pairBuffer=buffer(pairData),pairs=words(pairData,UInt32.self)
    let offsets=blank((f.k/f.r)*(f.r+1)*4),edges=blank(2*f.k*4),vertices=blank(f.k*4),anchors=blank(f.k*4),dims=blank(f.k/f.r*4)
    let g=blank(f.k*f.m*2),b=blank(f.k*f.n),out=blank(f.k/f.r*f.m*f.n*4)
    let expected=words(try data(f.prefixes),Int32.self)
    let expectedStates=words(try data(f.states),UInt32.self)
    let referencePlan=makePlan(pairs,f)
    let ev=words(try data(f.vertices),UInt32.self),ea=words(try data(f.anchors),UInt32.self),ed=words(try data(f.d),UInt32.self)
    precondition(referencePlan.vertices==ev && referencePlan.anchors==ea && referencePlan.dims==ed)
    func loadPlan(_ p:Plan) {copy(p.offsets,to:offsets);copy(p.edges,to:edges);copy(p.vertices,to:vertices);copy(p.anchors,to:anchors);copy(p.dims,to:dims)}
    func run(_ grouped:Bool,_ job:Bool,_ repetitions:Int=1)->Times {
        let begin=DispatchTime.now().uptimeNanoseconds
        var host=0.0
        if grouped && job {loadPlan(makePlan(pairs,f));host=elapsed(begin)}
        let command=queue.makeCommandBuffer()!
        for _ in 0..<repetitions {
            if grouped {
                if job {
                    encode("group_a",command,[a,offsets,edges,dims,g],&shape,f.k*f.m)
                    encode("contrasts",command,[v,vertices,anchors,dims,b],&shape,f.k*f.n)
                }
                encode("grouped_prefix",command,[g,b,dims,out],&shape)
            } else {
                if job {encode("materialize_b",command,[v,pairBuffer,bt],&shape,f.k*f.n)}
                encode("direct_prefix",command,[a,bt,out],&shape)
            }
        }
        complete(command)
        return Times(wall:elapsed(begin)/Double(repetitions),gpu:(command.gpuEndTime-command.gpuStartTime)*1e6/Double(repetitions),host:host)
    }
    loadPlan(referencePlan)
    _=run(false,true);validate(f,out,expected,expectedStates,"direct")
    _=run(true,true);validate(f,out,expected,expectedStates,"grouped")
    checkedEntries+=expected.count;checkedTickets+=f.tickets
    print("PASS \(f.name): \(expected.count) prefix entries, \(f.tickets) tile states")
    if cachedMode || f.id != 0 || f.nonzero_a {continue}
    // Warm both methods. Job timings include fresh host graph/CSR construction.
    _=run(false,true);_=run(true,true)
    var dw=[Double](),dg=[Double](),gw=[Double](),gg=[Double](),gh=[Double](),dk=[Double](),gk=[Double]()
    for round in 0..<7 {
        for offset in 0..<2 {
            if (round+offset)%2==0 {let t=run(false,true);dw.append(t.wall);dg.append(t.gpu)}
            else {let t=run(true,true);gw.append(t.wall);gg.append(t.gpu);gh.append(t.host)}
        }
        for offset in 0..<2 {
            if (round+offset)%2==0 {dk.append(run(false,false,10).gpu)}
            else {gk.append(run(true,false,10).gpu)}
        }
    }
    reports.append(["name":f.name,"m":f.m,"n":f.n,"tickets":f.tickets,"direct_wall_us":dw,"direct_gpu_us":dg,"grouped_wall_us":gw,"grouped_gpu_us":gg,"grouped_host_plan_us":gh,"direct_kernel_gpu_us":dk,"grouped_kernel_gpu_us":gk,"median_direct_wall_us":median(dw),"median_grouped_wall_us":median(gw),"wall_speedup":median(dw)/median(gw),"median_direct_kernel_gpu_us":median(dk),"median_grouped_kernel_gpu_us":median(gk)])
    print(String(format:"  medians: direct job %.2f us; grouped job %.2f us; %.3fx wall speedup; GPU kernels %.2f / %.2f us",median(dw),median(gw),median(dw)/median(gw),median(dk),median(gk)))
}
if cachedMode {
    let families=Dictionary(grouping:manifest.fixtures,by:{"\($0.m)-\($0.n)-\($0.id)"})
    let groups=families.values.sorted { a,b in
        if a[0].m*a[0].n != b[0].m*b[0].n {return a[0].m*a[0].n < b[0].m*b[0].n}
        if a[0].m != b[0].m {return a[0].m < b[0].m}
        return a[0].id < b[0].id
    }
    for unsorted in groups {
        let candidates=unsorted.sorted {$0.name < $1.name}
        let f=candidates[0]
        precondition(candidates.count==4)
        var shape=Shape(m:UInt32(f.m),n:UInt32(f.n),k:UInt32(f.k),r:UInt32(f.r))
        let inputs=try candidates.map {buffer(try data($0.a))}
        let vData=try data(f.v),pairData=try data(f.pairs),btData=try data(f.bt)
        // Fixed B inputs and distinct noised A inputs are required for this experiment.
        var seen=Set<Data>(),seenHashes=Set<Data>(),seenSeeds=Set<String>()
        for candidate in candidates {
            let cv=try data(candidate.v),cp=try data(candidate.pairs),cb=try data(candidate.bt),ca=try data(candidate.a)
            precondition(cv==vData && cp==pairData && cb==btData)
            let hashes=try data(candidate.hashes)
            precondition(seen.insert(ca).inserted && seenHashes.insert(hashes).inserted && seenSeeds.insert(candidate.seed_a).inserted)
        }
        let v=buffer(vData),pairBuffer=buffer(pairData),pairs=words(pairData,UInt32.self)
        let bt=blank(f.k*f.n),b=blank(f.k*f.n),g=blank(f.k*f.m*2),packedA=blank(f.k*f.m)
        let offsets=blank((f.k/f.r)*(f.r+1)*4),edges=blank(2*f.k*4),vertices=blank(f.k*4),anchors=blank(f.k*4),dims=blank(f.k/f.r*4)
        let out=blank(f.k/f.r*f.m*f.n*4)
        func prepareB(_ grouped:Bool)->Times {
            let start=DispatchTime.now().uptimeNanoseconds
            if grouped {
                let p=makePlan(pairs,f)
                copy(p.offsets,to:offsets);copy(p.edges,to:edges);copy(p.vertices,to:vertices);copy(p.anchors,to:anchors);copy(p.dims,to:dims)
            }
            let host=elapsed(start)
            let command=queue.makeCommandBuffer()!
            if grouped {encode("contrasts",command,[v,vertices,anchors,dims,b],&shape,f.k*f.n)}
            else {encode("materialize_b",command,[v,pairBuffer,bt],&shape,f.k*f.n)}
            complete(command)
            return Times(wall:elapsed(start),gpu:(command.gpuEndTime-command.gpuStartTime)*1e6,host:host)
        }
        func encodeCandidate(_ grouped:Bool,_ command:MTLCommandBuffer,_ input:MTLBuffer,_ packed:Bool=false) {
            if grouped {
                // This depends on A and MUST run for every new candidate.
                encode("group_a",command,[input,offsets,edges,dims,g],&shape,f.k*f.m)
                encode("grouped_prefix",command,[g,b,dims,out],&shape)
            } else if packed {
                encode("pack_a",command,[input,packedA],&shape,f.k*f.m)
                encode("direct_packed_prefix",command,[packedA,bt,out],&shape)
            } else {encode("direct_prefix",command,[input,bt,out],&shape)}
        }
        func candidatesTime(_ grouped:Bool,_ packed:Bool=false)->Times {
            let start=DispatchTime.now().uptimeNanoseconds
            let command=queue.makeCommandBuffer()!
            for input in inputs {encodeCandidate(grouped,command,input,packed)}
            complete(command)
            let count=Double(inputs.count)
            return Times(wall:elapsed(start)/count,gpu:(command.gpuEndTime-command.gpuStartTime)*1e6/count,host:0)
        }
        // Check the actual cached execution, not only the uncached job path above.
        for method in 0..<3 {
            let grouped=method==1,packed=method==2
            _=prepareB(grouped)
            for (index,candidate) in candidates.enumerated() {
                let command=queue.makeCommandBuffer()!
                encodeCandidate(grouped,command,inputs[index],packed);complete(command)
                validate(candidate,out,words(try data(candidate.prefixes),Int32.self),words(try data(candidate.states),UInt32.self),grouped ? "cached grouped" : (packed ? "cached packed direct" : "cached direct"))
            }
            _=candidatesTime(grouped,packed)
        }
        var directPrep=[Double](),groupedPrep=[Double](),directWall=[Double](),groupedWall=[Double](),directGpu=[Double](),groupedGpu=[Double]()
        var packedPrep=[Double](),packedWall=[Double](),packedGpu=[Double]()
        for round in 0..<11 {
            for offset in 0..<3 {
                let method=(round+offset)%3,grouped=method==1,packed=method==2
                let prep=prepareB(grouped),t=candidatesTime(grouped,packed)
                if grouped {groupedPrep.append(prep.wall);groupedWall.append(t.wall);groupedGpu.append(t.gpu)}
                else if packed {packedPrep.append(prep.wall);packedWall.append(t.wall);packedGpu.append(t.gpu)}
                else {directPrep.append(prep.wall);directWall.append(t.wall);directGpu.append(t.gpu)}
            }
        }
        let count=Double(candidates.count)
        let directAmortized=zip(directWall,directPrep).map {$0+$1/count}
        let groupedAmortized=zip(groupedWall,groupedPrep).map {$0+$1/count}
        let packedAmortized=zip(packedWall,packedPrep).map {$0+$1/count}
        var caseReport:[String:Any]=["m":f.m,"n":f.n,"job_id":f.id,"candidate_a_seeds":candidates.map {$0.seed_a},"distinct_candidates":candidates.count,"tickets_per_candidate":f.tickets,"direct_b_setup_wall_us":directPrep,"grouped_b_setup_wall_us":groupedPrep,"direct_candidate_wall_us":directWall,"grouped_candidate_wall_us":groupedWall,"direct_candidate_gpu_us":directGpu,"grouped_candidate_gpu_us":groupedGpu,"direct_amortized_wall_us":directAmortized,"grouped_amortized_wall_us":groupedAmortized,"median_direct_candidate_wall_us":median(directWall),"median_grouped_candidate_wall_us":median(groupedWall),"candidate_wall_speedup":median(directWall)/median(groupedWall),"median_direct_amortized_wall_us":median(directAmortized),"median_grouped_amortized_wall_us":median(groupedAmortized),"amortized_wall_speedup":median(directAmortized)/median(groupedAmortized),"candidate_names":candidates.map {$0.name}]
        let chunkDims=words(try data(f.d),UInt32.self)
        let reducedK=chunkDims.reduce(0,+)
        caseReport["chunk_dimensions"]=chunkDims
        caseReport["direct_scalar_multiplications_per_candidate"]=f.m*f.n*f.k
        caseReport["grouped_scalar_multiplications_per_candidate"]=f.m*f.n*Int(reducedK)
        caseReport["scalar_multiplication_reduction"]=1-Double(reducedK)/Double(f.k)
        caseReport["paired_amortized_speedup_samples"]=zip(directAmortized,groupedAmortized).map {$0/$1}
        caseReport["packed_direct_b_setup_wall_us"]=packedPrep
        caseReport["packed_direct_candidate_wall_us"]=packedWall
        caseReport["packed_direct_candidate_gpu_us"]=packedGpu
        caseReport["packed_direct_amortized_wall_us"]=packedAmortized
        caseReport["median_packed_direct_candidate_wall_us"]=median(packedWall)
        caseReport["median_packed_direct_amortized_wall_us"]=median(packedAmortized)
        let bestDirect=median(directAmortized)<=median(packedAmortized) ? directAmortized : packedAmortized
        caseReport["faster_direct_control"]=median(directAmortized)<=median(packedAmortized) ? "direct" : "packed-direct"
        caseReport["speedup_vs_faster_direct"]=median(bestDirect)/median(groupedAmortized)
        caseReport["paired_speedup_vs_faster_direct"]=zip(bestDirect,groupedAmortized).map {$0/$1}
        reports.append(caseReport)
        print(String(format:"  stronger A-layout control: packed direct %.2f us; %.3fx grouped speedup versus faster direct",median(packedAmortized),median(bestDirect)/median(groupedAmortized)))
        print(String(format:"CACHED %dx%d, job %d, %d distinct candidates: direct %.2f us / grouped %.2f us (%.3fx); with B setup / %d: %.2f / %.2f us (%.3fx)",f.m,f.n,f.id,candidates.count,median(directWall),median(groupedWall),median(directWall)/median(groupedWall),candidates.count,median(directAmortized),median(groupedAmortized),median(directAmortized)/median(groupedAmortized)))
    }
}

var report:[String:Any]=["device":device.name,"os":ProcessInfo.processInfo.operatingSystemVersionString,"fixtures":manifest.fixtures.count,"checked_prefix_entries":checkedEntries,"checked_tickets":checkedTickets,"cases":reports,"mode":cachedMode ? "fixed-b-distinct-a" : "uncached-prefix","scope":cachedMode ? "Fixed B reused equally by both methods. Four distinct sparse raw A commitments and fresh commitment-derived A noise per shape, prepared before timing. All candidates in each timed batch are distinct; timing rounds replay this finite fixture set. Per-candidate grouped timing includes GPU A grouping; packed direct includes GPU A packing; faster of both direct controls reported; B plan and contrasts prepared once per batch. Both methods write all prefixes. Amortized times add B setup divided by four. Excludes raw commitments, seed/noise generation, final state/hash generation, proof generation, buffer allocation and compilation. Local integer prototypes, no network mining claim." : "Metal integer prototypes. Both jobs start from A', V and signed B selections. Includes direct chunk-packed B materialization; grouped graph/CSR construction, GPU grouping and contrasts; both write every prefix. Job wall includes encoding and waiting. Kernel-only GPU timings exclude preparation. Metal buffer allocation, shader compilation, raw commitments, A noising, final hashing and proof generation excluded. No official GPU miner or network comparison."]
report["compiler"]="swiftc -O; runtime integer Metal compilation"
report["samples_per_case"]=cachedMode ? 11 : 7
report["rank"]=128;report["inner_dimension"]=2048;report["tile_shape"]=[8,32]
if cachedMode {
    report["synthetic_job_key_definition"]="BLAKE3(\"pearl-batch-case-{id}\"); id 0 and 1 predetermined, without selection search"
    report["raw_inputs"]="B is zero. Each A is zero except raw_a[0][candidate]=1 for candidate=0..3. Every raw AB is zero."
}
let reportData=try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys])
try reportData.write(to:URL(fileURLWithPath:CommandLine.arguments[3]))
print("PASS all \(checkedEntries) prefixes and \(checkedTickets) ticket states; saved \(CommandLine.arguments[3])")
