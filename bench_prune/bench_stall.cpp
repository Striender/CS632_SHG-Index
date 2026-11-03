// Per-distance-evaluation CPU time breakdown on the base layer (memory stall / compute / search overhead),
// measured from the real search: cold run vs immediate warm rerun of the same query.
#include "hnswlib_patched/hnswlib.h"
#include <sys/resource.h>
#include <sched.h>
#include <chrono>
#include <random>
#include <numeric>
using namespace std;
static size_t g_ndist=0; static hnswlib::DISTFUNC<float> g_real;
static float countingL2(const void*a,const void*b,const void*p){ g_ndist++; return g_real(a,b,p); }
struct CountingSpace : hnswlib::L2Space { CountingSpace(size_t d):hnswlib::L2Space(d){ g_real=hnswlib::L2Space::get_dist_func(); }
  hnswlib::DISTFUNC<float> get_dist_func() override { return countingL2; } };
static long majflt(){ rusage u; getrusage(RUSAGE_SELF,&u); return u.ru_majflt; }
static long rss_kb(){ FILE*f=fopen("/proc/self/status","r"); char l[256]; long v=0; while(fgets(l,256,f)) if(!strncmp(l,"VmRSS:",6)) v=atol(l+6); fclose(f); return v; }
int main(int argc,char**argv){
  // args: base.data_new query.fvecs groundtruth.ivecs [NQ]
  string fb=argv[1], fq=argv[2], fg=argv[3]; int NQ=argc>4?atoi(argv[4]):1000000; const int K=20, P=5;
  int d; vector<float> Q; bool bv=fq.size()>6 && fq.substr(fq.size()-6)==".bvecs"; { FILE*f=fopen(fq.c_str(),"rb"); int dim; int n=0; while(n<NQ && fread(&dim,4,1,f)==1){ vector<float> t(dim); if(bv){ vector<unsigned char> b(dim); fread(b.data(),1,dim,f); for(int j=0;j<dim;j++) t[j]=b[j]; } else fread(t.data(),4,dim,f); Q.insert(Q.end(),t.begin(),t.end()); d=dim; n++; } fclose(f); NQ=n; }
  vector<vector<int>> gt(NQ); { FILE*f=fopen(fg.c_str(),"rb"); for(int i=0;i<NQ;i++){int k; fread(&k,4,1,f); vector<int> t(k); fread(t.data(),4,k,f); gt[i].assign(t.begin(),t.begin()+K);} fclose(f);}
  FILE* fbase=fopen(fb.c_str(),"rb"); int hdr[3]; fread(hdr,4,3,fbase); size_t N=hdr[1]; if(argc>5) N=min(N,(size_t)atol(argv[5])); /* optional subset for sanitizer runs */ if(hdr[2]!=d){fprintf(stderr,"dim mismatch %d vs %d\n",hdr[2],d); return 1;}
  printf("N=%zu d=%d NQ=%d\n",N,d,NQ); fflush(stdout);
  CountingSpace sp(d); Performance per; auto T0=chrono::steady_clock::now(); auto sec=[&]{return chrono::duration<double>(chrono::steady_clock::now()-T0).count();};
  auto* S=new hnswlib::HEDS<float>(&sp,d,N+NQ,48,80);
  { vector<float> x(d); for(size_t i=0;i<N;i++){ fread(x.data(),4,d,fbase); S->addDataPoint(x.data(),i,1,per); if(i%100000==0){printf("  inserted %zu (%.0fs)\n",i,sec()); fflush(stdout);} } fclose(fbase); }
  double build=sec(); T0=chrono::steady_clock::now(); int ni=N; S->buildShortcuts(ni); double sc=sec();
  for(int i=0;i<NQ;i++) S->addDataPoint(&Q[(size_t)i*d],N+i,-1,per);
  S->resultsProcessing.assign(N*(S->maxlevel_+1),-1);
  size_t up=0; for(size_t i=0;i<N;i++) if(S->element_levels_[i]>0) up+=S->size_links_per_element_*S->element_levels_[i];
  double hnsw_mb=(N*S->size_data_per_element_+up)/1048576.0, shg_mb=hnsw_mb+(N*S->data_rep_size_*4.0+S->Shortcuts.size_in_bytes())/1048576.0;
  printf("build %.0fs shortcuts %.1fs n_shortcuts=%zu maxlevel=%d RSS=%.0fMB\n",build,sc,(size_t)S->shortcutsSize,S->maxlevel_,rss_kb()/1024.0); fflush(stdout);
  cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(2,&cs); sched_setaffinity(0,sizeof(cs),&cs);
  auto c0=__rdtsc(); auto t0=chrono::steady_clock::now(); while(chrono::duration<double>(chrono::steady_clock::now()-t0).count()<0.3); double hz=(__rdtsc()-c0)/0.3;

  // ---- 1. core clock: dependent add chain retires 1 iteration / core cycle
  auto core_ghz=[&]{ const uint64_t n=2000000000ULL; uint64_t x=0, c=n; auto t=chrono::steady_clock::now();
    asm volatile("1: add $1, %0\n\t dec %1\n\t jnz 1b" : "+r"(x), "+r"(c));
    return n/chrono::duration<double>(chrono::steady_clock::now()-t).count()/1e9; };
  double ghz=core_ghz(); double cyc_per_tsc=ghz*1e9/hz;
  printf("core clock %.3f GHz, TSC %.3f GHz\n",ghz,hz/1e9); fflush(stdout);
  // ---- 2. DRAM latency: pointer chase over 1 GiB, one dependent load per random cache line
  double dram_cyc; { size_t L=(1ULL<<30)/64; vector<size_t> perm(L); iota(perm.begin(),perm.end(),0); shuffle(perm.begin(),perm.end(),mt19937_64(7));
    vector<size_t> nxt(L*8); for(size_t i=0;i<L;i++) nxt[perm[i]*8]=perm[(i+1)%L]*8;
    size_t p=0; for(int i=0;i<1000000;i++) p=nxt[p]; const int R=20000000; auto a=__rdtsc(); for(int i=0;i<R;i++) p=nxt[p]; auto b=__rdtsc();
    dram_cyc=(b-a)*cyc_per_tsc/R; if(p==1) printf("."); }
  printf("DRAM dependent-load latency %.0f core cycles\n",dram_cyc); fflush(stdout);
  // ---- 3. pure distance compute: real L2 kernel on 8 cache-resident vectors
  double comp_cyc; { const void* v[8]; for(int j=0;j<8;j++) v[j]=S->getDataByInternalId(j*7+1); const void* q=S->getDataByInternalId(N);
    float acc=0; for(int i=0;i<100000;i++) acc+=g_real(q,v[i&7],sp.get_dist_func_param()); const int R=2000000; auto a=__rdtsc();
    for(int i=0;i<R;i++) acc+=g_real(q,v[i&7],sp.get_dist_func_param()); auto b=__rdtsc(); comp_cyc=(b-a)*cyc_per_tsc/R; if(acc==1.2345f) printf("."); }
  printf("distance compute (cache-resident) %.1f core cycles\n",comp_cyc); fflush(stdout);
  // ---- 4. real search: base layer only, cold run then warm rerun of the same query
  struct M{string name; int abl; hnswlib::PrefetchMode pm; bool hnsw; int la; size_t lines=3;};
  // Idea 2 lookahead distance L: while scanning neighbour j, prefetch neighbour j+L (visited flag + first 3 cache lines of its vector)
  vector<M> ms={{"SHG_original",0,hnswlib::PrefetchMode::BASELINE,false,2},{"SHG_stock",0,hnswlib::PrefetchMode::STOCK_HNSW,false,2},
                {"HNSW_noprefetch",0,hnswlib::PrefetchMode::BASELINE,true,2},{"HNSW_stock",0,hnswlib::PrefetchMode::STOCK_HNSW,true,2}};
#ifdef FULLVEC
  // whole-vector variant: same lookahead hook, but every cache line of the neighbour's vector is prefetched
  ms.push_back({"SHG_idea2_L2",0,hnswlib::PrefetchMode::STEP2_SKIP_LOOKAHEAD,false,2}); ms.push_back({"HNSW_idea2_L2",0,hnswlib::PrefetchMode::STEP2_SKIP_LOOKAHEAD,true,2});
  for(int L=1;L<=3;L++){ ms.push_back({"SHG_full_L"+to_string(L),0,hnswlib::PrefetchMode::STEP2_SKIP_LOOKAHEAD,false,L,(size_t)1<<20});
                         ms.push_back({"HNSW_full_L"+to_string(L),0,hnswlib::PrefetchMode::STEP2_SKIP_LOOKAHEAD,true,L,(size_t)1<<20}); }
#else
  for(int L=2;L<=5;L++){ ms.push_back({"SHG_idea2_L"+to_string(L),0,hnswlib::PrefetchMode::STEP2_SKIP_LOOKAHEAD,false,L});
                         ms.push_back({"HNSW_idea2_L"+to_string(L),0,hnswlib::PrefetchMode::STEP2_SKIP_LOOKAHEAD,true,L}); }
#endif
  int NM=ms.size(); const int P2=3;
  vector<int> ids_tmp; auto run=[&](M& m,int i){ g_abl=m.abl; hnswlib::PrefetchController::set_mode(m.pm); hnswlib::PrefetchController::lookahead_dist=m.la; hnswlib::PrefetchController::lookahead_lines=m.lines; Query q(N+i,K);
    auto a=__rdtsc(); auto r = m.hnsw ? S->searchKnnHNSW(q) : S->searchKnnShortcuts(q); auto b=__rdtsc();
    ids_tmp.clear(); while(!r.empty()){ ids_tmp.push_back((int)r.top().second); r.pop(); } sort(ids_tmp.begin(),ids_tmp.end());
    for(size_t x:g_touched) S->resultsProcessing[x]=-1; g_touched.clear(); return make_pair((double)(b-g_tsc_split),(double)S->baseNumDis); };
  vector<int> ref(NM); for(int a=0;a<NM;a++) for(int b=0;b<NM;b++) if(ms[b].hnsw==ms[a].hnsw && ms[b].pm==hnswlib::PrefetchMode::BASELINE) ref[a]=b;
  FILE* chk=fopen("correctness.csv","w"); fprintf(chk,"method,ef,reference,queries,mismatched_queries\n");
  FILE* out=fopen("stall.csv","w");
  fprintf(out,"method,ef,core_ghz,dram_latency_cyc,compute_cyc_per_eval,base_evals_per_q,cold_cyc_per_eval,warm_cyc_per_eval,stall_cyc_per_eval,overhead_cyc_per_eval,cold_base_us,warm_base_us\n");
  for(int ef:{20,64,100}){ S->setEf(ef);
    for(int i=0;i<min(NQ,300);i++) for(auto&m:ms) run(m,i);
    vector<double> cold(NM,0),warm(NM,0),ev(NM,0); vector<vector<vector<int>>> ids(NM,vector<vector<int>>(NQ));
    vector<vector<int>> perm(P2,vector<int>(NM)); { mt19937 rng(777+ef); for(auto& o:perm){ iota(o.begin(),o.end(),0); shuffle(o.begin(),o.end(),rng);} }
    for(int p=0;p<P2;p++) for(int k=0;k<NM;k++) for(int i=0;i<NQ;i++){ int mi=perm[p][k];
      auto c=run(ms[mi],i); if(p==0) ids[mi][i]=ids_tmp; auto w=run(ms[mi],i); cold[mi]+=c.first; warm[mi]+=w.first; ev[mi]+=c.second; }
    for(int mi=0;mi<NM;mi++){ int bad=0; for(int i=0;i<NQ;i++) bad+=ids[mi][i]!=ids[ref[mi]][i]; fprintf(chk,"%s,%d,%s,%d,%d\n",ms[mi].name.c_str(),ef,ms[ref[mi]].name.c_str(),NQ,bad);
      if(bad) printf("CORRECTNESS MISMATCH %s ef=%d: %d/%d\n",ms[mi].name.c_str(),ef,bad,NQ); } fflush(chk);
    for(int mi=0;mi<NM;mi++){ double n=P2*(double)NQ, e=ev[mi]/n, cc=cold[mi]/n*cyc_per_tsc/e, wc=warm[mi]/n*cyc_per_tsc/e;
      fprintf(out,"%s,%d,%.3f,%.0f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f\n",ms[mi].name.c_str(),ef,ghz,dram_cyc,comp_cyc,e,cc,wc,cc-wc,wc-comp_cyc,cold[mi]/n/hz*1e6,warm[mi]/n/hz*1e6);
      printf("%-16s ef=%3d evals/q=%.0f cold=%.0f warm=%.0f stall=%.0f compute=%.0f overhead=%.0f cyc/eval\n",ms[mi].name.c_str(),ef,e,cc,wc,cc-wc,comp_cyc,wc-comp_cyc); }
    fflush(out); fflush(stdout); }
}
