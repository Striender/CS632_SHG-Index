// HNSW vs SHG with / without base-layer pruning, same graph, interleaved per query.
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
  struct M{const char* name; int abl; hnswlib::PrefetchMode pm; bool hnsw;};
  // abl: 0 = SHG as published (base-layer pruning on), 2 = base-layer pruning off. Prefetch: none / hnswlib stock / Idea 2.
  vector<M> ms={{"HNSW_stock",0,hnswlib::PrefetchMode::STOCK_HNSW,true},{"HNSW_noprefetch",0,hnswlib::PrefetchMode::BASELINE,true},{"HNSW_idea2",0,hnswlib::PrefetchMode::STEP2_SKIP_LOOKAHEAD,true},
                {"SHG_original",0,hnswlib::PrefetchMode::BASELINE,false},{"SHG_stock",0,hnswlib::PrefetchMode::STOCK_HNSW,false},{"SHG_idea2",0,hnswlib::PrefetchMode::STEP2_SKIP_LOOKAHEAD,false},
                {"SHG_noprune",2,hnswlib::PrefetchMode::BASELINE,false},{"SHG_noprune_stock",2,hnswlib::PrefetchMode::STOCK_HNSW,false},{"SHG_noprune_idea2",2,hnswlib::PrefetchMode::STEP2_SKIP_LOOKAHEAD,false}};
  int NM=ms.size();
  // correctness reference: same search (hnsw flag + abl) with no prefetch
  vector<int> ref(NM); for(int a=0;a<NM;a++) for(int b=0;b<NM;b++) if(ms[b].hnsw==ms[a].hnsw && ms[b].abl==ms[a].abl && ms[b].pm==hnswlib::PrefetchMode::BASELINE) ref[a]=b;
  FILE* chk=fopen("correctness.csv","w"); fprintf(chk,"method,ef,reference,queries,mismatched_queries\n");
  // count full-precision distances via the metric counters is not uniform; count with a wrapper instead
  FILE* out=fopen("prune.csv","w");
  fprintf(out,"method,ef,recall,lat_mean_us,lat_p50_us,lat_p95_us,lat_p99_us,qps,dist_per_q,upper_frac,skip_q_frac,lvls_skipped_per_q,build_s,shortcut_s,index_mb,rss_mb,major_faults\n");
  FILE* pq=fopen("prune_perquery.csv","w"); fprintf(pq,"method,ef,qid,lat_us,recall\n");
  auto run=[&](M& m,int i){ g_abl=m.abl; hnswlib::PrefetchController::set_mode(m.pm); Query q(N+i,K);
    auto r = m.hnsw ? S->searchKnnHNSW(q) : S->searchKnnShortcuts(q);
    for(size_t x:g_touched) S->resultsProcessing[x]=-1; g_touched.clear(); return r; };
  for(int ef:{20,30,40,60,80,120,200,300}){ S->setEf(ef);
    for(int i=0;i<min(NQ,300);i++) for(auto&m:ms) run(m,i);
    vector<vector<vector<double>>> lat(NM,vector<vector<double>>(NQ)); vector<vector<float>> rec(NM,vector<float>(NQ)); vector<vector<vector<int>>> ids(NM,vector<vector<int>>(NQ));
    vector<double> dst(NM,0),upc(NM,0),totc(NM,0),skq(NM,0),lsk(NM,0); long f0=majflt();
    // Each method runs over ALL queries as one block; block order is reshuffled every repetition.
    // (Running all methods back-to-back on the same query let a method inherit the previous method's warm cache.)
    vector<vector<int>> perm(P,vector<int>(NM)); { mt19937 rng(12345+ef); for(auto& o:perm){ iota(o.begin(),o.end(),0); shuffle(o.begin(),o.end(),rng);} }
    for(int p=0;p<P;p++) for(int k=0;k<NM;k++) for(int i=0;i<NQ;i++){ int mi=perm[p][k]; auto& m=ms[mi];
      g_lvl_skipped=0; g_skip_events=0; size_t d0=g_ndist; auto a=__rdtsc(); auto r=run(m,i); auto b=__rdtsc(); dst[mi]+=g_ndist-d0;
      lat[mi][i].push_back((b-a)/hz*1e6); upc[mi]+=g_tsc_split-a; totc[mi]+=b-a; skq[mi]+=g_skip_events; lsk[mi]+=g_lvl_skipped;
      if(p==0){ unordered_set<int> g(gt[i].begin(),gt[i].end()); int h=0; while(!r.empty()){h+=g.count((int)r.top().second); ids[mi][i].push_back((int)r.top().second); r.pop();} sort(ids[mi][i].begin(),ids[mi][i].end()); rec[mi][i]=h/(float)K; } }
    long faults=majflt()-f0;
    for(int mi=0;mi<NM;mi++){ int bad=0; for(int i=0;i<NQ;i++) bad+=ids[mi][i]!=ids[ref[mi]][i]; fprintf(chk,"%s,%d,%s,%d,%d\n",ms[mi].name,ef,ms[ref[mi]].name,NQ,bad); if(bad) printf("CORRECTNESS MISMATCH %s ef=%d: %d/%d queries differ from %s\n",ms[mi].name,ef,bad,NQ,ms[ref[mi]].name); } fflush(chk);
    for(int mi=0;mi<NM;mi++){ vector<double> med(NQ); for(int i=0;i<NQ;i++){ auto v=lat[mi][i]; nth_element(v.begin(),v.begin()+P/2,v.end()); med[i]=v[P/2]; fprintf(pq,"%s,%d,%d,%.2f,%.3f\n",ms[mi].name,ef,i,med[i],rec[mi][i]); }
      vector<double> s=med; sort(s.begin(),s.end()); double mean=accumulate(med.begin(),med.end(),0.0)/NQ, R=accumulate(rec[mi].begin(),rec[mi].end(),0.0)/NQ; auto pc=[&](double q){return s[min((size_t)(q*NQ),s.size()-1)];};
      fprintf(out,"%s,%d,%.4f,%.1f,%.1f,%.1f,%.1f,%.1f,%.0f,%.4f,%.4f,%.3f,%.0f,%.1f,%.1f,%.1f,%ld\n",ms[mi].name,ef,R,mean,pc(.5),pc(.95),pc(.99),1e6/mean,dst[mi]/(P*NQ),upc[mi]/totc[mi],skq[mi]/(P*NQ),lsk[mi]/(P*NQ),build,sc,ms[mi].hnsw?hnsw_mb:shg_mb,rss_kb()/1024.0,faults);
      printf("%-20s ef=%3d recall=%.4f mean=%.1fus p50=%.1f p99=%.1f upper=%.1f%% majflt=%ld\n",ms[mi].name,ef,R,mean,pc(.5),pc(.99),100*upc[mi]/totc[mi],faults); }
    fflush(out); fflush(pq); fflush(stdout); }
}
