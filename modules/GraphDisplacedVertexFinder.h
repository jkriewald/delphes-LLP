#ifndef GraphDisplacedVertexFinder_h
#define GraphDisplacedVertexFinder_h

#include "classes/DelphesModule.h"
#include "classes/DelphesClasses.h"
#include "classes/DelphesFactory.h"
#include "TVector3.h"
#include "TVectorD.h"
#include "TMatrixDSym.h"
#include "TMatrixD.h"
#include <unordered_map>
#include <utility>
#include <vector>
#include <optional>
#include <unordered_set>

struct Cluster {
    std::vector<const Candidate*> tracks;     // Tracks in this cluster
    TVectorD fittedPos;                  // Fitted vertex position
    TMatrixDSym fittedCov;             // Covariance matrix of the vertex fit
    double chi2 = 1e9;                        // Total chi2 (can be weighted)
    int ndf = 0;                              // Degrees of freedom 
    std::vector<double> trackChi2;            // Per-track chi2
    std::vector<double> trackWeights;         // Per-track weights in fit
    std::vector<double> trackPhases;          // Per-track phases
    double fittedTime;                    // Fitted time
    double fittedTimeErr;                 // Fitted time error
    int nactive;

    Cluster() : fittedCov(3), fittedPos(3) { fittedCov.ResizeTo(3,3); fittedCov.Zero(); }
};

struct FitResult{
    std::vector<const Candidate*> tracks;
    TVectorD fittedPos;
    TMatrixDSym fittedCov;
    std::vector<double> trackWeights;
    std::vector<double> trackPhases;
    std::vector<double> trackChi2;
    double chi2;
    int ndf;
    double fittedTime;
    double fittedTimeErr;
    int nactive;

    FitResult() : fittedCov(3), fittedPos(3) {fittedCov.ResizeTo(3,3); fittedCov.Zero();}
};


struct FitOpts {
  // IRLS sigmoid
  bool   useWeights   = true;   // turn off for N=2/3 fits in graph building
  double c0           = 9.0;    // sigmoid mid
  double beta         = 2.0;    // sigmoid slope
  double sigma_floor  = 0.01;    // floor on seed spread (mm)
  double weightCut    = 0.3;    // weight cut below which a track does not contribute to the chi2
  double wEps         = 1e-6;   // weight floor
  double weightActive = 1e-2;   // weight cut below which a track is deactivated


  // Seeding & constraint
  bool   useSelfSeeding     = true;
  bool   useBeamConstraint  = false;

  //Iterations
  int    maxIter      = 50;      //Maximum inner iterations
  int    maxIRLS      = 50;      //Maximum IRLS iterations

  //Step clamps and backtracking
  double absCap       = 500.0;   // absolute cap for a single phase step (mm)
  double frac_turn    = 0.75;     // clamp: fraction of helix turn
  int    btMax       = 10;        // backtracking steps

  //Convergence
  double sTol         = 1e-4;    // stop if phase change |Δs| < sTol (mm)
  double vTol         = 1e-4;    // stop if vertex movement |Δv| < vTol (mm)
  double wTol         = 1e-2;    // stop if weight change |Δw| < wTol

  //Absolute floor
  double dEps = 1e-12;

  //timing
  bool useTiming = false;
};


struct PairEdge {
  size_t u, v;           // track indices (u < v)
  double chi2;           // minimized 2-track chi2 (from GetPairFit_guarded)
  double w;              // edge score used later 
};

struct GraphOpts {
    double chi2PairCut = 9.0;
    double chi2TripCut = 9.0;
    double chi2AssociationMax = 9.0;
    double weightCut = 0.3;
    double bridgeCut = 3.0;

    int minSupport = 1;
    int kNN = -1;
    bool salvage_pass = false;

};



struct PairKey {
  const Candidate* a;
  const Candidate* b;
  PairKey(const Candidate* x=nullptr, const Candidate* y=nullptr){
    if (x < y) { a=x; b=y; } else { a=y; b=x; }
  }
  bool operator==(const PairKey& o) const { return a==o.a && b==o.b; }
};
struct PairKeyHash {
  size_t operator()(const PairKey& k) const noexcept {
    auto h1 = std::hash<const void*>{}(k.a);
    auto h2 = std::hash<const void*>{}(k.b);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
  }
};

struct TripKey {
  const Candidate* a;
  const Candidate* b;
  const Candidate* c;
  TripKey(const Candidate* x=nullptr, const Candidate* y=nullptr, const Candidate* z=nullptr){
    // canonical sort by pointer value
    const Candidate* v[3] = {x,y,z};
    std::sort(v, v+3);
    a=v[0]; b=v[1]; c=v[2];
  }
  bool operator==(const TripKey& o) const { return a==o.a && b==o.b && c==o.c; }
};
struct TripKeyHash {
  size_t operator()(const TripKey& k) const noexcept {
    auto h1 = std::hash<const void*>{}(k.a);
    auto h2 = std::hash<const void*>{}(k.b);
    auto h3 = std::hash<const void*>{}(k.c);
    // simple mix
    size_t h = h1;
    h ^= h2 + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
    h ^= h3 + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
    return h;
  }
};

struct ViewEntry {
  bool hasTime=false;
  TVectorD p;          // (D,phi0,C,z0,ct)
  TMatrixDSym Cp;      // 5x5
  double n, omega, kappa, tau, sfirst, sexit, texit, sigma_t, sigma_t2, mass, mom, beta, R, xc, yc;
  ViewEntry() : p(5), Cp(5) {p.Zero(); Cp.ResizeTo(5,5); Cp.Zero();}
};

struct PairFit {
  bool ok=false;
  TVectorD v;            // 3
  TMatrixDSym Cv;        // 3x3
  double chi2=0.0;
  double l1=0.0, l2=0.0;
  bool fail_track1=false, fail_track2=false;   // phase/metric/fh-guard culprit flags
  bool fail_firsthit=false;                    // explicit first-hit guard failure

  PairFit() : v(3), Cv(3) { v.Zero(); Cv.ResizeTo(3,3); Cv.Zero(); }
};


struct TripFit {
  bool ok=false;
  TVectorD v;          // 3
  TMatrixDSym Cv;      // 3x3
  double chi2=0.0;     // sum of plane χ²
  // pair validity mask: (ab,ac,bc)
  bool pair_ab=false, pair_ac=false, pair_bc=false;
  // if triplet phase/PD/solve failed, note culprits
  bool fail_a=false, fail_b=false, fail_c=false;
  double l1 = 0.0, l2 = 0.0, l3 = 0.0;
  TripFit() : v(3), Cv(3) { v.Zero(); Cv.ResizeTo(3,3); Cv.Zero(); }
};

struct EventCache {
  std::unordered_map<const Candidate*, ViewEntry> view;
  std::unordered_map<PairKey,  PairFit, PairKeyHash>  pairFit;
  std::unordered_map<TripKey,  TripFit, TripKeyHash>  tripFit;
  void clear(){ view.clear(); pairFit.clear(); tripFit.clear(); }
};

class GraphDisplacedVertexFinder : public DelphesModule {

public:
  GraphDisplacedVertexFinder();
  ~GraphDisplacedVertexFinder();

  void Init() override;
  void Process() override;
  void Finish() override;

private:

  

  mutable EventCache m_ecache;
  FitOpts fFitOpts;
  GraphOpts fGraphOpts;
  
  
  bool fVerbose = false;

  // Input/output arrays
  TObjArray* fInputArray{nullptr};            // Input track or eflow array
  TObjArray* VertexArray{nullptr};            // Output displaced vertex candidates
  TObjArray* EFlowArray{nullptr};             // Output eflow collection (displaced (with possible momentum correction), prompt, unclaimed, neutral)
  TObjArray* PVArray{nullptr};                // Output primary vertex
  // Iterator for input tracks
  TIterator* fItTrack{nullptr};

  std::vector<Cluster> fClusters;
  std::vector<const Candidate*> fDisplacedTracks;
  std::vector<const Candidate*> fPromptTracks;
  std::vector<const Candidate*> fSoftTracks;

  
  //Track preselection
  double fMinsDisp = 5.0;
  double fMinD0 = 0.1;
  double fMinZ0 = 0.1;
  double fMinTrackPT = 0.5;
  double fMinSeedPT = 1.0;
  double fMinSeedIPSig = 5.0;
  
  //Fit Primary Vertex
  double fBeamSpotSigmaX = 0.01;
  double fBeamSpotSigmaY = 0.01;
  double fBeamSpotSigmaZ = 1.0;
  TVectorD fBeamPos;
  TMatrixDSym fBeamCov;
  double fPVChi2NDFMax = 9.0;
  double fPVCutChi2 = 9.0;
  bool fUsePVCut = true;
  TVector3 fPVPos;
  TMatrixDSym fPVCov;
  double fPVtime = 0.0;
  double fPVtimeErr = 0.0;
  bool fHasPV = false;

  //Fitter Settings
  double fc0 = 9.0;
  double fbeta = 2.0;
  
  
  // Fitter Cuts for graph building
  double fchi2PairCut = 9.0;
  double fchi2TripCut = 9.0;
  double fweightCut = 0.3;
  double fBridgeCut = 3.0;
  
  // Graph building
  int fMinTracks = 2;
  bool fRequireSeed = false;
  int fSeedSelector = -1;
  bool fCorrectMomenta = true;
  int fkNN = -1;
  int fMinSupport = 0;
  bool fUsePairGuard = true;
  bool fPruneBridges = true;

  // Vertex Quality Cuts
  double fMTChi2NDFMax = 9.0;
  
  //Leftover Assignments
  double fChi2AssociationMax = 4.0;

  //Timing Layer
  double fTimingR = 2.25; // Magnetic Field radius
  double fTimingZ = 2.5; // Magnetic Field half-length
  double fTimeVariance = 4.0; //Slack for additional timing variance to be considered "well-timed"
  bool fUseTiming = true;
  double fTimeGate = 9.0;
  
private:
  void BeginEvent(){ m_ecache.clear(); } // call once per Delphes event
  void ClearCaches();
  
  // Store Track Parameters in Struct and cache for faster access
  const ViewEntry& viewOf(const Candidate* t) const{
    auto it = m_ecache.view.find(t);
    if (it != m_ecache.view.end()) return it->second;
    ViewEntry e; e.p = trkPars(t); e.Cp = trkCov(t); 
    const double ct=e.p(4), C=e.p(2), n=std::sqrt(1.0+ct*ct);
    e.n=n; e.omega=C/n; e.kappa=2.0*C/(n*n); e.tau=-2.0*C*ct/(n*n);

    e.texit = t->Position.T();
    e.sigma_t = t->ErrorT;
    e.mass = t->Momentum.M();
    e.mom = t->Momentum.P();
    e.beta = t->Momentum.Beta();
    e.R = 1.0/(2.0*C);
    e.xc = -(e.p(0) + e.R)*std::sin(e.p(1));
    e.yc = (e.p(0) + e.R)*std::cos(e.p(1));
    return m_ecache.view.emplace(t, std::move(e)).first->second;
    }

    //non-const version to avoid const_cast
  ViewEntry& ViewOf(const Candidate* t) const{
    auto it = m_ecache.view.find(t);
    if (it != m_ecache.view.end()) return it->second;
    ViewEntry e; e.p = trkPars(t); e.Cp = trkCov(t); 
    const double ct=e.p(4), C=e.p(2), n=std::sqrt(1.0+ct*ct);
    e.n=n; e.omega=C/n; e.kappa=2.0*C/(n*n); e.tau=-2.0*C*ct/(n*n);
    // e.sexit = t->L;
    e.texit = t->Position.T();
    e.sigma_t = t->ErrorT;
    e.mass = t->Momentum.M();
    e.mom = t->Momentum.P();
    e.beta = t->Momentum.Beta();
    e.R = 1.0/(2.0*C);
    e.xc = -(e.p(0) + e.R)*std::sin(e.p(1));
    e.yc = (e.p(0) + e.R)*std::cos(e.p(1));
    return m_ecache.view.emplace(t, std::move(e)).first->second;
    }

  // Arc Length Kinematics
  TVectorD trkX(const ViewEntry& tv, double s) const;
  double trkT(const ViewEntry& tv, double s) const;
  TVectorD trkdXds(const ViewEntry& tv, double s) const;
  TVectorD trkd2Xds2(const ViewEntry& tv, double s) const;
  TMatrixD trkJx(const ViewEntry& tv, double s) const;
  TMatrixDSym trkW(const ViewEntry& tv, double s) const;
  TMatrixDSym trkCx(const ViewEntry& tv, double s) const;
  TVectorD trkPars(const Candidate* t) const;
  TMatrixDSym trkCov(const Candidate* t) const;

  //Helpers
  bool checkPosDef(const TMatrixD& Mat) const;
  double dotD(const TVectorD& a, const TVectorD& b) const;
  bool SolveSPD(const TMatrixDSym& H, const TMatrixD& rhs, TMatrixD& sol) const;
  bool InvertSPD(const TMatrixDSym& H, TMatrixD& Hinv) const;
  bool SolveSPD(const TMatrixDSym& H, const TVectorD& rhs, TVectorD& sol) const;
  bool InvertSPD(const TMatrixDSym& H, TMatrixDSym& Hinv) const;
  TMatrixDSym CovPerigee_mm_to_m(const TMatrixDSym& Cmm) const;
  double sigmoid(double rho, double c0, double beta) const;

  //Fitters
  bool FindTimingExitAndVariance(
    const ViewEntry& tv,
    double& s_exit,               // [out] phase at timing surface
    double& sigma_t_eff2          // [out] effective time variance to use in 4D residual
) const;
  double GetPocaT2V(const ViewEntry& trk, 
            const TVectorD& vertex,
            double s0,                 
            int maxIter,
            double tol) const;

  std::pair<double, double> GetPocaT2T(const ViewEntry& trk1,
                                       const ViewEntry& trk2, double* s1_0, double* s2_0,
                                       int maxIter, double tol) const;
  void GetMetricPOCAT2V(const ViewEntry& tv,
                                             const TVectorD& v, 
                                             const TMatrixDSym& Cv,
                                             double& s_out,
                                             double& chi2) const;

  bool EuclideanSeed(const std::vector<const ViewEntry*>& tvs,
                    std::vector<double>& s_io,
                    std::vector<double>* w_io,
                    TVectorD& mu_out,               
                    TMatrixDSym* cov_out,
                    const FitOpts& opts) const;
  bool SeedFromTriplets(
    const std::vector<const Candidate*>& tracks,
    const FitOpts& opts,
    TVectorD& v_seed, TMatrixDSym& C_seed,
    double chi2TripCut,         // e.g. fChi2TripCut 
    int    maxTriplets         
) const;

    bool CoupledMetricFit(const std::vector<const ViewEntry*>& tvs,
                        std::vector<double>& s_io,
                        std::vector<double>* w_io,
                        std::vector<double>& chi2i_out,
                        double& chi2_out,
                        TVectorD& vstar_io,
                        double& vt_io,
                        TMatrixDSym& cov_out,
                        double& vart_out,
                        TVectorD* v_prior,
                        TMatrixDSym* cov_prior,
                        const FitOpts& opts) const;

  bool PreGatePair(
    const ViewEntry& t1, const ViewEntry& t2,
    double epsXY_mm   = 1.0
    ) const;

  const PairFit& GetPairFit(const Candidate* a,
                            const Candidate* b,
                            bool   usePairGuard) const;

  const PairFit& GetPairFit_guarded(const Candidate* a, 
    const Candidate* b) const;

    bool BlueCombine(const std::vector<TVectorD>& vs,
                    const std::vector<TMatrixDSym>& Cs,
                    TVectorD& v_out, TMatrixDSym& C_out) const;

  const TripFit& GetTripFit(
    const Candidate* a,
    const Candidate* b,
    const Candidate* c,
    const FitOpts&          opts,       
    double                 chi2PairCut) const; 

    FitResult RobustFit(
        const std::vector<const Candidate*>& tracks,
        const FitOpts& opts,
        const TVectorD* vseedOpt) const;

    FitResult RobustFit(Cluster& cl, const FitOpts& opts) const;
    void RobustFitInPlace(Cluster& cl, const FitOpts& opts) const;

    void PolishOrReconcile(
    std::vector<Cluster>&               clusters,
    const std::vector<const Candidate*>& displacedTracks,
    bool refitTracks) const;




  //Fitters: Refit Perigee Track Parameters
  inline void RefreshDerived(ViewEntry& e) const;

  bool KalmanUpdateTrackAtVertex3D(
    ViewEntry&         tv,
    double             s,
    const TVectorD&    v_meas,
    const TMatrixDSym& R) const;

    void RefitTracksToVertex(Cluster& cluster) const; 

  TLorentzVector CorrectedTrackMomentumAtVertex(
    const Candidate* trk, const TVectorD& vertex, double* s_used) const;




  //Graph Building
  std::vector<PairEdge> BuildScoredPairGraph(const std::vector<const Candidate*>& trk, double chi2PairCut) const;
  std::vector<std::vector<std::pair<size_t,double>>> AdjacencyFromPairEdges(const std::vector<PairEdge>& E, size_t N) const;
  
  void BuildTriangleSupportWithTriplets(
    const std::vector<std::vector<size_t>>& adj,
    const std::vector<const Candidate*>& tracks,
    double chi2PairCut,
    double chi2TripletMax,
    double condMax,
    std::vector<int>& triSup_out,
    std::vector<double>& edgeChi2Min) const;

  void ApplyMutualKNNPruneWeighted(std::vector<std::vector<std::pair<size_t,double>>>& adjW, int K) const;

  std::vector<std::vector<size_t>> StripWeights(const std::vector<std::vector<std::pair<size_t,double>>>& adjW) const;
  
  std::vector<std::vector<size_t>> ConnectedComponents(
    const std::vector<std::vector<size_t>>& adj) const;

  void PruneUnsupportedBridges(
  std::vector<std::vector<size_t>>& adj,
  const std::vector<std::vector<std::pair<size_t,double>>>& /*adjW*/,
  const std::vector<int>& triSup,
  int sMin,
  double strongChi2Cut,
  const std::vector<double>* edgeChi2MinOpt 
) const;

  std::vector<std::pair<size_t,size_t>> GreedyMaxWeightMatching(
    const std::vector<std::vector<std::pair<size_t,double>>>& adjW,
    const std::vector<char>& allowed) const;
  
  std::vector<Cluster> GraphClusteringHybrid(std::vector<const Candidate*>& tracks, const GraphOpts& opts) const;

  std::vector<const Candidate*> SelectSeeds(
    const std::vector<const Candidate*>& tracks,
    int selectorType,
    double minSeedPT) const;

  // Fit the PV and sort tracks into prompt/displaced
  void FitPrimaryVertex();
  void ExportClusters();
  double ipSig_d0z0(const Candidate* trk) const;

  };
#endif