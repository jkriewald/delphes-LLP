/*
 *  Delphes: a framework for fast simulation of a generic collider experiment
 *  Copyright (C) 2012-2014  Universite catholique de Louvain (UCL), Belgium
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

 /**  \class GraphDisplacedVertexFinder
 * 
 * Find displaced vertices via graph-based clustering of tracks/eflow-tracks
 * Fit the vertex candidates with a coupled Gauss-Newton fit as outlined in arXiv:2510.XXXXXX
 * Input is a DelphesCollection of tracks (or eflow-tracks) that have a 5x5 covariance matrix (from TrackCovariance or TrackSmearing)
 * Output is a DelphesCollection of vertices and re-fitted tracks that belong to the DVs
 * 
 * If timing information is available (via TimeSmearing), vertex time is fitted and can be used as gates in the graph
 * 
 * \author Jonathan Kriewald 
 */

#include "modules/GraphDisplacedVertexFinder.h"
#include "classes/DelphesFactory.h"
#include "TVector3.h"
#include "TMatrixD.h"
#include "TDecompSVD.h"
#include "TMatrixDSym.h"
#include "TDecompLU.h"
#include "TDecompChol.h"
#include "TDecompBK.h"
#include <TMatrixDSymEigen.h>
#include <TMatrixDEigen.h>
#include <TMath.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <tuple>
#include <numeric>
#include <random>
#include <stack>
#include <deque>

//Clear caches

void GraphDisplacedVertexFinder::ClearCaches(){
    m_ecache.clear();
}

void GraphDisplacedVertexFinder::Finish(){
    ClearCaches();
    if (fItTrack) { delete fItTrack; fItTrack = nullptr; }
}

GraphDisplacedVertexFinder::GraphDisplacedVertexFinder() :
    fInputArray(nullptr),
    VertexArray(nullptr),
    EFlowArray(nullptr),
    fItTrack(nullptr),
    fMinsDisp(5.0),
    fMinD0(0.0),
    fMinZ0(0.0),
    fMinTrackPT(0.1),
    fMinSeedPT(1.0),

    fBeamSpotSigmaX(0.01),
    fBeamSpotSigmaY(0.01),
    fBeamSpotSigmaZ(1.0),
    fPVChi2NDFMax(9.0),
    fPVCutChi2(9.0),
    
    fchi2PairCut(9.0),
    fchi2TripCut(27.0),
    fweightCut(0.3),

    fMinTracks(2),
    fSeedSelector(-1),
    fCorrectMomenta(true),
    fkNN(-1),
    fMinSupport(0),
    fUsePairGuard(true),
    fPruneBridges(true),

    fMTChi2NDFMax(9.0),

    fChi2AssociationMax(4.0)
{
    fPVPos = TVector3(0.0, 0.0, 0.0);
    fPVCov.ResizeTo(3,3); fPVCov.Zero();
}

GraphDisplacedVertexFinder::~GraphDisplacedVertexFinder(){
    ClearCaches();
    if (fItTrack) { delete fItTrack; fItTrack = nullptr; }
}

void GraphDisplacedVertexFinder::Init(){
    // --- IO arrays ---
    fInputArray = ImportArray(GetString("InputArray", "TrackFinder/Tracks"));
    VertexArray = ExportArray(GetString("VertexOutputArray", "DisplacedVertices"));
    EFlowArray  = ExportArray(GetString("EFlowOutputArray",  "EFlowOut"));
    PVArray     = ExportArray(GetString("PVOutputArray", "PrimaryVertex"));

    // Iterator for input tracks
    if (fItTrack) { delete fItTrack; fItTrack = nullptr; }
    fItTrack = fInputArray ? fInputArray->MakeIterator() : nullptr;

    // Track Preselection
    fMinsDisp = GetDouble("MinTrackIPSig", 3.0); // Impact parameter Significance
    fMinD0    = GetDouble("MinD0", 0.0);    // D0
    fMinZ0    = GetDouble("MinZ0", 0.0);    // DZ
    fMinTrackPT = GetDouble("MinTrackPT", 0.1); // Min Track PT

    //Primary Vertex
    fBeamSpotSigmaX = GetDouble("BeamSpotSigmaX", 0.1); // transverse beam spot uncertainty mm
    fBeamSpotSigmaY = GetDouble("BeamSpotSigmaY", 0.1); // transverse beam spot uncertainty mm
    fBeamSpotSigmaZ = GetDouble("BeamSpotSigmaZ", 1.0); // longitudinal beam spot uncertainty mm
    fPVChi2NDFMax = GetDouble("PVChi2NDFMax", 9.0); // Max Chi2/NDF for primary vertex fit
    fUsePVCut = GetBool("UsePVCut", true);
    fPVCutChi2 = GetDouble("PVCutChi2", 9.0); // Chi2 cut for track association to PV

    // Clustering
    fSeedSelector = GetInt("SeedSelector", 0); // 0: Min PT, 1: higher IP significance, 2: high pT lepton
    fMinSeedPT = GetDouble("MinSeedPT", 1.0); // Min Seed PT (for a Seed Track)
    fMinSeedIPSig = GetDouble("MinSeedIPSig", 5.0); // Min impact parameter significance
    fMinTracks = GetInt("MinTracks", 2); // Min number of tracks in a cluster

    fkNN = GetInt("kNN", -1); // kNN for pruning -1 means no pruning
    fMinSupport = GetInt("MinSupport", 0); // Min triplet support for pruning
    fUsePairGuard = GetBool("UsePairGuard", true); // Use first hit guard for pruning
    fPruneBridges = GetBool("PruneBridges", true); // Prune bridges

    fchi2PairCut = GetDouble("chi2PairCut", 9.21); // Chi2 cut for pairs 3 sigma
    fchi2TripCut = GetDouble("chi2TripletCut", 9.21); // Chi2 cut for triplets 3 sigma
    fweightCut = GetDouble("weightCut", 0.1); // Weight cut for tracks
    fMTChi2NDFMax = GetDouble("Chi2NDFMax", 9.21); // Max Chi2/NDF for vertex fit
    fBridgeCut = GetDouble("BridgeCut", 3.0); // Chi2 cut for bridges
    

    // Fitter Settings
    fUseTiming = GetBool("UseTiming", false);
    fTimeGate  = GetDouble("TimeGate", 9.21);
    fTimeVariance = GetDouble("TimeVarianceCap", 4.0);
    
    fc0 = GetDouble("chi2_0", 9.21); // Sigmoid mid
    fbeta = GetDouble("beta", 2.0); // Sigmoid width  
    fFitOpts.useSelfSeeding = GetBool("UseSelfSeeding", true); //Switch off for ablations only


    fFitOpts.beta = fbeta;
    fFitOpts.weightCut = fweightCut;
    fFitOpts.c0 = fc0;
    fFitOpts.useTiming = fUseTiming;

    //Greedy Assign
    fChi2AssociationMax = GetDouble("Chi2AssociationMax", 4.0); // Max Chi2 to assign leftover tracks to closest vertex
    
    //Refit track parameters and correct track momenta at the DV
    fCorrectMomenta = GetBool("CorrectMomenta", true); // Correct momenta of tracks in clusters

    fVerbose = GetBool("Verbose", true);

    fTimingR = GetDouble("TimingR", 2.25);
    fTimingZ = GetDouble("TimingZ", 2.5);

    fPVPos.SetXYZ(0.0, 0.0, 0.0);
    fPVCov.ResizeTo(3, 3); fPVCov.Zero();
    fPVCov(0,0) = fBeamSpotSigmaX*fBeamSpotSigmaX;
    fPVCov(1,1) = fBeamSpotSigmaY*fBeamSpotSigmaY;
    fPVCov(2,2) = fBeamSpotSigmaZ*fBeamSpotSigmaZ;
    fHasPV = false;

    fBeamPos.ResizeTo(3); fBeamPos.Zero();
    fBeamCov.ResizeTo(3,3); fBeamCov.Zero();

    fGraphOpts.kNN = fkNN;
    fGraphOpts.chi2PairCut = fchi2PairCut;
    fGraphOpts.chi2TripCut = fchi2TripCut;
    fGraphOpts.weightCut = fweightCut;
    fGraphOpts.minSupport = fMinSupport;
    fGraphOpts.chi2AssociationMax = fChi2AssociationMax;
    fGraphOpts.bridgeCut = fBridgeCut;

}



// natural turn scale: s_turn = 2π/|κ| with κ = 2C/n^2  
static inline double turn_scale(double C, double n) {
  const double tiny = 1e-14;
  if (std::abs(C) < tiny) return 1e12; // effectively straight
  const double kappa = 2.0*C/(n*n);
  return 2.0*M_PI/std::abs(kappa); // = M_PI * n^2/|C|
}

// Decide barrel vs disk; intersect forward; propagate covariance; add turn-ambiguity inflation.
// Need to find the track phase at which it intersects the timing layer for the timing model
// assumes a cylindrical timing layer (no ATLAS-like HGTD possible for now)
// Time coordinate has been smeared by TimeSmearing, but the position hasn't 
// TrackSmearing or TrackCovariance smear track parameters, but we don't have the smeared total arc length of the track 
// We find here the total arc length and assess the reliability of the derived timing measurement:
// If a track enters almost in parallel with the timing layer, the timing measurement is not reliable so we inflate the time variance:
// First, we propagate the covariance of the track to the timing layer 
// Second, we check if the track has several acceptable minima for the intersection (soft multi-loopers) and inflate the time variance accordingly
// This is a bit of a hack but it works for now to sort out tracks for which the s_exit reference is not reliable and their timing information isn't used downstream
// A real track fit should use the hit in the timing layer for the track fit so this is accommodating for that since we currently don't have a 6D track fit in delphes

// Decide barrel vs disk; intersect forward; propagate covariance; add turn-ambiguity inflation.
// Robust, forward-only, "first intersection wins" search 
// This is all a bit hacky right now, should be updated once we have 6D tracking in delphes and reliable references
bool GraphDisplacedVertexFinder::FindTimingExitAndVariance(
    const ViewEntry& tv,
    double& s_exit,               // [out] phase at timing surface
    double& sigma_t_eff2          // [out] effective time variance [mm^2]
) const
  {
    // ---------- config ----------
    const double gate_max_miss_to_hit = 80.0;   // mm; set <=0 to disable sanity gate
    const double tau_turn = 3.0;                // χ gate for turn-ambiguity scan
    const int    Kmax     = 5;                  // scan ±K turns
    const double gTolR    = 1e-3;               // mm^2 tolerance for barrel constraint
    const double gTolZ    = 1e-3;               // mm   tolerance for disk constraint

    // geometry (convert to mm)
    const double TimingR = fTimingR * 1e3;   // mm
    const double TimingZ = fTimingZ * 1e3;   // mm
    const double Rtim_mm = TimingR;
    const double Ztim_mm = TimingZ;
    bool isBarrel;
    

    // per-track
    const double beta   = tv.beta;
    const double sTurn  = std::max(1.0, turn_scale(tv.p(2), tv.n));  // guard
    const double sig_t1 = tv.sigma_t; // single-layer σ (mm)

    // pull helix parameters
    const double d0  = tv.p(0);
    const double p0  = tv.p(1);
    const double C   = tv.p(2);
    const double z0  = tv.p(3);
    const double ct  = tv.p(4);
    const double n   = tv.n;

    // tiny guards
    const double eps   = 1e-12;
    const double twoPi = 2.0*M_PI;

    // straight-line fallback if |C| ~ 0
    const bool nearlyStraight = (std::abs(C) < 1e-12);

    // ---------- DISKS: z = ±Ztim ----------
    auto try_disk = [&](double Zplane, double& s_out)->bool {
      const double denom = ct / n;                // z(s) = z0 + (ct/n) s
      if (std::abs(denom) < 1e-14) return false;  // parallel to disks
      double s = (Zplane - z0) / denom;           // may be negative
      if (s < 0.0) return false;                  // forward only
      // check cylinder bound at that s
      TVectorD X = trkX(tv, s);
      const double R = std::hypot(X(0), X(1));
      if (R <= Rtim_mm + 1e-6) { s_out = s; return true; }
      return false;
    };

    // ---------- BARREL: R(s) = Rtim ----------
    auto try_barrel = [&](double& s_out)->bool
    {
      if (nearlyStraight) {
        // Solve straight line (your C->0 limit): x = -d0 sin p0 + (s/n) cos p0; y = d0 cos p0 + (s/n) sin p0
        const double ax = std::cos(p0)/n,  bx = -d0*std::sin(p0);
        const double ay = std::sin(p0)/n,  by =  d0*std::cos(p0);
        // (ax s + bx)^2 + (ay s + by)^2 = Rtim^2  ->  (ax^2+ay^2)s^2 + 2(ax bx + ay by)s + (bx^2+by^2 - Rtim^2) = 0
        const double A = (ax*ax + ay*ay);                   // = 1/n^2
        const double B = 2.0*(ax*bx + ay*by);
        const double Cq= (bx*bx + by*by - Rtim_mm*Rtim_mm);
        const double disc = B*B - 4*A*Cq;
        if (disc < 0.0) return false;
        const double s1 = (-B - std::sqrt(disc)) / (2*A);
        const double s2 = (-B + std::sqrt(disc)) / (2*A);
        double s = 1e300;
        if (s1 >= 0.0) s = std::min(s, s1);
        if (s2 >= 0.0) s = std::min(s, s2);
        if (!std::isfinite(s) || s==1e300) return false;
        // z bound
        const double z = z0 + (ct/n)*s;
        if (std::abs(z) > Ztim_mm + 1e-6) return false;
        s_out = s; return true;
      }

      // General helix case:
      // Write x(s) = xc + A sin(th), y(s) = yc - A cos(th) with A = 1/(2C), th = p0 + 2(C/n)s.
      const double A = 1.0/(2.0*C);       // signed radius in xy param
      const double U = tv.xc;             // = -(d0 + A) sin p0
      const double V = tv.yc;             // =  (d0 + A) cos p0
      const double rc= std::hypot(U, V);

      // Solve R^2(s) = (U + A sin th)^2 + (V - A cos th)^2 = Rtim^2
      // => 2A rc sin(th - φ) = D,   with φ = atan2(V,U),  D = Rtim^2 - (rc^2 + A^2)
      const double D = Rtim_mm*Rtim_mm - (rc*rc + A*A);
      const double den = 2.0*A*rc;

      if (std::abs(den) < eps) {
        // Degenerate (circle concentric or infinitesimal): either no or infinite solutions -> treat as no robust root
        return false;
      }
      double y = D / den;
      if (y < -1.0 - 1e-12 || y > 1.0 + 1e-12) return false;
      y = std::clamp(y, -1.0, 1.0);

      const double phi = std::atan2(V, U);
      const double psi = std::asin(y);          // principal

      // Two base families of θ solutions:
      const double th_base1 = phi + psi;
      const double th_base2 = phi + (M_PI - psi);

      // Convert θ -> s: θ(s) = p0 + 2(C/n) s  ⇒  s(θ) = (n/(2C)) (θ - p0)
      auto theta_to_s_forward = [&](double th)->double {
        double s0 = (n/(2.0*C)) * (th - p0);
        // bring to first forward crossing by adding integer multiples of 2π in θ
        const double ds_2pi = (n/(2.0*C)) * twoPi;      // s increment per +2π in θ
        if (s0 < 0.0) {
          // smallest m such that s0 + m*ds_2pi ≥ 0
          const double m = std::ceil((-s0) / ds_2pi);
          s0 += m * ds_2pi;
        }
        return s0;
      };

      // generate up to two candidates (k=0 families), both mapped to minimal s>=0 using +2π wraps
      double s1 = theta_to_s_forward(th_base1);
      double s2 = theta_to_s_forward(th_base2);

      // pick earliest forward that also respects |z| ≤ Ztim
      double s_pick = 1e300;
      auto check_and_min = [&](double s)->void {
        if (s >= 0.0 && std::isfinite(s)) {
          const double z = z0 + (ct/n)*s;
          if (std::abs(z) <= Ztim_mm + 1e-6) s_pick = std::min(s_pick, s);
        }
      };
      check_and_min(s1);
      check_and_min(s2);

      if (s_pick==1e300) return false;
      s_out = s_pick;
      return true;
    };

    // try all three, keep the earliest s>=0
    double s_bar=0, s_zp=0, s_zm=0;
    bool ok_bar = try_barrel(s_bar);
    bool ok_zp  = try_disk(+Ztim_mm, s_zp);
    bool ok_zm  = try_disk(-Ztim_mm, s_zm);

    if (!ok_bar && !ok_zp && !ok_zm) return false;

    // choose first-forward
    s_exit = 1e300; isBarrel = false;
    auto consider = [&](bool ok, double s, bool barrel){
      if (!ok) return;
      if (s >= 0.0 && s < s_exit) { s_exit = s; isBarrel = barrel; }
    };
    consider(ok_bar, s_bar, true);
    consider(ok_zp,  s_zp,  false);
    consider(ok_zm,  s_zm,  false);

  // const bool isBarrel = C[best].isBarrel;
  // s_exit = C[best].s;

  // ---------- phase variance from transported spatial cov ----------
  TVectorD Xexit = trkX(tv, s_exit);
  TVectorD Vexit = trkdXds(tv, s_exit);
  TMatrixDSym Cx  = trkCx(tv, s_exit);

  TVectorD grad(3); grad.Zero();
  double dgs = 0.0;
  if (isBarrel) { grad(0)=2*Xexit(0); grad(1)=2*Xexit(1); dgs = 2.0*(Xexit(0)*Vexit(0)+Xexit(1)*Vexit(1)); }
  else          { grad(2)=1.0;                                   dgs = Vexit(2); }

  const double sigma_g2      = std::max(Cx.Similarity(grad), 1e-24);
  const double sigma_s_exit2 = sigma_g2 / std::max(dgs*dgs, 1e-24);

  // ---------- turn-ambiguity inflation ----------
  double ds_alt = 0.0;
  {
    TVectorD Xref = Xexit;
    for (int k=1; k<=Kmax; ++k){
      for (int sgn=-1; sgn<=1; sgn+=2){
        const double sc = s_exit + sgn*k*sTurn;
        // if (sc < 0.0 || sc > s_cap) continue;

        TVectorD Xc = trkX(tv, sc);
        TMatrixDSym Cc = trkCx(tv, sc), CcInv(3); CcInv.Zero();
        if (!InvertSPD(Cc, CcInv)) continue;

        TVectorD dX = Xc - Xref; double chi2_x = 0.0;
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) chi2_x += dX(i)*CcInv(i,j)*dX(j);

        // surface residual χ² and bounds
        double g_val; TVectorD grad_c(3); grad_c.Zero(); bool ok_surface=false;
        if (isBarrel){
          grad_c(0)=2*Xc(0); grad_c(1)=2*Xc(1);
          g_val = Xc(0)*Xc(0)+Xc(1)*Xc(1) - TimingR*TimingR;
          ok_surface = (std::abs(Xc(2)) <= TimingZ + 1e-3);
        } else {
          grad_c(2)=1.0;
          const double ZrefDisk = (Xexit(2)>=0 ? +TimingZ : -TimingZ);
          g_val = Xc(2) - ZrefDisk;
          ok_surface = (std::hypot(Xc(0),Xc(1)) <= TimingR + 1e-3);
        }
        if (!ok_surface) continue;

        const double sigma_g2_alt = std::max(Cc.Similarity(grad_c), 1e-24);
        const double chi2_tot = chi2_x + (g_val*g_val)/sigma_g2_alt;

        if (chi2_tot <= tau_turn*tau_turn) {
          ds_alt = std::min(std::max(ds_alt, std::abs(sc - s_exit)), 2.0*sTurn); // cap
        }
      }
    }
  }

  // ---------- effective time variance ----------
  const double prof = sig_t1 * sig_t1;            
  const double phas = sigma_s_exit2 / (beta*beta);
  const double turn = (ds_alt > 0.0) ? (ds_alt*ds_alt)/(beta*beta) : 0.0;
  sigma_t_eff2 = prof + phas + turn;
  if (!(sigma_t_eff2 >= 0.0) || !std::isfinite(sigma_t_eff2)) sigma_t_eff2 = 1e12;

  // std::cout<<"[timing] sigma_t_eff2 = "<<sigma_t_eff2
  //          <<"  (profile="<<prof<<", phase="<<phas<<", turn="<<turn<<")\n";

  return true;
}

void GraphDisplacedVertexFinder::Process() {
  ClearCaches();
  if (!fInputArray) return;

    int ntimed = 0;
    int ntot = 0;
    fItTrack->Reset();
    const Candidate* trk = nullptr;
    while ((trk = static_cast<const Candidate*>(fItTrack->Next()))){
      if (trk->Charge == 0) continue;
      ++ntot;
      ViewEntry& tv = ViewOf(trk); //Fill event-level track cache

      if (tv.sigma_t > 0.){ //If not, then TimeSmearing probably wasn't used and we shouldn't use timing here

        double sexit = 0;
        double sigma_t2 = 0;

        bool haveTiming = FindTimingExitAndVariance(tv, sexit, sigma_t2);
  
        if (!haveTiming) {tv.sigma_t2 = 1e12; tv.sexit = 0; tv.hasTime = false;} //Fallback if track was not timed or timing couldn't be established (intersection search failed)
        else {
          tv.sexit = sexit; 
          tv.sigma_t2 = sigma_t2; 
          if (sigma_t2 > tv.sigma_t * tv.sigma_t + fTimeVariance) {
            tv.hasTime = false; // track was likely mis-timed, don't use time information down-stram
          } 
          else{
            tv.hasTime = true; ++ntimed;
          }
        }
    } else{
       tv.sigma_t2 = 1e12; 
       tv.sexit = 0; 
       tv.hasTime = false; //Fallback if track was not timed
    }
  }

  // std::cout << "Found " << ntimed << " tracks with timing information out of " << ntot << " tracks" << std::endl;

  // Primary vertex
  FitPrimaryVertex();

  if (fSeedSelector >= 0) fRequireSeed = true;

  //   Initial Vertex Candidates
  if (fVerbose) std::cout<< "Running GraphClustering on " << fDisplacedTracks.size() << " tracks" << std::endl;
  std::vector<Cluster> initial_clusters = GraphClusteringHybrid(fDisplacedTracks, fGraphOpts);

  // Find tracks not assigned to initial clusters
  std::vector<const Candidate*> initial_tracks;
  std::vector<const Candidate*> leftover_tracks;

  for (size_t i = 0; i < initial_clusters.size(); i++) {
    for (size_t j = 0; j < initial_clusters[i].tracks.size(); j++) {
      const Cluster& c = initial_clusters[i];
      const Candidate* t = c.tracks[j];
      initial_tracks.push_back(t);
    }
  }

  for (size_t i = 0; i < fDisplacedTracks.size(); i++) {
    const Candidate* t = fDisplacedTracks[i];
    if (std::find(initial_tracks.begin(), initial_tracks.end(), t) == initial_tracks.end()) {
      leftover_tracks.push_back(t);
    }
  }
  // One salvage round with tighter gates, helpful if a fit failed because multi-modality was not detected
  if (leftover_tracks.size() >= fMinTracks){
    if (fVerbose) std::cout<< "Running Salvage GraphClustering on " << leftover_tracks.size() << " leftover tracks" << std::endl;

    GraphOpts tight_opts = fGraphOpts;
    tight_opts.chi2PairCut = std::min(1.0, tight_opts.chi2PairCut *= 0.8);
    tight_opts.chi2TripCut = std::min(1.0, tight_opts.chi2TripCut *= 0.8);
    tight_opts.chi2AssociationMax = std::min(1.0, tight_opts.chi2AssociationMax *= 0.8);
    tight_opts.bridgeCut = std::min(1.0, tight_opts.bridgeCut *= 0.8);
    tight_opts.minSupport += 1;
    tight_opts.salvage_pass = true;

    std::vector<Cluster> leftover_clusters = GraphClusteringHybrid(leftover_tracks, tight_opts);
    if (fVerbose) std::cout << "Salvage pass found " << leftover_clusters.size() << " additional clusters" << std::endl;
    initial_clusters.insert(initial_clusters.end(), leftover_clusters.begin(), leftover_clusters.end());
  }

  //    Clusters have hard track assignments already 
  //    Assign leftover single tracks to best cluster
  //    Refit track perigee parameters to the DVs with single Kalman update (will still be perigees in the end)
  bool refitTracks = fCorrectMomenta;
  PolishOrReconcile(initial_clusters, fDisplacedTracks, refitTracks);
  
  fClusters = initial_clusters;

  if (fVerbose) std::cout<< "Found " << fClusters.size() << " clusters" << std::endl;

  // Export
  ExportClusters();
  ClearCaches();
}



void GraphDisplacedVertexFinder::ExportClusters()
{
  if (!VertexArray || !EFlowArray) return;

  std::unordered_set<const Candidate*> usedOriginal;
  usedOriginal.reserve(fDisplacedTracks.size());

  for (int i = 0; i < fClusters.size(); i++) 
  { Cluster cluster = fClusters[i];
    if (fVerbose) std::cout<< "Exporting cluster i = " << i << " with " << cluster.tracks.size() << " tracks, chi2 = " << cluster.chi2 << ", effective ndf = " << cluster.ndf << std::endl;
    if (cluster.ndf <=0 || cluster.nactive < fMinTracks || cluster.chi2/std::max(1, cluster.ndf) > fMTChi2NDFMax) continue;
    if (cluster.tracks.size() < 2) continue;

    // Create DV candidate
    Candidate* vtx = fFactory->NewCandidate();

    const TVectorD& dv = cluster.fittedPos;
    const TVector3 dv3 = TVector3(dv(0), dv(1), dv(2));
    double vT = cluster.fittedTime;
    double vTErr = cluster.fittedTimeErr;

    double XErr, YErr, ZErr;
    if (cluster.fittedCov.GetNrows() == 3) {
      XErr = std::sqrt(std::max(0.0, cluster.fittedCov(0,0)));
      YErr = std::sqrt(std::max(0.0, cluster.fittedCov(1,1)));
      ZErr = std::sqrt(std::max(0.0, cluster.fittedCov(2,2)));
    } else {
      XErr = YErr = ZErr = 0.0;
    }
    vtx->Position.SetXYZT(dv3.X(), dv3.Y(), dv3.Z(), vT);
    vtx->PositionError.SetXYZT(XErr, YErr, ZErr, vTErr);

    vtx->ErrorXY = cluster.fittedCov(0,1);
    vtx->ErrorXZ = cluster.fittedCov(0,2);
    vtx->ErrorYZ = cluster.fittedCov(1,2);

    vtx->InitialPosition.SetXYZT(fPVPos.X(), fPVPos.Y(), fPVPos.Z(), fPVtime);

    // Displacement wrt PV + uncertainties
    TVector3 disp = dv3 - fPVPos;
    const double Lxy  = disp.Perp();
    const double Lz   = disp.Z();
    const double Lxyz = disp.Mag();
    vtx->Lxy  = Lxy;
    vtx->Lz   = Lz;
    vtx->Lxyz = Lxyz;

    double varLxy=0, varLz=0, varLxyz=0;
    if (cluster.fittedCov.GetNrows()==3) {
      const TMatrixDSym& C = cluster.fittedCov;
      const double dx=disp.X(), dy=disp.Y(), dz=disp.Z();

      //Gaussian Error propagation
      if (Lxy>0) {
        const double d2 = std::max(1e-12, dx*dx+dy*dy);
        varLxy = (dx*dx*C(0,0) + 2*dx*dy*C(0,1) + dy*dy*C(1,1)) / d2;
      }
      varLz = C(2,2);

      if (Lxyz>0) {
        const double d2 = std::max(1e-12, dx*dx+dy*dy+dz*dz);
        varLxyz = ( dx*dx*C(0,0) + dy*dy*C(1,1) + dz*dz*C(2,2)
                  + 2*dx*dy*C(0,1) + 2*dx*dz*C(0,2) + 2*dy*dz*C(1,2) ) / d2;
      }
    }
    vtx->ErrorLxy  = (varLxy>0)  ? std::sqrt(varLxy)  : 0.0;
    vtx->ErrorLz   = (varLz>0)   ? std::sqrt(varLz)   : 0.0;
    vtx->ErrorLxyz = (varLxyz>0) ? std::sqrt(varLxyz) : 0.0;

    // χ²/ndf and PV
    const int ndf = std::max(1, cluster.ndf);
    vtx->vChi2NDF = cluster.chi2 / ndf;
    vtx->vNDF     = cluster.ndf;


    // Track loop
    vtx->NTracks = 0;
    int qsum = 0, nEl=0, nMu=0, nChHad=0;
    TLorentzVector sumP4(0,0,0,0);


    for (size_t i=0;i<cluster.tracks.size();++i) {
      if (cluster.trackWeights[i] < fweightCut) continue;
      const Candidate* orig = cluster.tracks[i];
      const ViewEntry& tv = viewOf(orig);

      // Clone eflow tracks to not mess with other delphes modules
      Candidate* c = static_cast<Candidate*>(const_cast<Candidate*>(orig)->Clone());

      // Write refitted perigee parameters if present (values already in mm)
      
        const TVectorD& p = tv.p;
        if (p.GetNrows()==5) {
          c->D0       = p(0);
          c->Phi      = p(1);
          c->C        = p(2);
          c->DZ       = p(3);
          c->CtgTheta = p(4);
        }
        // And refitted 5x5 covariance, converting back to meters
        const TMatrixDSym& Cmm = tv.Cp;
        if (Cmm.GetNrows()==5) {
          c->TrackCovariance = CovPerigee_mm_to_m(Cmm); // store in Delphes' native units which are for whatever reason in m
        }
      

      
      c->InitialPosition.SetXYZT(dv3.X(), dv3.Y(), dv3.Z(), vT);

      TLorentzVector p4 = fCorrectMomenta ? CorrectedTrackMomentumAtVertex(c, dv, nullptr) : c->Momentum;
      c->Momentum = p4;
      c->PT       = p4.Pt();
      c->Phi      = p4.Phi();
      c->P        = p4.P();
      c->Mass     = p4.M();

      vtx->AssociatedTracks.Add(c);
      ++vtx->NTracks;
      sumP4 += p4;

      const int apid = std::abs(c->PID);
      if (apid==11) ++nEl;
      if (apid==13) ++nMu;
      if (c->Charge!=0 && apid>13) ++nChHad;
      qsum += c->Charge;

      // also into EFlow
      EFlowArray->Add(c);

      // remember original so we don't duplicate into EFlow later
      usedOriginal.insert(orig);
    }
    vtx->SumPt = sumP4.Pt();
    vtx->Charge          = qsum;
    vtx->NElectrons      = nEl;
    vtx->NMuons          = nMu;
    vtx->NChargedHadrons = nChHad;

    // DV momentum / mass / geometry
    vtx->Momentum = sumP4;
    vtx->PT       = sumP4.Pt();
    vtx->P        = sumP4.P();
    vtx->Mass     = sumP4.M();

    vtx->ErrorT = cluster.fittedTimeErr;

    TVector3 dir = dv3 - fPVPos;
    double cosTheta = 1.0;
    if (dir.Mag()>0) {
      cosTheta = dir.Unit().Dot(sumP4.Vect().Unit());
    }
    vtx->CosThetaDVMom = cosTheta;

    const double pTmiss = (dir.Mag()>0) ? (sumP4.Vect().Cross(dir.Unit())).Mag() : 0.0;
    vtx->MassCorr = std::sqrt(std::max(0.0, sumP4.M2()) + pTmiss*pTmiss) + pTmiss;

    // boosts and ctau
    const double p = sumP4.P(), e = sumP4.E(), m = sumP4.M();
    const double beta = (e>0.0) ? p/e : 0.0;
    const double gamma = (m>0.0) ? e/m : 0.0;
    vtx->boostbeta  = beta;
    vtx->boostgamma = gamma;
    vtx->betagamma  = beta*gamma;
    vtx->ctau       = (vtx->betagamma>0.0) ? (Lxyz / vtx->betagamma) : 0.0;

    // Done
    VertexArray->Add(vtx);
  }

  // Add leftover (non-DV) tracks to EFlow but uncorrected
  if (fItTrack) fItTrack->Reset();
  Candidate* trk = nullptr;
  while (fItTrack && (trk = static_cast<Candidate*>(fItTrack->Next()))) {
    if (usedOriginal.find(trk) == usedOriginal.end()) {
      EFlowArray->Add(trk);
    }
  }
}

// Helper: pick a reasonable mass hypothesis (prefer the one carried by Delphes if set)
// Assumes that PID has already happened
static inline double massHypothesis(const Candidate* trk){
  if (trk && trk->Mass > 0.0) return trk->Mass;       // Delphes should have set this
  const int pdg = trk ? std::abs(trk->PID) : 0;
  if      (pdg == 11) return 0.000511;                // e
  else if (pdg == 13) return 0.105658;                // μ
  else if (pdg == 211) return 0.13957039;              // π+
  else if (pdg == 321) return 0.493677;                // K+
  else if (pdg == 2212) return 0.938272;               // p
  // light hadrons default (π±)
  return 0.13957039;
}

// Rotate three momentum so that it has the correct pointing from the vertex
// Only really relevant for soft tracks with pT < 1-2 GeV
// Only affects phi, not theta (or eta)
// Should improve kinematical resolution if used as input for jet clustering
TLorentzVector GraphDisplacedVertexFinder::CorrectedTrackMomentumAtVertex(
    const Candidate* trk, const TVectorD& vertex, double* s_used) const
{
  if (!trk) return TLorentzVector();  // fallback

  const ViewEntry& tv = viewOf(trk);

  double s = 0.0, chi2 = 0.0;
  TMatrixDSym cov(3); cov.Zero();
  GetMetricPOCAT2V(tv, vertex, cov, s, chi2);
  if (s_used) *s_used = s;

  const TVectorD tD = trkdXds(tv, s);
  TVector3 t(tD(0), tD(1), tD(2));
  const double t2 = t.Mag2();
  if (!(t2 > 0.0 && std::isfinite(t2))) return trk->Momentum;
  t *= (1.0 / std::sqrt(t2));


  const double pMag = trk->Momentum.Vect().Mag();
  if (!(pMag > 0.0 && std::isfinite(pMag))) return trk->Momentum;

  TVector3 pvec = t * pMag;
  double   E    = trk->Momentum.E();

  const double m = massHypothesis(trk);
  if (m > 0.0) E = std::sqrt(std::max(0.0, pMag*pMag + m*m));

  TLorentzVector out; out.SetPxPyPzE(pvec.X(), pvec.Y(), pvec.Z(), E);
  return out;
}



// ================== Arc Length Kinematics =====

// simple timing model, take s_exit and t_exit as references where the track intersects the timing layer
double GraphDisplacedVertexFinder::trkT(const ViewEntry& tv, double s) const {
    const double sexit = tv.sexit;
    const double texit = tv.texit;
    const double beta = tv.beta;

    double T =  texit - (sexit - s)/beta;

    return T;
}

TVectorD GraphDisplacedVertexFinder::trkX(const ViewEntry& tv, double s) const {
  TVectorD X(3); X.Zero();

  const double D0 = tv.p(0);
  const double p0 = tv.p(1);
  const double C  = tv.p(2);
  const double z0 = tv.p(3);
  const double ct = tv.p(4);
  const double n  = tv.n;

  const double u  = (C/n)*s;             // u = (C/n) s
  const double th = p0 + 2.0*u;          // θ(s) = p0 + 2u
  const double sp0 = std::sin(p0),  cp0 = std::cos(p0);
  const double sth = std::sin(th),  cth = std::cos(th);

  if (std::abs(C) > 1e-12) {
    X(0) = -D0*sp0 + 1.0/(2.0*C)*(sth - sp0);
    X(1) =  D0*cp0 - 1.0/(2.0*C)*(cth - cp0);
  } else {
    // C → 0 straight-line limit: x'=(cos p0)/n, y'=(sin p0)/n
    X(0) = -D0*sp0 + (s/n)*cp0;
    X(1) =  D0*cp0 + (s/n)*sp0;
  }
  X(2) = z0 + (ct/n)*s;

  return X;
}

// First derivative (tangent)
TVectorD GraphDisplacedVertexFinder::trkdXds(const ViewEntry& tv, double s) const {
  TVectorD t(3); t.Zero();

  const double p0 = tv.p(1);
  const double C  = tv.p(2);
  const double ct = tv.p(4);
  const double n  = tv.n;

  const double u  = (C/n)*s;
  const double th = p0 + 2.0*u;

  t(0) = std::cos(th)/n;
  t(1) = std::sin(th)/n;
  t(2) = ct/n;
  return t;
}

// second derivative (curvature)
TVectorD GraphDisplacedVertexFinder::trkd2Xds2(const ViewEntry& tv, double s) const {
  TVectorD a(3); a.Zero();

  const double p0 = tv.p(1);
  const double C  = tv.p(2);
  const double n  = tv.n;

  const double u  = (C/n)*s;
  const double th = p0 + 2.0*u;
  const double kappa = 2.0*C/(n*n);      // κ consistent with 2u convention

  a(0) = -kappa * std::sin(th);
  a(1) =  kappa * std::cos(th);
  a(2) =  0.0;
  return a;
}

// Jacobian
TMatrixD GraphDisplacedVertexFinder::trkJx(const ViewEntry& tv, double s) const {
  TMatrixD J(3,5); J.Zero();

  const double D0 = tv.p(0);
  const double p0 = tv.p(1);
  const double C  = tv.p(2);
  const double z0 = tv.p(3);
  const double ct = tv.p(4);
  const double n  = tv.n;                   // sqrt(1+ct^2)

  const double u  = (C/n)*s;                // u = (C/n) s
  const double A  = p0 + u;                 // A = p0 + u

  const double s_over_n = s / n;
  const double ct_over_n3 = ct / (n*n*n);

  // small-angle safe sinc and its derivative
  auto sinc = [](double x)->double {
    double ax = std::fabs(x);
    if (ax < 1e-6) { // 1 - x^2/6 + x^4/120
      double x2 = x*x;
      return 1.0 - x2/6.0 + (x2*x2)/120.0;
    }
    return std::sin(x)/x;
  };
  auto dsinc = [](double x)->double {
    double ax = std::fabs(x);
    if (ax < 1e-6) { // (x cos x - sin x)/x^2 series: -x/3 + x^3/30 ...
      // Using series up to x^3 is plenty here
      return -x/3.0 + (x*x*x)/30.0;
    }
    // exact formula
    return (x*std::cos(x) - std::sin(x)) / (x*x);
  };

  const double S      = s_over_n * sinc(u);
  const double dS_dC  = (s_over_n) * dsinc(u) * (s_over_n);               // (s/n)^2 * sinc'(u)
  const double dS_dct = - (s * ct) / (n*n*n) * ( sinc(u) + (u / n) * dsinc(u) );
  const double dA_dC  = s_over_n;
  const double dA_dct = - (C * s * ct) / (n*n*n);

  const double sinp0 = std::sin(p0),       cosp0 = std::cos(p0);
  const double sinA  = std::sin(A),        cosA  = std::cos(A);

  // x = -D0 sin p0 + S cos A
  // y =  D0 cos p0 + S sin A
  // z =  z0 + (ct/n) s

  // ∂/∂D0
  J(0,0) = -sinp0;       // ∂x/∂D0
  J(1,0) =  cosp0;       // ∂y/∂D0
  // ∂/∂p0
  J(0,1) = -D0*cosp0 - S*sinA;  // S doesn't depend on p0; A does with dA/dp0=1
  J(1,1) = -D0*sinp0 + S*cosA;
  // ∂/∂C
  J(0,2) = dS_dC * cosA + S * (-sinA) * dA_dC;
  J(1,2) = dS_dC * sinA + S * ( cosA) * dA_dC;
  // ∂/∂z0
  J(2,3) = 1.0;
  // ∂/∂ct  (through S and A in x,y; direct in z)
  J(0,4) = dS_dct * cosA + S * (-sinA) * dA_dct;
  J(1,4) = dS_dct * sinA + S * ( cosA) * dA_dct;
  J(2,4) = s / (n*n*n);

  return J;
}



TMatrixDSym GraphDisplacedVertexFinder::trkW(const ViewEntry& tv, double s) const {
  // 1) Cx = J Cp J^T
  TMatrixDSym Cx(3); Cx.Zero();
  {
    TMatrixD J = trkJx(tv, s);          // 3×5
    TMatrixDSym Cp = tv.Cp;             // 5×5
    TMatrixD JC(3,5); JC.Mult(J, Cp);   // 3×5
    TMatrixD JCT(3,3); JCT.Mult(JC, TMatrixD(TMatrixD::kTransposed, J));
    for (int i=0;i<3;++i)
      for (int j=i;j<3;++j)
        Cx(i,j) = Cx(j,i) = JCT(i,j);
  }

  // Invert to get W
  TMatrixDSym W(3); W.ResizeTo(3,3); W.Zero();

  //Regulated via eigenfloor (and repaired via residual refinement)
  if (!InvertSPD(Cx, W)) {
    // last-resort ridge (very small, won’t bias)
    for (int i=0;i<3;++i) Cx(i,i) *= (1.0 + 1e-9);
    if (!InvertSPD(Cx, W)) { W.Zero(); }
  }
  return W;
}

// Cx at a fixed s is SPD but has a strongly underestimated variance along the tangent
// this will lead to a blow up in inverses, so we add a tangent floor σ_min to Cx to make it 
// stably invertible with Cholesky, otherwise we would need to build pseudo-inverses which are more expensive
TMatrixDSym GraphDisplacedVertexFinder::trkCx(const ViewEntry& tv, double s) const {
  // 1) Cx = J Cp J^T
  TMatrixDSym Cx(3); Cx.Zero();
  {
    TMatrixD J = trkJx(tv, s);          // 3×5
    TMatrixDSym Cp = tv.Cp;             // 5×5
    TMatrixD JC(3,5); JC.Mult(J, Cp);   // 3×5
    TMatrixD JCT(3,3); JCT.Mult(JC, TMatrixD(TMatrixD::kTransposed, J));
    for (int i=0;i<3;++i)
      for (int j=i;j<3;++j)
        Cx(i,j) = Cx(j,i) = JCT(i,j);
  }

  // (optional tiny isotropic jitter to avoid exact zeros)
  const double eps2 = 1e-12; // mm^2
  Cx(0,0) += eps2; Cx(1,1) += eps2; Cx(2,2) += eps2;

  // Tangent stabilizer: ensure variance along t̂ is at least σ_min^2
  const TVectorD t = trkdXds(tv, s);           // 3×1
  double tn2 = t.Norm2Sqr();
  if (tn2 > 0.0) {
    TVectorD that = t; that *= (1.0 / std::sqrt(tn2));  // unit tangent
    // current variance along t̂
    double vt = 0.0;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) vt += that(i)*Cx(i,j)*that(j);

    // choose a modest floor for tangent variance (set this as a config param if you like)
    const double sigma_tan_floor2 = 1.0; // mm^2 (√=1 mm). Make it bigger if you want *less* weight along t̂.

    if (vt < sigma_tan_floor2) {
      const double lam = (sigma_tan_floor2 - vt); // add exactly what’s missing
      // Cx ← Cx + lam * t̂ t̂^T
      for (int i=0;i<3;++i)
        for (int j=i;j<3;++j)
          Cx(i,j) = Cx(j,i) = Cx(i,j) + lam * that(i) * that(j);
    }
  }

  return Cx;
}


// a · b in R^D as a convenience helper (I hate root sometimes)
double GraphDisplacedVertexFinder::dotD(const TVectorD& a, const TVectorD& b) const {
  const int n = a.GetNrows();
  assert(b.GetNrows() == n && "dotD: size mismatch");
  double res = 0.0;
  for (int i = 0; i < n; ++i) res += a(i) * b(i);
  return res;
}

//Minimise euclidean distance between two tracks via coupled GN
std::pair<double, double>
GraphDisplacedVertexFinder::GetPocaT2T(const ViewEntry& trk1,
                                       const ViewEntry& trk2, double* s1_0, double* s2_0,
                                       int maxIter, double tol) const
{
  // ---- parameters ----
  const double smax      = 70000.0;                 // box on arc-length
  const int    iters     = (maxIter > 0 ? maxIter : 50);
  const double stepTol   = (tol > 0 ? tol : 1e-3);  // mm in arc-length
  const double gradTol   = 1e-6;                    // mm scale for gradient
  const double fTol      = 1e-12;                   // tiny absolute reduction
  double lambda          = 1e-3;                    // LM damping
  const double lambda_dn = 0.333333333333;          // ↓ when good step
  const double lambda_up = 10.0;                    // ↑ when bad step
  const double seed_tau  = 1e-12;                   // Tikhonov seed reg (units of a11+a22)

  auto clampS = [&](double& s){
    if (s >  smax) s =  smax;
    if (s < -smax) s = -smax;
  };

  // ---- initial seed ----
  double s1 = 0.0, s2 = 0.0;
  if (s1_0 && s2_0) {
    s1 = *s1_0; s2 = *s2_0;
  } else {
    // Linear PoCA seed (ignore curvature) with Tikhonov regularization
    TVectorD x1_0 = trkX(trk1, 0.0);
    TVectorD x2_0 = trkX(trk2, 0.0);
    TVectorD r0   = x1_0; r0 -= x2_0;

    TVectorD v1_0 = trkdXds(trk1, 0.0);
    TVectorD v2_0 = trkdXds(trk2, 0.0);

    const double a11 = dotD(v1_0, v1_0);
    const double a22 = dotD(v2_0, v2_0);
    const double a12 = dotD(v1_0, v2_0);
    const double b1  = dotD(v1_0, r0);
    const double b2  = dotD(v2_0, r0);

    double denom = a11 * a22 - a12 * a12;
    denom += seed_tau * (a11 + a22);      // Tikhonov stabilizer

    // near-parallel guard: fall back to independent projections
    const double cosang2 = (a12*a12) / std::max(1e-24, a11*a22);
    if (cosang2 > 0.999*0.999) {
      s1 = (a11 > 0.0) ? (-b1 / a11) : 0.0;
      s2 = (a22 > 0.0) ? (-b2 / a22) : 0.0;
    } else {
      s1 = ( a12 * b2 - a22 * b1) / denom;
      s2 = ( a11 * b2 - a12 * b1) / denom;
    }
  }

  clampS(s1); clampS(s2);

  // ---- objective ----
  auto F = [&](double s1c, double s2c)->double {
    TVectorD d = trkX(trk1, s1c); d -= trkX(trk2, s2c);
    return dotD(d, d);
  };

  double fcur = F(s1, s2);

  for (int itN = 0; itN < iters; ++itN) {
    // Geometry at current s
    // std::cerr << "Current s_1 = " << s1 << " s_2 = " << s2 << std::endl;
    const TVectorD x1 = trkX(trk1, s1);
    const TVectorD x2 = trkX(trk2, s2);
    TVectorD d  = x1; d -= x2;

    const TVectorD v1 = trkdXds(trk1, s1);
    const TVectorD v2 = trkdXds(trk2, s2);

    // Gradient (Gauss–Newton)
    const double g0 =  2.0 * dotD(d, v1);
    const double g1 = -2.0 * dotD(d, v2);

    // Stop if gradient is tiny
    if (std::max(std::abs(g0), std::abs(g1)) < gradTol) break;

    // GN Hessian (SPD unless v1 ‖ v2)
    double H00 = 2.0 * dotD(v1, v1);
    double H11 = 2.0 * dotD(v2, v2);
    double H01 = -2.0 * dotD(v1, v2);

    // LM damping: (H + λ I) Δ = g
    double H00l = H00 + lambda;
    double H11l = H11 + lambda;
    double det  = H00l * H11l - H01 * H01;

    // If singular, increase lambda and retry
    if (std::abs(det) < 1e-24) {
      lambda *= lambda_up;
      continue;
    }

    const double ds1 = ( H11l * g0 - H01 * g1) / det;
    const double ds2 = (-H01 * g0 + H00l * g1) / det;

    // std::cerr << "Proposed step = (" << ds1 << ", " << ds2 << ")" << std::endl;

    // Trial step (trust region, no line search)
    double s1_try = s1 - ds1;
    double s2_try = s2 - ds2;

    // Handle box: project; treat projection as "suspicious" to grow lambda if needed
    bool hitWall = false;
    auto project = [&](double& s){
      if (s >  smax) { s =  smax; hitWall = true; }
      if (s < -smax) { s = -smax; hitWall = true; }
    };
    project(s1_try); project(s2_try);

    const double fnew = F(s1_try, s2_try);

    // Gain ratio ρ = ared / pred, with pred using (H+λI)
    const double ared = fcur - fnew;
    const double quad = 0.5*( ds1*(H00l*ds1 + H01*ds2)
                            + ds2*(H01*ds1 + H11l*ds2) );
    const double pred = (g0*ds1 + g1*ds2) - quad;
    const double rho  = (pred > 0.0 ? (ared / pred) : -1.0);

    if (!hitWall && rho > 0.0) {
      // Accept
      s1 = s1_try; s2 = s2_try; fcur = fnew;
      lambda = std::max(1e-12, lambda * lambda_dn);
      if (std::max(std::abs(ds1), std::abs(ds2)) < stepTol) break;
      if (ared < fTol) break;
    } else {
      // Reject
      lambda *= lambda_up;
      // Optional: if we keep hitting walls, shrink step by inflating lambda more
      if (hitWall) lambda *= 3.0;
    }
    if (std::max(std::abs(ds1), std::abs(ds2)) < stepTol) break;
  }

  // Final clamp and return
  clampS(s1); clampS(s2);

  // std::cerr << "GetPocaT2T found s1,s2 = " << s1 << ", " << s2 << std::endl;
  // std::cerr << "With distance = " << std::sqrt(F(s1, s2)) << " mm"<< std::endl;
  return {s1, s2};
}

// Minimise euclidean distance between track and vertex via GN
double GraphDisplacedVertexFinder::GetPocaT2V(
    const ViewEntry& trk,
    const TVectorD& vertex,
    double s0,
    int maxIter,
    double tol) const
{
    if (maxIter <= 0) maxIter = 50;
    if (tol     <= 0) tol     = 1e-4;

    // --- initial guess
    double s = s0;
    const double sMax = 2.5e6;
    s = std::clamp(s, -sMax, sMax);

    // natural helical half–turn for step control
    const double s_turn = turn_scale(trk.p(2), trk.n);
    const double ds_cap = 0.5 * s_turn;   // never jump more than half a turn

    // Newton on φ(s) = ½||x(s)−v||² with backtracking
    for (int it = 0; it < maxIter; ++it) {
        TVectorD x = trkX(trk, s);
        TVectorD v = trkdXds(trk, s);
        TVectorD a = trkd2Xds2(trk, s);

        TVectorD r = x - vertex;
        double phi  = 0.5 * r.Norm2Sqr();
        double dphi = dotD(r, v);
        double d2phi = dotD(v,v) + dotD(r, a);
        // double d2phi = dotD(v,v);
        if (std::abs(d2phi) < 1e-14) d2phi = 1e-14;

        if (std::abs(dphi) < tol) break;

        double ds = -dphi / d2phi;
        // clamp to half-turn step
        ds = std::clamp(ds, -ds_cap, ds_cap);

        // backtracking to ensure true distance decreases
        double alpha = 1.0;
        for (int bt = 0; bt < 10; ++bt) {
            double s_try = std::clamp(s + alpha * ds, -sMax, sMax);
            TVectorD r_try = trkX(trk, s_try) - vertex;
            if (0.5 * r_try.Norm2Sqr() <= phi) { s = s_try; break; }
            alpha *= 0.5;
        }
        if (alpha < 1e-3) break;
        if (std::abs(ds) < tol) break;
    }

    // --- local minimum found; refine global choice
    // Scan ±1 turn around s for true global minimum
    // Catches and refines multi-loopers
    double s_best = s;
    double d2_best = (trkX(trk, s) - vertex).Norm2Sqr();

    for (int k = 1; k <= 10; ++k){
      for (int sign : {-1, +1}){
        double s_scan = s + sign * k * s_turn;
        TVectorD r_scan = trkX(trk, s_scan) - vertex;
        double d2 = r_scan.Norm2Sqr();
        if (d2 < d2_best) { d2_best = d2; s_best = s_scan; }
        // else break;
      }
    }

    return s_best;
}

// robust MAD scale; returns (sigma, median_rho)
static inline std::pair<double,double>
mad_scale(const std::vector<double>& rho)
{
  if (rho.empty()) return {0.0, 0.0};
  std::vector<double> r = rho;
  std::nth_element(r.begin(), r.begin()+r.size()/2, r.end());
  double med = r[r.size()/2];
  std::vector<double> dev(r.size());
  for (size_t i=0;i<r.size();++i) dev[i] = std::abs(rho[i]-med);
  std::nth_element(dev.begin(), dev.begin()+dev.size()/2, dev.end());
  double mad = dev[dev.size()/2];
  double sigma = 1.4826 * mad;
  if (!(sigma>0.0)) {
    // fallback: use median itself as a crude scale, or 1.0
    sigma = (med>0.0 ? 1.4826*0.5*med : 1.0);
  }
  return {sigma, med};
}


// Find good and robust vertex seed fast from minimising euclidean distances (less linear algebra churn)
// Self-tuning IRLS via MAD scaling
// Solved via GN with backtracking
bool GraphDisplacedVertexFinder::EuclideanSeed(const std::vector<const ViewEntry*>& tvs,
                                               std::vector<double>& s_io,
                                               std::vector<double>* w_io,
                                               TVectorD& mu_out,               
                                               TMatrixDSym* cov_out,
                                               const FitOpts& opts) const
{
  const size_t N = tvs.size();
  if (N < 2) return false;

  // std::cerr << "Entering EuclideanSeed" << std::endl;

  //Extract options
  double wtol = (opts.wTol > 0 ? opts.wTol : 1e-4);
  double sigma_floor = (opts.sigma_floor > 0 ? opts.sigma_floor : 0.5);
  double vtol = (opts.vTol > 0 ? opts.vTol : 1e-4);
  double stol = (opts.sTol > 0 ? opts.sTol : 1e-4);
  int max_iter = (opts.maxIter > 0 ? opts.maxIter : 50);
  int max_irls = (opts.maxIRLS > 0 ? opts.maxIRLS : 10);
  double absCap = (opts.absCap > 0 ? opts.absCap : 1000.0);
  double frac_turn = (opts.frac_turn > 0 ? opts.frac_turn : 0.5);
  int max_backtrack = (opts.btMax > 0 ? opts.btMax : 10);
  double wcutoff = (opts.weightActive > 0 ? opts.weightActive : 0.01);

  // phases s and weights w
  std::vector<double> s(N, 0.0);
  if (s_io.size() == N) s = s_io;

  std::vector<double> w(N, 1.0);
  if (w_io && w_io->size() == N) w = *w_io;

  // inner state buffers (all TVectorD(3))
  std::vector<TVectorD> X(N, TVectorD(3)), t(N, TVectorD(3)),
                        tp(N, TVectorD(3)), r(N, TVectorD(3));
  std::vector<double>   d(N), Hdiag(N), g(N), s_turn(N);
  std::vector<double>   delta_s(N, 0.0);

  for (size_t i=0;i<N;++i) {
    const auto& tv = *tvs[i];
    s_turn[i] = turn_scale(tv.p(2), tv.n);
  }

  // ---- Tangent-direction stabilizer (rank-1; only used when needed)
  // K <- K + λ * t̂ t̂ᵀ, where t̂ is weighted mean of tangents
  auto tangent_stab3D = [&](TMatrixDSym& K, double eta = 1e-8) -> void {
    TVectorD tbar(3); tbar.Zero();
    double wsum = 0.0;
    for (size_t i=0;i<N;++i) { TVectorD tmp = t[i]; tmp *= w[i]; tbar += tmp; wsum += w[i]; }
    const double n2 = dotD(tbar,tbar);
    if (n2 <= 0.0 || wsum <= 0.0) return;
    tbar *= (1.0/std::sqrt(n2));

    const double trK = K(0,0)+K(1,1)+K(2,2);
    const double lambda0 = eta * (trK > 0.0 ? trK/3.0 : 1.0);

    for (int k=0; k<3; ++k) {
      TMatrixDSym Ktrial = K;
      const double lam = lambda0 * std::pow(10.0, k);
      Ktrial.Rank1Update(tbar, +lam);
      TDecompChol chol(Ktrial);
      if (chol.Decompose()) { K = Ktrial; return; }
      else std::cerr << "EuclideanSeed: Adding tangent-stabilizer" << std::endl;
    }
    // leave to caller to jitter if still not SPD
  };

  // --- helpers --------------------------------------------------------------

  auto recompute_state = [&](const std::vector<double>& s_vec,
                             TVectorD& mu,
                             double&   J)->void
  {
    double Wsum = 0.0;
    TVectorD mu_num(3); mu_num.Zero();

    for (size_t i=0;i<N;++i) {
      
      const auto& tv = *tvs[i];
      X[i]  = trkX(tv, s_vec[i]);        // TVectorD(3)
      t[i]  = trkdXds(tv, s_vec[i]);     // unit-speed tangent
      tp[i] = trkd2Xds2(tv, s_vec[i]);  // curvature vector
      if (w[i] < wcutoff) continue;
      Wsum += w[i];
      TVectorD tmp = X[i]; tmp *= w[i];
      mu_num += tmp;
    }
    const double invW = (Wsum>0.0) ? (1.0/Wsum) : 0.0;
    mu = mu_num; mu *= invW;

    // residuals and cost
    J = 0.0;
    for (size_t i=0;i<N;++i) {
      if (w[i] < wcutoff) continue;
      r[i] = mu; r[i] -= X[i];
      J += 0.5 * w[i] * dotD(r[i], r[i]);
    }

    // gradient g_i = -(r_i·t_i)
    for (size_t i=0;i<N;++i) g[i] = - dotD(r[i], t[i]);

    // diagonals (with curvature gate)
    for (size_t i=0;i<N;++i) {
      double Hdiagi = (1 - w[i]*invW);
      double di     = 1.0;
      // const double rdot_tp = dotD(r[i], tp[i]);
      // if (rdot_tp <= 0) { di -= rdot_tp; Hdiagi -= rdot_tp; }
      if (di     < 1e-12) di     = 1e-12;
      if (Hdiagi < 1e-12) Hdiagi = 1e-12;
      d[i]     = di;
      Hdiag[i] = Hdiagi;
    }
  };

  auto reBLUE_and_cost = [&](const std::vector<double>& s_vec,
                             TVectorD& mu,
                             double&   J)->void
  {
    double Wsum = 0.0;
    TVectorD mu_num(3); mu_num.Zero();
    for (size_t i=0;i<N;++i) {
      const auto& tv = *tvs[i];
      X[i]  = trkX(tv, s_vec[i]);
      if (w[i] < wcutoff) continue;
      Wsum += w[i];
      TVectorD tmp = X[i]; tmp *= w[i];
      mu_num += tmp;
    }
    const double invW = (Wsum>0.0) ? (1.0/Wsum) : 0.0;
    mu = mu_num; mu *= invW;

    J = 0.0;
    for (size_t i=0;i<N;++i) {
      r[i] = mu; r[i] -= X[i];
      if (w[i] < wcutoff) continue;
      J += 0.5 * w[i] * dotD(r[i], r[i]);
    }
  };

  // Build S3 and rhs3 for coupled Δs via Woodbury
  auto build_Woodbury_core = [&](TMatrixDSym& S3, TVectorD& rhs3) -> void
  { 
    // std::cerr << "Euclidean Seed: Build Woodbury" << std::endl;
    double Wsum = 0.0; for (size_t i=0;i<N;++i) Wsum += w[i];

    TMatrixDSym A(3);  A.Zero();
    TVectorD b(3);     b.Zero();

    for (size_t i=0;i<N;++i) {
      if (w[i] < wcutoff) continue;
      const double di = std::max(1e-12, d[i]);
      const double wi = w[i];

      A.Rank1Update(t[i], (wi*wi)/di);           // sum (wi^2/di) t_i t_iᵀ

      TVectorD ti = t[i]; ti *= wi * (g[i]/di);  // sum (wi gi/di) t_i
      b += ti;
    }

    S3.ResizeTo(3,3);
    S3 = A; S3 *= -1.0;                          // Wsum*I - A
    S3(0,0) += Wsum; S3(1,1) += Wsum; S3(2,2) += Wsum;

    rhs3.ResizeTo(3); rhs3 = b;
    // std::cerr << "Test SPD for S3" << std::endl;
    TDecompChol test(S3);
    if (!test.Decompose()){
      std::cerr << "EuclideanSeed: S3 not SPD, adding tangent stabilizer" << std::endl;
      tangent_stab3D(S3);
      TDecompChol retry(S3);
      if (!retry.Decompose()) {
        std::cerr << "EuclideanSeed: S3 still not SPD, adding jitter" << std::endl;
        S3(0,0) += 1e-12; S3(1,1) += 1e-12; S3(2,2) += 1e-12; //last resort
      }
    }
  };

  // backtracking line-search on fixed weights
  auto line_search = [&](const std::vector<double>& s_cur,
                         const std::vector<double>& step,
                         const TVectorD& /*mu_cur*/,
                         double J_cur,
                         std::vector<double>& s_new,
                         TVectorD& mu_new,
                         double& J_new)->bool
  {
    s_new = s_cur;
    double alpha = 1.0;
    for (int bt=0; bt<max_backtrack; ++bt) {
      for (size_t i=0;i<N;++i) {
        const double clamp = std::min(frac_turn * s_turn[i], absCap);
        const double ds = std::max(-clamp, std::min(clamp, alpha*step[i]));
        s_new[i] = s_cur[i] + ds;
      }

      reBLUE_and_cost(s_new, mu_new, J_new);
      if (J_new <= J_cur) return true;
      alpha *= 0.5;
    }
    return false;
  };

  // ---------------- IRLS OUTER LOOP ----------------
  bool ok = true;
  TVectorD mu(3); mu.Zero();
  TVectorD mu_prev(3); mu_prev.Zero();
  double J = 0.0;

  for (int irls=0; irls<max_irls; ++irls) {

    int accepted = 0;
    for (int it=0; it<max_iter; ++it) {

      recompute_state(s, mu, J);
      mu_prev = mu;

      // Build Woodbury core S3, rhs3 and solve S3 z = rhs3
      TMatrixDSym S3; S3.ResizeTo(3,3); S3.Zero();
      TVectorD rhs3(3);
      build_Woodbury_core(S3, rhs3);

      TVectorD z(3);
      bool spd_ok = SolveSPD(S3, rhs3, z);
      if (!spd_ok) {
        std::cerr << "[Euclidean Seed] WARNING: SPD solve failed using Cholesky" << std::endl;
        return false;
        }

      // Δs = -D^{-1} g - D^{-1} U^T z  with U_i = w_i t_i
      for (size_t i=0;i<N;++i) {
        if (!spd_ok) {
          delta_s[i] = - g[i]/Hdiag[i];
        } else {
          delta_s[i] = - g[i]/d[i] - (dotD(t[i], z))/d[i];
        }
      }

      // line search
      std::vector<double> s_try;
      TVectorD mu_try(3); mu_try.Zero();
      double J_try = 0.0;
      double max_ds = 0.0;
      bool accepted_step = line_search(s, delta_s, mu, J, s_try, mu_try, J_try);
      for (size_t i=0;i<N;++i) max_ds = std::max(max_ds, std::abs(s_try[i]-s[i]));

      if (!accepted_step && max_ds > stol) {
        // fallback: uncoupled step
        std::vector<double> pure(N);
        for (size_t i=0;i<N;++i) pure[i] = - g[i]/Hdiag[i];
        accepted_step = line_search(s, pure, mu, J, s_try, mu_try, J_try);
        if (!accepted_step) { ok = false; }
      }

      // accept
      // double max_ds = 0.0;
      for (size_t i=0;i<N;++i) max_ds = std::max(max_ds, std::abs(s_try[i]-s[i]));
      if (!ok && max_ds > stol) break;
      else ok = true;
      
      s.swap(s_try);
      mu = mu_try;
      J  = J_try;
      ++accepted;

      // std::cerr << "Euclidean Seed:" << std::endl;
      // for (int i=0;i<N;++i) std::cerr << "  s[" << i << "] = " << s[i] << std::endl;
      // std::cerr << "  mu = " << mu(0) << " " << mu(1) << " " << mu(2) << std::endl;

      // convergence?
      TVectorD dmu = mu; dmu -= mu_prev;
      if (max_ds < stol && std::sqrt(dotD(dmu,dmu)) < vtol) break;
    } // inner GN

    if (!ok) break;

    // --- IRLS weight update
    std::vector<double> rho(N);
    for (size_t i=0;i<N;++i) { 
      TVectorD dr = mu; 
      dr -= X[i]; 
      rho[i] = std::sqrt(dotD(dr,dr)); 
    }

    // MAD scaling to make residuals dimensionless and get a self-tuning sigmoid
    auto [sigma_raw, med] = mad_scale(rho);
    if (!(sigma_raw>0.0) || !opts.useWeights || N <= 2) break;

    std::vector<double> w_new(N, 1.0);
    double max_dw = 0.0;
    const double sigma = std::max(sigma_raw, sigma_floor); // simple floor to not collapse to 0 for clean seeds
    // std::cerr <<"Euclidean seed: current sigma = " << sigma << std::endl;
    // std::cerr << "Current median = " << med << std::endl;
    // std::cerr << "Weights:" << std::endl;
    for (size_t i=0;i<N;++i) {
      const double z2  = (rho[i]*rho[i])/(sigma*sigma);
      const double arg = (z2 - opts.c0)/opts.beta;
      const double wi  = 1.0/(1.0 + std::exp(arg));
      // std::cout << "  w[" << i << "] = " << wi << std::endl;
      max_dw = std::max(max_dw, std::abs(wi - w[i]));
      w_new[i] = wi;
    }
    w.swap(w_new);

    if (max_dw < wtol) break; // weights stabilized
  } // IRLS

  if (!ok) return false;

  // outputs
  mu_out = mu;        // TVectorD(3)
  s_io = s;
  if (w_io) *w_io = w;

  // std::cerr << "Found seed at " << mu(0) << " " << mu(1) << " " << mu(2) << std::endl;

  if (cov_out) {
    // crude seed covariance ≈ (W I)^(-1)
    double Wsum = 0.0; for (double wi: w) Wsum += wi;
    const double invW = (Wsum>1e-12) ? (1.0/Wsum) : 1e12;
    cov_out->ResizeTo(3,3);
    (*cov_out)(0,0)=invW; (*cov_out)(0,1)=0.0;  (*cov_out)(0,2)=0.0;
    (*cov_out)(1,0)=0.0;  (*cov_out)(1,1)=invW; (*cov_out)(1,2)=0.0;
    (*cov_out)(2,0)=0.0;  (*cov_out)(2,1)=0.0;  (*cov_out)(2,2)=invW;
  }

  return true;
}


// sigmoid function for weights, could be made optional so we can have Huber or Tukey weights as alternatives
double GraphDisplacedVertexFinder::sigmoid(double rho, double c0, double beta) const {
    double arg = (rho - c0)/beta;
    if (arg >= 0.) return std::exp(-arg)/(1.0 + std::exp(-arg));
    return 1.0/(1.0 + std::exp(arg));
}

bool GraphDisplacedVertexFinder::CoupledMetricFit(const std::vector<const ViewEntry*>& tvs,
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
                                                  const FitOpts& opts) const
{
  const size_t N = tvs.size();
  if (N < 2) return false;

  //Extract options
  double c0 = (opts.c0 > 0.0) ? opts.c0 : 9.0;
  double beta = (opts.beta > 0.0) ? opts.beta : 2.0;
  double weightCut = (opts.weightCut > 0.0) ? opts.weightCut : 0.5;
  double wEps = (opts.wEps > 0.0) ? opts.wEps : 1e-6;
  double w_tol = (opts.wTol > 0.0) ? opts.wTol : 1e-4;
  double s_tol = (opts.sTol > 0.0) ? opts.sTol : 1e-4;
  double v_tol = (opts.vTol > 0.0) ? opts.vTol : 1e-4;
  int maxInnerIter = (opts.maxIter > 0) ? opts.maxIter : 50;
  int maxOuterIter = (opts.maxIRLS > 0) ? opts.maxIRLS : 10; 
  int max_backtrack = (opts.btMax > 0) ? opts.btMax : 10;
  double frac_turn = (opts.frac_turn > 0.0) ? opts.frac_turn : 0.75;
  double absCap = (opts.absCap > 0.0) ? opts.absCap : 25000.0;
  double wcutoff = (opts.weightActive > 0.0) ? opts.weightActive : 1e-3;
  bool useWeights = opts.useWeights && N > 3;
  bool useTiming = opts.useTiming;

  // --- init s, w

  std::vector<double> s(N, 0.0);
  if (s_io.size() == N) s = s_io;

  std::vector<double> chi2i(N, 0.0), chi2i_sp(N, 0.0);
  chi2i_out.assign(N, 0.0);

  //Quickly initialise phases to the initial vertex guess, if s_io provided, converges usually in 1 iteration each
  TMatrixDSym nullcov(3); nullcov.Zero();
  for (size_t i=0;i<N;++i){
    GetMetricPOCAT2V(*tvs[i], vstar_io, nullcov, s[i], chi2i_out[i]);
  }

  std::vector<double> w(N, 1.0);
  if (w_io && w_io->size() == N) w = *w_io;

  // --- caches
  std::vector<TVectorD> X(N), t(N), tp(N), r(N);
  std::vector<double> tau(N, 0.0), sigtau2(N, 1.0), rtau(N, 0.0);
  
  std::vector<double> d(N, 0.0), Hdiag(N, 0.0), g(N, 0.0), s_turn(N, 0.0);
  std::vector<TMatrixDSym> W(N);

  std::vector<bool> hasTime(N, false);

  // std::cerr << "CoupledMetricFit: initialised caches" << std::endl;

  for (size_t i=0;i<N;++i){
    X[i].ResizeTo(3); 
    X[i].Zero();
    t[i].ResizeTo(3); 
    t[i].Zero();
    tp[i].ResizeTo(3); 
    tp[i].Zero();
    r[i].ResizeTo(3);
    r[i].Zero();

    W[i].ResizeTo(3,3);
    W[i].Zero();
  }

  // std::cerr << "CoupledMetricFit: resized caches" << std::endl;

  for (size_t i=0;i<N;++i) s_turn[i] = turn_scale(tvs[i]->p(2), tvs[i]->n);

  // reusable temporaries (avoid heap churn)
  TVectorD tmpWX(3), tmpWu(3);

  TVectorD vcurrent = vstar_io;


  // ---- prior into information space
  TMatrixDSym Wprior(3); Wprior.Zero();
  TVectorD    Wvprior(3); Wvprior.Zero();
  if (cov_prior) {
    if (InvertSPD(*cov_prior, Wprior)) {
      if (v_prior) { Wvprior = Wprior * (*v_prior);  }
    }
  }

  // ---- build Stilde and b at initial s (also compute W, chi2 and weights)
  TMatrixDSym Stilde(3); Stilde.Zero();
  TVectorD    b(3); b.Zero();
  if (cov_prior) { Stilde += Wprior; if (v_prior) b += Wvprior; }

  // std::cerr<<"Filling caches"<<std::endl;

  for (size_t i=0;i<N;++i) {
    const auto& tv = *tvs[i];
    hasTime[i] = tv.hasTime;
    sigtau2[i] = tv.sigma_t2;

    X[i]  = trkX(tv, s[i]);
    t[i]  = trkdXds(tv, s[i]);
    tp[i] = trkd2Xds2(tv, s[i]);

    W[i].ResizeTo(3,3); W[i].Zero();
    // std::cerr << "Shape of W[i] = " << W[i].GetNrows() << ", " << W[i].GetNcols() << std::endl;
    TMatrixDSym Wi = trkW(tv, s[i]);
    // std::cerr << "Shape of Wi = " << Wi.GetNrows() << ", " << Wi.GetNcols() << std::endl;
    W[i]  = Wi;

    TMatrixDSym Cx = trkCx(tv, s[i]);

    // std::cerr << "Diagonal of Cx = " << Cx(0,0) << ", " << Cx(1,1) << ", " << Cx(2,2) << std::endl;
    // std::cerr << "Diagonal of Wi = " << Wi(0,0) << ", " << Wi(1,1) << ", " << Wi(2,2) << std::endl;
    
    r[i] = vcurrent - X[i];
    chi2i[i] = W[i].Similarity(r[i]);
    chi2i_sp[i] = chi2i[i];

    // std::cerr << "chi2i[" << i << "] = " << chi2i[i] << std::endl;
    double c2test = dotD(r[i], Wi * r[i]);
    // std::cerr << "c2test[" << i << "] = " << c2test << std::endl;
    if (useWeights && opts.useSelfSeeding && vstar_io.Norm2Sqr() > 0.0 && N > 3)
    {
      w[i] = sigmoid(chi2i[i], opts.c0, opts.beta);
    }else {
      w[i] = 1.0;
    }
    
    if (w[i] < wcutoff) continue;

    Stilde += (w[i] * W[i]);
    tmpWX = W[i] * X[i];
    b += (w[i] * tmpWX);
  }

  // initial BLUE
  TVectorD vstar(3);
  // std::cerr << "CoupledMetricFit: initial BLUE" << std::endl;
  if (!SolveSPD(Stilde, b, vstar)) return false;
  // std::cerr << "Start vertex = " << vcurrent(0) << ", " << vcurrent(1) << ", " << vcurrent(2) << std::endl;
  // std::cerr << "Start BLUE = " << vstar(0) << ", " << vstar(1) << ", " << vstar(2) << std::endl;
  vcurrent = vstar;

  // reweight with metric-consistent residuals (+ time)
  for (size_t i=0;i<N;++i) {
    r[i] = vcurrent - X[i];
    double c2 = W[i].Similarity(r[i]);
    // std::cerr << "Initial chi2 of track spatial " << i << " = " << c2 << std::endl;
    // std::cerr << "Initial vertex at " << vcurrent(0) << ", " << vcurrent(1) << ", " << vcurrent(2) << std::endl;
    // std::cerr << "Track i = " << i << " at " << X[i](0) << ", " << X[i](1) << ", " << X[i](2) << std::endl;
    // std::cerr << "Track i = " << i << " residual at " << r[i](0) << ", " << r[i](1) << ", " << r[i](2) << std::endl;
    chi2i[i] = c2;
    chi2i_sp[i] = c2;
    
    // std::cerr << "Initial chi2 of track " << i << " = " << chi2i[i] << std::endl;
    if (useWeights && opts.useSelfSeeding && N > 3) 
    { 
      w[i] = sigmoid(c2, c0, beta);
    }else{
      w[i] = 1.0;
    }
  }

  // --- helpers --------------------------------------------------------------

  auto reBLUE_and_cost = [&](const std::vector<double>& s_vec,
                             TVectorD& vstar_out, double& J) -> void
  { 
    // std::cerr << "Entering reBLUE_and_cost" << std::endl;
    TMatrixDSym St(3); St.ResizeTo(3,3); St.Zero();
    TVectorD    rhs(3); rhs.Zero();

    if (cov_prior) { St += Wprior; if (v_prior) rhs += Wvprior; }

    double Wt_loc = 0.0, tau_num = 0.0;

    for (size_t i=0;i<N;++i) {
      const auto& tv = *tvs[i];
      X[i]  = trkX(tv, s_vec[i]);
      t[i]  = trkdXds(tv, s_vec[i]);
      tp[i] = trkd2Xds2(tv, s_vec[i]);

      if (w[i] < wcutoff) continue;
      St += (w[i] * W[i]);
      tmpWX = W[i] * X[i];
      rhs  += (w[i] * tmpWX);
    }
    // std::cerr << "reBLUE_and_cost: building BLUE" << std::endl;
    if (!SolveSPD(St, rhs, vstar_out)) {
      std::cerr << "reBLUE_and_cost: SolveSPD failed" << std::endl;
    };
    // std::cerr << "reBLUE_and_cost: new BLUE at " << vstar_out(0) << ", " << vstar_out(1) << ", " << vstar_out(2) << std::endl;
    // if (useTiming) std::cerr << "Current vertex time = " << vtstar_out << std::endl;


    // cost (frozen metrics)
    J = 0.0;
    if (cov_prior && v_prior) {
      TVectorD dv = vstar_out - *v_prior;
      J += 0.5 * Wprior.Similarity(dv);
    }
    for (size_t i=0;i<N;++i) {
      double Jsi = 0;
      const auto& tv = *tvs[i];
      r[i] = vstar_out - X[i];
      
      Jsi += 0.5 *  W[i].Similarity(r[i]);
      chi2i_sp[i] = 2.0 * Jsi;
      

      chi2i[i] = 2.0 * Jsi; // keep per-track
      // std::cerr << "chi2i[" << i << "] = " << chi2i[i] << std::endl;

      if (w[i] > wcutoff) J += w[i] * Jsi;
    }
  };

  // recompute W at new s, then reBLUE
  auto recompute_outer = [&](TVectorD& vstar_out, double& J) -> void
  {
    for (size_t i=0;i<N;++i) {W[i].ResizeTo(3,3); W[i].Zero(); W[i] = trkW(*tvs[i], s[i]);}
    reBLUE_and_cost(s, vstar_out, J);
  };

  // recompute g, d, Hdiag with frozen metrics
  auto recompute_inner = [&](TVectorD& vstar_out,  double& J) -> void
  {
    // reBLUE (fills r[i], chi2i, tau if timing)
    reBLUE_and_cost(s, vstar_out, J);

    // Build Wsum = Σ w_i W_i (with cutoff) once, then invert for Hdiag
    TMatrixDSym Wsum(3); Wsum.Zero();
    for (size_t i=0;i<N;++i) if (w[i] >= wcutoff) Wsum += (w[i] * W[i]);

    // add prior information 
    if (cov_prior) Wsum += Wprior;

    TMatrixDSym Wsum_inv(3); Wsum_inv.ResizeTo(3,3); Wsum_inv.Zero();  // 3×3
    bool have_inv = InvertSPD(Wsum, Wsum_inv);

    for (size_t i=0;i<N;++i) {
      // gradient: g_i = - t_i^T W_i r_i   (+ time component if used)
      tmpWu = W[i] * r[i];       // W_i r_i
      g[i] = - dotD(tmpWu, t[i]);

      // diagonal d_i (includes curvature gate)
      double di = W[i].Similarity(t[i]);   // t_i^T W_i t_i
      const double curvi = dotD(W[i] * tp[i], r[i]);  // tp_i^T W_i r_i
      // if (curvi <= 0.0) di -= curvi;
      d[i] = std::max(di, opts.dEps);

      // Hdiag_i = d_i - (Wsum^{-1}).Similarity(W_i t_i)   (spatial-only case)
      if (have_inv) {
        tmpWu = W[i] * t[i];                     // W_i t_i
        Hdiag[i] = d[i] - w[i] *Wsum_inv.Similarity(tmpWu); // tmpWu^T Wsum_inv tmpWu
      } else {
        Hdiag[i] = d[i]; // fallback if inversion failed
      }

      if (Hdiag[i] < opts.dEps) Hdiag[i] = opts.dEps;
    }
  };

  // Woodbury cores (3D)
  auto build_Woodbury_core_3D = [&](TMatrixDSym& K, TVectorD& rhs) -> void
  { 
    // std::cerr << "Building Woodbury core for 3D" << std::endl;
    K.ResizeTo(3,3); K.Zero();
    rhs.ResizeTo(3); rhs.Zero();

    TMatrixDSym Wsum(3); Wsum.Zero();
    if (cov_prior) Wsum += Wprior;

    for (size_t i=0;i<N;++i){
      if (w[i] < wcutoff) continue;
      Wsum += w[i] * W[i];
      tmpWu = W[i] * t[i]; // Wu = W_i t_i
                         
      K.Rank1Update(tmpWu, - (w[i]/d[i]));

      rhs += (tmpWu * (w[i]/d[i] * g[i]));
    }
    K += Wsum;

  };

  auto line_search = [&](const std::vector<double>& s_cur,
                         const std::vector<double>& step,
                         const double J_old,
                         std::vector<double>& s_new,
                         TVectorD& v_new,
                         double& J_new) -> bool
  {
    s_new = s_cur;
    double alpha = 1.0;
    double ds_max = 0.;
    for (int bt=0; bt<max_backtrack; ++bt){
      for (size_t i=0;i<N;++i){
        const double clamp = std::min(frac_turn * s_turn[i], absCap);
        const double dsi = std::clamp(alpha*step[i], -clamp, clamp);
        s_new[i] = s_cur[i] + dsi;
        if(w[i] > wcutoff) ds_max = std::max(ds_max, std::abs(dsi));
      }
      reBLUE_and_cost(s_new, v_new,  J_new);
      if (J_new <= J_old) return true;
      // else if (ds_max < s_tol){s_new = s_cur; reBLUE_and_cost(s_new, v_new, vt_new, J_new); return true;}
      alpha *= 0.5;
    }
    return false;
  };

  // ---------------- IRLS outer loop ----------------
  TVectorD v_last = vcurrent;
  double   Jcurrent = 0.0;
  reBLUE_and_cost(s, vcurrent,  Jcurrent);

  for (int iter = 0; iter < maxOuterIter; ++iter) {
    int accepted = 0;

    for (int it = 0; it < maxInnerIter; ++it) {
      recompute_inner(vcurrent,  Jcurrent); // fills g,d,Hdiag, and chi2i
      // std::cerr<<"Iter "<<iter<<" inner "<<it<<" chi2 "<<Jcurrent<<" /  accepted "<<accepted<<" / "<<N<<"\n";
      std::vector<double> ds(N, 0.0);
      double ds_max = 0.0;

      TMatrixDSym K3(3); TVectorD rhs3(3); 
      build_Woodbury_core_3D(K3, rhs3);
      TVectorD z3(3);
      if (!SolveSPD(K3, rhs3, z3)) { std::cerr<<"K3 solve failed\n"; return false; }
      for (size_t i=0;i<N;++i){
        // if (w[i] < wcutoff) continue;
        double dsi = -g[i]/d[i];
        tmpWu = W[i] * t[i];
        dsi  -= (dotD(tmpWu, z3) / d[i]);
        ds[i] = dsi;
        if (w[i] > wcutoff) ds_max = std::max(ds_max, std::abs(dsi));
      }
      
      // line search
      std::vector<double> s_try = s;
      TVectorD v_try = vcurrent;  double J_try = Jcurrent;
      if (!line_search(s, ds, Jcurrent, s_try, v_try,  J_try)) {
        // fallback: uncoupled step
        std::vector<double> pure(N);
        for (size_t i=0;i<N;++i) pure[i] = - g[i] / Hdiag[i];
        if (!line_search(s, pure, Jcurrent, s_try, v_try,  J_try)) break;
      }

      // accept
      v_last = vcurrent; 
      s.swap(s_try); vcurrent = v_try;  Jcurrent = J_try;
      ++accepted;

      const double dv = std::sqrt((vcurrent - v_last).Norm2Sqr());
      // std::cerr << "Last vertex move "<<dv<<" mm \n";
      if (ds_max < s_tol && dv < v_tol) break;
    } // inner

    // refresh metrics at new s and reBLUE
    recompute_outer(vcurrent, Jcurrent);

    // IRLS reweight
    // std::cerr<<"Weights: "<<std::endl;
    double max_dw = 0.0;
    if (useWeights && N > 3 && accepted > 0) {
      for (size_t i=0;i<N;++i) {
        double wi;
        wi = sigmoid(chi2i[i], c0, beta);
        
        if (wi < wEps) wi = wEps;
        max_dw = std::max(max_dw, std::abs(wi - w[i]));

        w[i] = wi;

        // std::cerr << i << " " << w[i] << std::endl;
      }
    }

    TVectorD v_try(3);
    recompute_outer(v_try, Jcurrent);
    double dv = std::sqrt((v_try - vcurrent).Norm2Sqr());
    vcurrent = v_try;
    if (max_dw < w_tol || dv < v_tol) break;    
  } // IRLS

  recompute_inner(vcurrent, Jcurrent);

  // -------- Final outputs --------
  TVectorD vseed = vstar_io;
  vstar_io = vcurrent;

  // chi2 per track (final)
  chi2i_out = chi2i;

  // final chi2: only tracks with w[i] > opts.weightcut
  chi2_out = 0.0;
  double totw = 0;
  int nactive = 0;
  for (size_t i=0;i<N;++i)
    if (w[i] > weightCut){ 
      chi2_out += chi2i[i]; nactive++; totw += w[i];
    }
    // else w[i] = 0.;
  
  if (nactive <= 1 || chi2_out <= 0. || totw <= 1.5) {
    if (fVerbose) std::cerr << "Weights collapsed, starting over with new seed" << std::endl;
    return false;}

  // --- profiled vertex covariance (matches Schur/Woodbury order) ---
  {
    
    // ---------- 3D (position only) ----------
    TMatrixDSym K3(3); TVectorD rhs3(3);
    // K3 = S_v - U^T D^{-1} U
    build_Woodbury_core_3D(K3, rhs3); 

    TMatrixDSym Cov3(3); Cov3.Zero();
    if (InvertSPD(K3, Cov3)) {
      cov_out.ResizeTo(3,3);
      for (int r=0;r<3;++r)
        for (int c=r;c<3;++c)
          cov_out(r,c) = cov_out(c,r) = Cov3(r,c);
    } else {
      cov_out.ResizeTo(3,3); cov_out.Zero();
    }

  }

  //Compute vertex time and uncertainty
  // double vtstar = 0;
  double Wt = 0;
  double tauwt = 0;
  for (size_t i=0;i<N;++i) {
    if (!hasTime[i]) continue;
    const auto& tv = *tvs[i];
    tau[i] = trkT(tv, s[i]);
    tauwt += tau[i] * w[i]/sigtau2[i];
    Wt += w[i]/sigtau2[i];
  }

  double vtstar = Wt > 0 ? tauwt / Wt : 0;
  double vartstar = Wt > 0 ? 1.0 / Wt : 0;

  vt_io = vtstar;
  vart_out = vartstar;


  if (w_io) *w_io = w;
  s_io = s;
  
  // std::cerr << "Coupled Metric Fit last s:" << std::endl;
  // for (size_t i=0;i<N;++i)
  //   std::cerr << "s[" << i << "]  " << s[i] << std::endl;

  if (fVerbose && useWeights) {
    TVectorD distvec = vseed - vstar_io;
    double dist = std::sqrt(dotD(distvec, distvec));
    std::cout << "Vertex Seed at (" << vseed(0) << ", " << vseed(1) << ", " << vseed(2) << ")" << std::endl;
    std::cout << "Fitted Vertex at (" << vstar_io(0) << ", " << vstar_io(1) << ", " << vstar_io(2) << ")" << std::endl;
    std::cout << "Distance to seed = " << dist << std::endl;
    std::cout << "Vertex time at " << vt_io << " +/- " << vart_out << " [mm]" << std::endl;
    std::cout << "Final chi2 = " << chi2_out << std::endl;
    std::cout << "Active tracks = " << nactive << " out of " << N << " tracks total " <<std::endl;
  }

  return true;
}

// Multi-looper aware pre-gate (arc-length world).
// keep == true  -> pair is feasible;
// keep == false -> reject outright.
//
// Tunables:
//   epsXY_mm  : XY circle gap tolerance (1 mm safe default).
bool GraphDisplacedVertexFinder::PreGatePair(
  const ViewEntry& t1, const ViewEntry& t2,
  double epsXY_mm    // e.g. 0.5-1mm
  ) const
{
  double x1c = t1.xc, y1c = t1.yc, r1 = std::abs(t1.R);
  double x2c = t2.xc, y2c = t2.yc, r2 = std::abs(t2.R);

  // quick XY feasibility
  const double dx = x2c - x1c, dy = y2c - y1c, d = std::hypot(dx,dy);
  const double gap_out = d - (r1 + r2);
  const double gap_in  = std::max(r1,r2) - (d + std::min(r1,r2));
  if (std::max(gap_out, gap_in) > epsXY_mm) return false;

  return true;
}

// Try to fit two tracks to a common vertex but first check that their trajectories overlap in the (x,y) plane
const PairFit& GraphDisplacedVertexFinder::GetPairFit(const Candidate* a,
                                                      const Candidate* b,
                                                      bool   usePairGuard) const
{
  PairKey key(a,b);
  auto it = m_ecache.pairFit.find(key);
  if (it != m_ecache.pairFit.end()) return it->second;

  const ViewEntry& t1 = viewOf(a);
  const ViewEntry& t2 = viewOf(b);

  double s1p = 0.0, s2p = 0.0;

  PairFit pf; // defaults: not ok
  pf.ok = false;
  if (usePairGuard) {
    bool guard_ok = PreGatePair(t1, t2, 1.0);
    if (!guard_ok) return m_ecache.pairFit.emplace(key, std::move(pf)).first->second;
  }

  TVectorD vstart(3); vstart.Zero();
  std::vector<double> s_seed(2, 0.0);

  // --- Try two-track POCA seed
  
  bool use_parallel_fallback = false;

  // quick parallel check at s=0 (unit-speed assumed; normalize if not guaranteed)
  {
    TVectorD u1 = trkdXds(t1, 0.0);
    TVectorD u2 = trkdXds(t2, 0.0);
    const double n1 = std::sqrt(dotD(u1,u1));
    const double n2 = std::sqrt(dotD(u2,u2));
    if (n1 > 0.0) u1 *= 1.0/n1;
    if (n2 > 0.0) u2 *= 1.0/n2;
    const double c = std::abs(dotD(u1,u2));
    use_parallel_fallback = (1.0 - c) < 1e-6; // nearly parallel
  }

  if (!use_parallel_fallback) {
    auto s_pair = GetPocaT2T(t1, t2, &s1p, &s2p,/*maxIter=*/20, /*tol=*/1e-5);
    s1p = s_pair.first;
    s2p = s_pair.second;
    s_seed[0] = s1p;
    s_seed[1] = s2p;
    // simple center for vstart (CoupledMetricFit will immediately re-BLUE anyway)
    vstart = trkX(t1, s1p); vstart += trkX(t2, s2p); vstart *= 0.5;
    double dist = std::sqrt(dotD(trkX(t1, s1p) - trkX(t2, s2p), trkX(t1, s1p) - trkX(t2, s2p)));
    TVectorD res = trkX(t1, s1p) - trkX(t2, s2p);
    TMatrixDSym Covs = trkCx(t1, s1p) + trkCx(t2, s2p);
    TMatrixDSym Covinv(3);
    InvertSPD(Covs, Covinv);
    double mahalanobis = Covinv.Similarity(res);
    if (mahalanobis > 1000.0){ // rough threshold that should reject the worst but keep decent candidates
      // std::cerr << "Current distance between tracks is " << dist << std::endl;
      // std::cerr << "With mahalanobis " << mahalanobis << std::endl;
      // std::cerr << "Pair fit will fail, bailing out" << std::endl;
      return m_ecache.pairFit.emplace(key, std::move(pf)).first->second;
    }
  } else {
    // --- Parallel fallback: neutral center + outside-in phases
    TVectorD x10 = trkX(t1, 0.0);
    TVectorD x20 = trkX(t2, 0.0);
    vstart  = x10; vstart += x20; vstart *= 0.5;

    TVectorD u1 = trkdXds(t1, 0.0);
    TVectorD u2 = trkdXds(t2, 0.0);
    const double n1 = std::sqrt(dotD(u1,u1));
    const double n2 = std::sqrt(dotD(u2,u2));
    if (n1 > 0.0) u1 *= 1.0/n1;
    if (n2 > 0.0) u2 *= 1.0/n2;

    TVectorD r02 = x20; r02 -= x10;
    const double sign = (dotD(r02, u1) >= 0.0) ? +1.0 : -1.0;

    const double s1_scale = turn_scale(t1.p(2), t1.n);
    const double s2_scale = turn_scale(t2.p(2), t2.n);
    const double k = 3.0; // modest “outside-in” push

    s_seed[0] = std::clamp(-sign * k * s1_scale, -fFitOpts.absCap, fFitOpts.absCap);
    s_seed[1] = std::clamp(+sign * k * s2_scale, -fFitOpts.absCap, fFitOpts.absCap);
  }

  // --- Run CoupledMetricFit on the two tracks
  const std::vector<const ViewEntry*> tvs{ &t1, &t2 };

  std::vector<double>        s_io = s_seed;
  std::vector<double>        chi2i(2, 0.0);
  double                     chi2  = 0.0;
  double                     vt    = 0.0;       // if timing off, remains 0
  TMatrixDSym                cov(3); cov.Zero();
  std::vector<double>        *w_io = nullptr;  
  TVectorD                   vfit  = vstart;
  double                     vart  = 0.0;

  FitOpts opts = fFitOpts;
  opts.useWeights = false;
  opts.useTiming = false;

  bool ok = CoupledMetricFit(tvs,
                             s_io,
                             w_io,
                             chi2i,
                             chi2,
                             vfit,
                             vt,
                             cov,
                             vart,
                             /*v_prior=*/nullptr,
                             /*cov_prior=*/nullptr,
                             opts);

  
    //Check timing consistency if enabled
  if (fUseTiming && t1.hasTime && t2.hasTime){
    double T1 = trkT(t1, s_io[0]);
    double T2 = trkT(t2, s_io[1]);
    double sig1 = t1.sigma_t2;
    double sig2 = t2.sigma_t2;

    double dist = (T1 - T2)*(T1 - T2)/(sig1 + sig2);

    if (dist > fTimeGate) ok = false;
  }

  if (!ok) {
    // Cache as failure with minimal diagnostics
    pf.ok = false;
    pf.v    = vfit;
    pf.Cv   = cov;
    pf.chi2 = chi2;
    pf.l1   = s_io[0];
    pf.l2   = s_io[1];
    return m_ecache.pairFit.emplace(key, std::move(pf)).first->second;
  }

  // --- Success: fill and cache
  // if (fUseTiming && t1.hasTime && t2.hasTime) ndf = 2;
  pf.ok   = true;
  pf.v    = vfit;
  pf.Cv   = cov;
  pf.chi2 = chi2;
  pf.l1   = s_io[0];
  pf.l2   = s_io[1];

  return m_ecache.pairFit.emplace(key, std::move(pf)).first->second;
}

const PairFit& GraphDisplacedVertexFinder::GetPairFit_guarded(const Candidate* a,
                                                              const Candidate* b) const {
  return GetPairFit(a,b, fUsePairGuard);
}

// --- BLUE combiner for a set of (v_k, C_k). Reuse for N-track seed. ---
bool GraphDisplacedVertexFinder::BlueCombine(const std::vector<TVectorD>& vs,
                                             const std::vector<TMatrixDSym>& Cs,
                                             TVectorD& v_out, TMatrixDSym& C_out) const
{
  const int K = (int)vs.size();
  if (K == 0) return false;

  // dimension check (expect 3)
  const int n = vs[0].GetNrows();
  if (n != 3) return false;
  for (int k=0;k<K;++k) if (vs[k].GetNrows()!=n || Cs[k].GetNrows()!=n) return false;

  TMatrixDSym H(n); H.Zero();
  TMatrixD    b(n,1); b.Zero();

  for (int k=0; k<K; ++k){
    TMatrixD W(3,3); 
    if (!InvertSPD(Cs[k], W)) return false;   // W = C^{-1} (symmetrized)

    // H += W
    for (int r=0; r<n; ++r)
      for (int c=0; c<n; ++c)
        H(r,c) += W(r,c);

    // b += W * v_k
    for (int r=0; r<n; ++r){
      double acc = 0.0;
      for (int c=0; c<n; ++c) acc += W(r,c) * vs[k](c);
      b(r,0) += acc;
    }
  }

  TMatrixD vcol = b;      // copy RHS

  if(!SolveSPD(H, b, vcol)) return false;

  v_out.ResizeTo(n);
  for (int i=0;i<n;++i) v_out(i) = vcol(i,0);

  // C_out = H^{-1} (covariance of the BLUE)

  TMatrixD I(n,n); I.UnitMatrix();
  TMatrixD Hinv = I;
  if (!InvertSPD(H, Hinv)) return false;

  // write back as symmetric
  C_out.ResizeTo(n,n);
  C_out.Zero();
  for (int i=0;i<n;++i)
    for (int j=0;j<n;++j)
      C_out(i,j) = 0.5*(Hinv(i,j)+Hinv(j,i));

  return true;
}

const TripFit& GraphDisplacedVertexFinder::GetTripFit(
    const Candidate* a,
    const Candidate* b,
    const Candidate* c,
    const FitOpts&          opts,
    double                  chi2PairCut) const
{
  TripKey key(a,b,c);
  if (auto it = m_ecache.tripFit.find(key); it != m_ecache.tripFit.end())
    return it->second;

  const ViewEntry& tA = viewOf(a);
  const ViewEntry& tB = viewOf(b);
  const ViewEntry& tC = viewOf(c);

  // -------- gather pairs (unchanged gating logic) --------
  const PairFit& pAB = GetPairFit_guarded(a,b);
  const PairFit& pAC = GetPairFit_guarded(a,c);
  const PairFit& pBC = GetPairFit_guarded(b,c);

  TripFit tf; // !ok by default; will be cached on all return paths
  tf.ok = false;
  tf.pair_ab = (pAB.ok && pAB.chi2 <= chi2PairCut);
  tf.pair_ac = (pAC.ok && pAC.chi2 <= chi2PairCut);
  tf.pair_bc = (pBC.ok && pBC.chi2 <= chi2PairCut);

  // If not all pairs pass the gate, bail early 
  if (!tf.pair_ab || !tf.pair_ac || !tf.pair_bc)
    return m_ecache.tripFit.emplace(key, std::move(tf)).first->second;

  // -------- build track list --------
  const std::vector<const ViewEntry*> tvs{ &tA, &tB, &tC };

  // -------- seed phases & vertex via BLUE over pairs --------
  std::vector<double> s_seed(3, 0.0);
  std::vector<double> w_seed;                 // optional; can stay empty
  TVectorD            v_seed(3); v_seed.Zero();
  TMatrixDSym         C_seed(3); C_seed.Zero();

  bool have_seed = false;

  
  // - BLUE over the pair vertices that passed 
  // - phases = PoCA-to-vertex per track.
   {
    std::vector<TVectorD>    vps;
    std::vector<TMatrixDSym> Cps;
    if (tf.pair_ab) { vps.push_back(pAB.v); Cps.push_back(pAB.Cv); }
    if (tf.pair_ac) { vps.push_back(pAC.v); Cps.push_back(pAC.Cv); }
    if (tf.pair_bc) { vps.push_back(pBC.v); Cps.push_back(pBC.Cv); }

    if (!vps.empty()) {
      TVectorD vBLUE(3); vBLUE.Zero();
      TMatrixDSym CB(3); CB.Zero();
      if (BlueCombine(vps, Cps, vBLUE, CB)) {
        v_seed = vBLUE;
        // get phases from PoCA-to-vertex (fast scalar 1D solve per track)
        for (int i = 0; i < 3; ++i) {
          const ViewEntry& tv = *(tvs[i]);
          double dummyChi=0.0;
          s_seed[i] = GetPocaT2V(tv, v_seed, /*s0=*/0.0, /*maxIter=*/25, /*tol=*/1e-4);
        }
        have_seed = true;
      }
    }
  }

  // -------- seed phases & vertex via EuclideanSeed as fallback --------
  if (!have_seed) {
    have_seed = EuclideanSeed(tvs, s_seed, &w_seed, v_seed, &C_seed, opts);

  }

  if (!have_seed) {
    // As a last resort: start at (0,0,0) and hope that the metric fitter will sort it out
    v_seed.Zero();
    s_seed = {0.0, 0.0, 0.0};
  }

  // -------- coupled metric fit  --------
  std::vector<double> chi2i(3, 0.0);
  double              chi2  = 0.0;
  double              vt    = 0.0;        // will remain 0 if timing disabled
  TMatrixDSym         C_out(3); C_out.Zero();
  TVectorD            v_fit  = v_seed;
  double              vart   = 0.0;

  // (Optionally pass prior here; using nullptrs by default)
  bool ok = CoupledMetricFit(tvs,
                             s_seed,         // in/out phases
                             /*w_io=*/nullptr,
                             chi2i,
                             chi2,
                             v_fit,          // in/out vertex
                             vt,
                             C_out,
                             vart,
                             /*v_prior=*/nullptr,
                             /*cov_prior=*/nullptr,
                             opts);

  if (!ok) {
    // Cache as failure
    return m_ecache.tripFit.emplace(key, std::move(tf)).first->second;
  }
  int ndf = 3;

  // -------- success: fill and cache --------
  tf.ok   = true;
  tf.v    = v_fit;
  tf.Cv   = C_out;
  tf.chi2 = chi2/ndf;
  tf.l1   = s_seed[0];
  tf.l2   = s_seed[1];
  tf.l3   = s_seed[2];

  return m_ecache.tripFit.emplace(key, std::move(tf)).first->second;
}


// ------------------ Refit track parameters to the vertex constraint
// --- refresh derived kinematics in a ViewEntry after p/Cp changes ---
inline void GraphDisplacedVertexFinder::RefreshDerived(ViewEntry& e) const {
  const double ct = e.p(4), C = e.p(2);
  const double n  = std::sqrt(1.0 + ct*ct);
  e.n     = n;
  e.omega = C/n;
  e.kappa = 2.0*C/(n*n);
  e.tau   = 2.0*C*ct/(n*n);
}

// 3D Kalman measurement update at a given phase s (no chi2 gate).
// - Updates tv.p and tv.Cp in place and refreshes derived fields.
// - v_meas: fitted vertex (3), R: vertex covariance (3x3 SPD).
bool GraphDisplacedVertexFinder::KalmanUpdateTrackAtVertex3D(
    ViewEntry&         tv,
    double             s,
    const TVectorD&    v_meas,
    const TMatrixDSym& R) const
{ 
  double dum=0;
  GetMetricPOCAT2V(tv, v_meas, R, s, dum);
  // Predict and Jacobian at (p,s)
  TVectorD x  = trkX(tv, s);     // 3x1
  TMatrixD H  = trkJx(tv, s);    // 3x5
  const TMatrixDSym& P = tv.Cp;  // 5x5 prior

  // Innovation y = z - h(p)
  TVectorD y = v_meas; y -= x;   // 3x1

  // Innovation covariance S = H P H^T + R
  TMatrixD HP(3,5);      HP.Mult(H, P);
  TMatrixD S (3,3);      S.Mult(HP, TMatrixD(TMatrixD::kTransposed, H));
  TMatrixDSym Rj = R;
  // Rj(0,0)+=jitterR; Rj(1,1)+=jitterR; Rj(2,2)+=jitterR;
  for (int i=0;i<3;++i) for (int j=i;j<3;++j) S(i,j) += Rj(i,j);

  // Symmetrize S and solve once: S * K^T = H P
  TMatrixDSym Ssym(3);
  for (int i=0;i<3;++i)
    for (int j=i;j<3;++j)
      Ssym(i,j) = 0.5*(S(i,j) + S(j,i));

  TMatrixD Kt(3,5);
  if (!SolveSPD(Ssym, HP, Kt)) return false;  // Kt = S^{-1} (H P)
  TMatrixD K(5,3); K.Transpose(Kt);

  // State update: p ← p + K y
  TMatrixD ycol(3,1); ycol(0,0)=y(0); ycol(1,0)=y(1); ycol(2,0)=y(2);
  TMatrixD Ky(5,1); Ky.Mult(K, ycol);
  for (int j=0;j<5;++j) tv.p(j) += Ky(j,0);

  // Covariance update (Joseph)
  TMatrixD I5(5,5); I5.UnitMatrix();
  TMatrixD KH(5,5); KH.Mult(K, H);
  TMatrixD A (I5);  A -= KH;

  TMatrixD AP (5,5); AP.Mult(A, P);
  TMatrixD AT (TMatrixD(TMatrixD::kTransposed, A));
  TMatrixD APAT(5,5); APAT.Mult(AP, AT);

  TMatrixD KR (5,3); KR.Mult(K, Rj);
  TMatrixD KT (TMatrixD(TMatrixD::kTransposed, K));
  TMatrixD KRKT(5,5); KRKT.Mult(KR, KT);

  TMatrixD Pnew(5,5); Pnew = APAT; Pnew += KRKT;
  for (int r=0;r<5;++r)
    for (int c=r;c<5;++c)
      tv.Cp(r,c) = tv.Cp(c,r) = 0.5*(Pnew(r,c) + Pnew(c,r));

  RefreshDerived(tv);
  return true;
}

void GraphDisplacedVertexFinder::RefitTracksToVertex(Cluster& cluster) const 
{
  for (size_t i = 0; i < cluster.tracks.size(); ++i) {
    if (cluster.trackWeights[i] < fweightCut) continue;
    const Candidate* t = cluster.tracks[i];
    ViewEntry& tv = ViewOf(t);
    double s = cluster.trackPhases[i];
    (void)KalmanUpdateTrackAtVertex3D(tv, s, cluster.fittedPos, cluster.fittedCov);
  }
}



bool GraphDisplacedVertexFinder::SeedFromTriplets(
    const std::vector<const Candidate*>& tracks,
    const FitOpts& opts,
    TVectorD& v_seed, TMatrixDSym& C_seed,
    double chi2TripCut, 
    int    maxTriplets         
) const
{
  const int N = (int)tracks.size();
  if (N < 3) return false;

  FitOpts tripopts = opts;
  tripopts.useWeights = false; 
  tripopts.useTiming = false;
  std::vector<TVectorD>    vs;  vs.reserve(std::min(maxTriplets, N*(N-1)*(N-2)/6));
  std::vector<TMatrixDSym> Cs;  Cs.reserve(vs.capacity());

  int used = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = i+1; j < N; ++j) {
      for (int k = j+1; k < N; ++k) {
        if (maxTriplets > 0 && used >= maxTriplets) break;
        ++used;

        const Candidate* a = tracks[i];
        const Candidate* b = tracks[j];
        const Candidate* c = tracks[k];

        // Pull from cache (runs pair gates internally)
        const TripFit& tf = GetTripFit(a, b, c, tripopts, /*chi2PairCut=*/fchi2PairCut);
        if (!tf.ok) continue;                    // failed triplet
        if (!(tf.chi2 < chi2TripCut)) continue;  

        
        vs.push_back(tf.v);
        Cs.push_back(tf.Cv);
        
      }
    }
  }

  // if (fVerbose) {
  //   std::cout << "[SeedFromTriplets] Tried " << used << " triplets" << std::endl;
  //   std::cout << "[SeedFromTriplets] Found " << vs.size() << " valid triplets" << std::endl;  
  // }
  if (vs.empty()) return false;

  // BLUE across accepted triplets
  TVectorD    v_out(3);
  TMatrixDSym C_out(3);
  if (!BlueCombine(vs, Cs, v_out, C_out)) return false;

  v_seed = v_out;
  C_seed = C_out;
  return true;
}


FitResult GraphDisplacedVertexFinder::RobustFit(
  const std::vector<const Candidate*>& tracks,
  const FitOpts& opts,
  const TVectorD* vseedOpt) const
{
  FitResult out; out.fittedCov.ResizeTo(3,3); out.fittedCov.Zero();
  const size_t N = tracks.size();
  //can't fit less than two tracks obviously (should be updated so that when a seed is given we can create a pseudo track)
  if (N < 2) {out.chi2=999.0, out.ndf=0; return out;}

  if (fVerbose) std::cerr << "Fitting " << N << " tracks" << std::endl;
  
  std::vector<const ViewEntry*> tvs;
  tvs.reserve(N);
  
  for (const Candidate* t : tracks) {
    tvs.push_back(&viewOf(t));
  }   

  TVectorD vstar(3); vstar.Zero();
  std::vector<double> s_io;
  std::vector<double> w_io(N, 1.0);
  std::vector<double> chi2i(N, 0.0);
  double chi2=0.;
  double vt=0.;
  TMatrixDSym vCov(3); vCov.Zero();
  double vart;

  if (vseedOpt) {
    vstar = *vseedOpt;
    vt = 0.0;
  }
  else if (opts.useSelfSeeding) {
    bool have_seed = EuclideanSeed(tvs, s_io, &w_io, vstar, &vCov, opts);
    double tot_w = 0;
    for (size_t i=0;i<N;++i) tot_w += w_io[i];

    if (!have_seed || tot_w < 1.8){
      if (fVerbose) std::cerr << "EuclideanSeed silently failed, falling back to triplet BLUE" << std::endl;
      bool have_seed = SeedFromTriplets(tracks, opts, vstar, vCov, fchi2TripCut, 500);
      if (!have_seed) {
        if (fVerbose) std::cerr << "Triplet BLUE also failed, giving up and trying without seed" << std::endl;
        }
    }
  }
  else{
    for (size_t i=0;i<N;++i) s_io.push_back(0.0);
  }
  bool fit_ok = false;
  if (opts.useBeamConstraint){
    TVectorD BeamPos = fBeamPos;
    TMatrixDSym BeamCov = fBeamCov;
    fit_ok = CoupledMetricFit(tvs, s_io, &w_io, chi2i, chi2, vstar, vt, vCov, vart, &BeamPos, &BeamCov, opts);
  } 
  else{
    fit_ok = CoupledMetricFit(tvs, s_io, &w_io, chi2i, chi2, vstar, vt, vCov, vart, nullptr, nullptr, opts);
  }

  //If fit failed for whatever reason start over without seed (need better fallback here eventually)
  if (!fit_ok){
    if (fVerbose) std::cerr << "CoupledMetricFit failed, trying with seed from triplet BLUE" << std::endl;
    s_io.resize(N);
    vstar.Zero();
    bool have_seed = SeedFromTriplets(tracks, opts, vstar, vCov, fchi2TripCut, 500);
    if (!have_seed) {
      if (fVerbose) std::cerr << "Triplet BLUE also failed, giving up and trying without seed" << std::endl;
      vstar.Zero();
    }
    for (size_t i=0;i<N;++i){
      s_io[i] = 0.0;
      w_io[i] = 1.0;
    }
    if (opts.useBeamConstraint){
    TVectorD BeamPos = fBeamPos;
    TMatrixDSym BeamCov = fBeamCov;
    fit_ok = CoupledMetricFit(tvs, s_io, &w_io, chi2i, chi2, vstar, vt, vCov, vart, &BeamPos, &BeamCov, opts);
      } 
    else{
      fit_ok = CoupledMetricFit(tvs, s_io, &w_io, chi2i, chi2, vstar, vt, vCov, vart, nullptr, nullptr, opts);
    }
  }

  if (!fit_ok){
    if (fVerbose) std::cerr << "CoupledMetricFit still failed, trying without seed" << std::endl;
    vstar.Zero();
    if (opts.useBeamConstraint){
    TVectorD BeamPos = fBeamPos;
    TMatrixDSym BeamCov = fBeamCov;
    fit_ok = CoupledMetricFit(tvs, s_io, &w_io, chi2i, chi2, vstar, vt, vCov, vart, &BeamPos, &BeamCov, opts);
      } 
    else{
      fit_ok = CoupledMetricFit(tvs, s_io, &w_io, chi2i, chi2, vstar, vt, vCov, vart, nullptr, nullptr, opts);
    }
  }

  if (!fit_ok){
    if (fVerbose) std::cerr << "CoupledMetricFit still failed, giving up!" << std::endl;
    out.ndf=-1;
    return out;
  }


  int ndf = -3;
  for (size_t i=0;i<N;++i)
  {
     if (w_io[i]>opts.weightCut) ndf += 2;
  }

  if (opts.useBeamConstraint) ndf += 3;

  int nactive = 0;
  for (size_t i=0;i<N;++i){
    if (w_io[i] > opts.weightCut) ++nactive;
  }

  out.tracks = tracks;
  out.fittedPos = vstar;
  out.fittedCov = vCov;
  out.fittedTime = vt;
  out.fittedTimeErr = (vart>0)? std::sqrt(vart) : 0.0;
  out.chi2 = chi2;
  out.ndf = ndf;
  out.trackWeights = w_io;
  out.trackChi2    = chi2i;
  out.trackPhases = s_io;
  out.nactive = nactive;
  return out;

}

FitResult GraphDisplacedVertexFinder::RobustFit(Cluster& cl, const FitOpts& opts) const {
  FitResult out; out.fittedCov.ResizeTo(3,3); out.fittedCov.Zero();
  const size_t N = cl.tracks.size();
  //can't fit less than two tracks obviously (should be updated so that when a seed is given we can create a pseudo track)
  if (N < 2) {out.chi2=0., out.ndf=-1; return out;}
  
  std::vector<const ViewEntry*> tvs;
  tvs.reserve(N);
  
  for (const Candidate* t : cl.tracks) {
    tvs.push_back(&viewOf(t));
  } 
  TVectorD vstar = cl.fittedPos;
  std::vector<double> s_io;
  std::vector<double> w_io(N, 1.0);
  std::vector<double> chi2i(N, 0.0);
  double chi2=0.;
  double vt=0.;
  TMatrixDSym vCov = cl.fittedCov;
  double vart;

  if (cl.trackPhases.size() == N){
    s_io = cl.trackPhases;
  }
  else{
    s_io.resize(N);
    for (size_t i=0;i<N;++i){
      double s_loc, chi2_loc;
      GetMetricPOCAT2V(*tvs[i], vstar, vCov, s_loc, chi2_loc);
      s_io[i] = s_loc;
    }
  }

  bool fit_ok = false;
  if (opts.useBeamConstraint){
    TVectorD BeamPos = fBeamPos;
    TMatrixDSym BeamCov = fBeamCov;
    fit_ok = CoupledMetricFit(tvs, s_io, &w_io, chi2i, chi2, vstar, vt, vCov, vart, &BeamPos, &BeamCov, opts);
  } 
  else{
    fit_ok = CoupledMetricFit(tvs, s_io, &w_io, chi2i, chi2, vstar, vt, vCov, vart, nullptr, nullptr, opts);
  }

  //If fit failed for whatever reason start over without seed (need better fallback here eventually)
  if (!fit_ok){
    s_io.resize(N);
    vstar.Zero();
    for (size_t i=0;i<N;++i){
      s_io[i] = 0.0;
      w_io[i] = 1.0;
    }
    if (opts.useBeamConstraint){
    TVectorD BeamPos = fBeamPos;
    TMatrixDSym BeamCov = fBeamCov;
    fit_ok = CoupledMetricFit(tvs, s_io, &w_io, chi2i, chi2, vstar, vt, vCov, vart, &BeamPos, &BeamCov, opts);
      } 
    else{
      fit_ok = CoupledMetricFit(tvs, s_io, &w_io, chi2i, chi2, vstar, vt, vCov, vart, nullptr, nullptr, opts);
    }
  }

  if (!fit_ok){
    out.ndf=-1;
    return out;
  }

  int ndf = -3;
  for (size_t i=0;i<N;++i)
  {
    if (w_io[i]>opts.weightCut) ndf += 2;
  }
  

  if (opts.useBeamConstraint) ndf += 3;

  int nactive = 0;
  for (size_t i=0;i<N;++i){
    if (w_io[i] > opts.weightCut) ++nactive;
  }

  out.tracks = cl.tracks;
  out.fittedPos = vstar;
  out.fittedCov = vCov;
  out.fittedTime = vt;
  out.fittedTimeErr = (vart>0)? std::sqrt(vart) : 0.0;
  out.chi2 = chi2;
  out.ndf = ndf;
  out.trackWeights = w_io;
  out.trackChi2    = chi2i;
  out.trackPhases = s_io;
  out.nactive = nactive;
  return out;
}

void GraphDisplacedVertexFinder::RobustFitInPlace(Cluster& cl, const FitOpts& opts) const {
  FitResult out = RobustFit(cl, opts);
  cl.fittedPos = out.fittedPos;
  cl.fittedCov = out.fittedCov;
  cl.fittedTime = out.fittedTime;
  cl.fittedTimeErr = out.fittedTimeErr;
  cl.chi2 = out.chi2;
  cl.ndf = out.ndf;
  cl.trackWeights = out.trackWeights;
  cl.trackChi2    = out.trackChi2;
  cl.trackPhases = out.trackPhases;
  cl.nactive = out.nactive;
}



// ================== Graph Building ====================

std::vector<PairEdge>
GraphDisplacedVertexFinder::BuildScoredPairGraph(
    const std::vector<const Candidate*>& trk, const double chi2PairCut) const
{
  const size_t N = trk.size();
  std::vector<PairEdge> E; E.reserve(N*4);

  const double c0   = (fc0  > 0 ? fc0  : 9.0);
  const double beta = 2.0;
  const double wMin = fweightCut;


  for (size_t i=0; i<N; ++i) for (size_t j=i+1; j<N; ++j) {
    const PairFit& pf = GetPairFit_guarded(trk[i], trk[j]); // uses caches + circle & optionally time guard
    if (!pf.ok) continue;
    if (!std::isfinite(pf.chi2) || pf.chi2 > chi2PairCut) continue;

    const double w = sigmoid(pf.chi2, c0, beta);
    if (w < wMin) continue;

    PairEdge e;
    e.u    = (int)i;
    e.v    = (int)j;
    e.chi2 = pf.chi2;
    e.w    = w;
    E.push_back(std::move(e));
  }
  return E;
}

std::vector<std::vector<std::pair<size_t,double>>>
GraphDisplacedVertexFinder::AdjacencyFromPairEdges(const std::vector<PairEdge>& E, size_t N) const
{
  std::vector<std::unordered_map<size_t,double>> tmp(N);
  for (const auto& e : E) {
    auto upd = [&](size_t a, size_t b, double w){
      auto& m = tmp[a];
      auto it = m.find(b);
      if (it==m.end()) m[b]=w; else it->second = std::max(it->second, w);
    };
    upd(e.u, e.v, e.w);
    upd(e.v, e.u, e.w);
  }
  std::vector<std::vector<std::pair<size_t,double>>> adjW(N);
  for (size_t u=0; u<N; ++u) {
    adjW[u].reserve(tmp[u].size());
    for (auto& kv : tmp[u]) adjW[u].push_back({kv.first, kv.second});
    std::sort(adjW[u].begin(), adjW[u].end(),
              [](auto& a, auto& b){ return a.second > b.second; });
  }
  return adjW;
}


void GraphDisplacedVertexFinder::BuildTriangleSupportWithTriplets(
    const std::vector<std::vector<size_t>>& adj,
    const std::vector<const Candidate*>& tracks,
    double chi2PairCut,
    double chi2TripletMax,
    double condMax,
    std::vector<int>& triSup_out,
    std::vector<double>& edgeChi2Min) const
{
  const size_t N = tracks.size();
  triSup_out.assign(N*N, 0);
  edgeChi2Min.assign(N*N, std::numeric_limits<double>::infinity());

  // ensure acending node-order so the two-pointer intersections works
  std::vector<std::vector<size_t>> adj_sorted(adj);
  for (auto& nb : adj_sorted){
    std::sort(nb.begin(), nb.end());
    nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
  }

  auto idx = [N](size_t a, size_t b)->size_t { if (a>b) std::swap(a,b); return a*N + b; };

  auto condEst = [&](const TMatrixDSym& M)->double {
    TMatrixDSym Mc(M);
    TMatrixDSymEigen e(Mc);
    TVectorD ev = e.GetEigenValues();
    double lo=1e300, hi=0.0;
    for (int i=0;i<ev.GetNrows();++i) { const double v=std::abs(ev(i)); if (v>0 && v<lo) lo=v; if (v>hi) hi=v; }
    if (lo<=0 || hi<=0) return std::numeric_limits<double>::infinity();
    return hi/lo;
  };
  FitOpts tripops = fFitOpts;
  tripops.useWeights = false;
  tripops.useTiming = false;
  for (size_t u=0; u<N; ++u) {
    const auto& Nu = adj_sorted[u];
    for (size_t a=0; a<Nu.size(); ++a) {
      size_t v = Nu[a]; if (v <= u) continue;
      const auto& Nv = adj_sorted[v];

      // intersect Nu and Nv, require w>v
      size_t iu = 0, iv = 0;
      while (iu < Nu.size() && iv < Nv.size()) {
        size_t x = Nu[iu], y = Nv[iv];
        if (x == y) {
          size_t w = x;
          if (w > v) {
            const TripFit& tf = GetTripFit(tracks[u], tracks[v], tracks[w], tripops, chi2PairCut);
            if (tf.ok && std::isfinite(tf.chi2) && tf.chi2 <= chi2TripletMax
                     && (!condMax || condEst(tf.Cv) <= condMax)) {

              ++triSup_out[idx(u,v)];
              ++triSup_out[idx(u,w)];
              ++triSup_out[idx(v,w)];

              const size_t uv = idx(u,v), uw = idx(u,w), vw = idx(v,w);
              edgeChi2Min[uv] = std::min(edgeChi2Min[uv], tf.chi2);
              edgeChi2Min[uw] = std::min(edgeChi2Min[uw], tf.chi2);
              edgeChi2Min[vw] = std::min(edgeChi2Min[vw], tf.chi2);

            }
          }
          ++iu; ++iv;
        } else if (x < y) ++iu; else ++iv;
      }
    }
  }
}
// Prune edges, only keep edges if track i and track j are within their k mutual nearest neighbors in chi^2 space
void GraphDisplacedVertexFinder::ApplyMutualKNNPruneWeighted(
    std::vector<std::vector<std::pair<size_t,double>>>& adjW, int K) const
{
  if (K < 0) return;
  if (K == 0) { adjW.assign(adjW.size(), {}); return; }

  const size_t N = adjW.size();
  std::vector<std::vector<size_t>> topIdx(N);

  for (size_t u=0; u<N; ++u) {
    auto& nb = adjW[u];
    if ((int)nb.size() > K) {
      std::nth_element(nb.begin(), nb.begin()+K, nb.end(),
                       [](const auto& a, const auto& b){ return a.second > b.second; });
      // sort only top-K for stable tie-breaks
      std::sort(nb.begin(), nb.begin()+K,
                [](const auto& a, const auto& b){
                  if (a.second != b.second) return a.second > b.second;
                  return a.first  < b.first;
                });
      nb.resize(K);
    } else {
      std::sort(nb.begin(), nb.end(),
                [](const auto& a, const auto& b){
                  if (a.second != b.second) return a.second > b.second;
                  return a.first  < b.first;
                });
    }
    topIdx[u].reserve(nb.size());
    for (auto& p : nb) topIdx[u].push_back(p.first);
    std::sort(topIdx[u].begin(), topIdx[u].end());
  }

  // mutual pass
  std::vector<std::vector<std::pair<size_t,double>>> mutual(adjW.size());
  for (size_t u=0; u<N; ++u) {
    for (auto [v,w] : adjW[u]) {
      const auto& tv = topIdx[v];
      if (std::binary_search(tv.begin(), tv.end(), u)) mutual[u].push_back({v,w});
    }
  }

  // symmetrize (keeps best weight if duplicates)
  std::vector<std::vector<std::pair<size_t,double>>> sym(N);
  for (size_t u=0; u<N; ++u) for (auto [v,w] : mutual[u]) {
    sym[u].push_back({v,w}); sym[v].push_back({u,w});
  }
  adjW.swap(sym);
}


// Remove weights from scored pair graph
std::vector<std::vector<size_t>>
GraphDisplacedVertexFinder::StripWeights(
    const std::vector<std::vector<std::pair<size_t,double>>>& adjW) const {
  std::vector<std::vector<size_t>> adj(adjW.size());
  for (size_t u=0; u<adjW.size(); ++u) {
    adj[u].reserve(adjW[u].size());
    for (auto [v,_] : adjW[u]) adj[u].push_back(v);
  }
  return adj;
}

// Extract connected components via DFS
std::vector<std::vector<size_t>>
GraphDisplacedVertexFinder::ConnectedComponents(
    const std::vector<std::vector<size_t>>& adj) const{
  const size_t N = adj.size();
  std::vector<char> vis(N,0);
  std::vector<std::vector<size_t>> comps;
  std::vector<size_t> st; st.reserve(N);

  for (size_t i=0;i<N;++i){
    if (vis[i]) continue;
    st.clear(); st.push_back(i); vis[i]=1;
    std::vector<size_t> comp; comp.reserve(16);
    while(!st.empty()){
      size_t u=st.back(); st.pop_back();
      comp.push_back(u);
      for (size_t v : adj[u]) if (!vis[v]) { vis[v]=1; st.push_back(v); }
    }
    comps.push_back(std::move(comp));
  }
  return comps;
}

// ---------- optional bridge pruning via Tarjan bridge search (keeps only supported strong bridges) ----------
void GraphDisplacedVertexFinder::PruneUnsupportedBridges(
  std::vector<std::vector<size_t>>& adj,
  const std::vector<std::vector<std::pair<size_t,double>>>& /*adjW*/,
  const std::vector<int>& triSup,
  int sMin,
  double strongChi2Cut,
  const std::vector<double>* edgeChi2MinOpt 
) const
{
  const size_t N = adj.size();
  std::vector<int> tin(N,-1), low(N,0), parent(N,-1); int timer=0;
  std::vector<std::pair<size_t,size_t>> bridges;

  auto dfs = [&](auto&& self, size_t u)->void {
    tin[u]=low[u]=timer++;
    for (size_t v : adj[u]) {
      if ((int)v==parent[u]) continue;
      if (tin[v]!=-1) { low[u]=std::min(low[u], tin[v]); continue; }
      parent[v]=(int)u; self(self, v);
      low[u]=std::min(low[u], low[v]);
      if (low[v] > tin[u]) bridges.push_back({u,v});
    }
  };
  for (size_t i=0;i<N;++i) if (tin[i]==-1) dfs(dfs, i);

  auto idx = [N](size_t a,size_t b){ if (a>b) std::swap(a,b); return a*N+b; };
  auto eraseEdge = [&](size_t a,size_t b){
    auto& A = adj[a];
    A.erase(std::remove(A.begin(),A.end(),b), A.end());
  };

  for (auto [u,v] : bridges) {
    const int s = triSup[idx(u,v)];
    double eChi2 = std::numeric_limits<double>::infinity();
    if (edgeChi2MinOpt) eChi2 = (*edgeChi2MinOpt)[idx(u,v)];

    if (s < sMin || eChi2 > strongChi2Cut) {
      eraseEdge(u,v); eraseEdge(v,u);
    }
  }
}


// ---------- greedy maximum-weight matching for 2-track layer ----------
std::vector<std::pair<size_t,size_t>>
GraphDisplacedVertexFinder::GreedyMaxWeightMatching(
  const std::vector<std::vector<std::pair<size_t,double>>>& adjW,
  const std::vector<char>& allowed) const
{
  struct Edge { size_t u,v; double w; };
  std::vector<Edge> edges;
  const size_t N=adjW.size();
  edges.reserve(N*2);

  for (size_t u=0; u<N; ++u) if (allowed[u]) {
    for (auto [v,w]: adjW[u]) if (allowed[v] && u<v) edges.push_back({u,v,w});
  }
  std::sort(edges.begin(), edges.end(),
            [](const Edge& a,const Edge& b){
              if (a.w!=b.w) return a.w>b.w;
              if (a.u!=b.u) return a.u<b.u;
              return a.v<b.v;
            });

  std::vector<char> used(N,0);
  std::vector<std::pair<size_t,size_t>> M; M.reserve(N/2);
  for (const auto& e : edges) {
    if (used[e.u] || used[e.v]) continue;
    used[e.u]=used[e.v]=1;
    M.push_back({e.u,e.v});
  }
  return M;
}


std::vector<Cluster>
GraphDisplacedVertexFinder::GraphClusteringHybrid(std::vector<const Candidate*>& tracks, const GraphOpts& opts) const
{
  std::vector<Cluster> out;
  const size_t N = tracks.size();
  if ((int)N < 2) return out;

  int kNN = opts.kNN;
  double chi2PairCut = opts.chi2PairCut;
  double chi2TripCut = opts.chi2TripCut;
  double chi2AssociationMax = opts.chi2AssociationMax;
  double strongChi2Cut = opts.bridgeCut;
  int minSupport = opts.minSupport;
  bool salvage_pass = opts.salvage_pass;


  std::vector<const Candidate*> seedsList;
  std::vector<char> isSeed(N, 0);

  // fRequireSeed = false;

  if (fRequireSeed) {
    seedsList = SelectSeeds(tracks, fSeedSelector, fMinSeedPT);
    if (!seedsList.empty()) {
      // mark
      for (const Candidate* s : seedsList) {
        auto it = std::find(tracks.begin(), tracks.end(), s);
        if (it != tracks.end()) isSeed[ size_t(it - tracks.begin()) ] = 1;
      }
    }
  }

  //  pair scores (loose physics gate inside)
  auto E    = BuildScoredPairGraph(tracks, chi2PairCut);
  if (fVerbose) std::cout<< "[GraphClusteringHybrid] Pair graph has " << E.size() << " edges" << std::endl;
  auto adjW = AdjacencyFromPairEdges(E, N);

  

  //  multi-track layer: mutual-kNN (opt), triangle support, (opt) bridge prune, DFS
  ApplyMutualKNNPruneWeighted(adjW, kNN);            // keep K<0 to disable
  auto adj = StripWeights(adjW);

  std::vector<int> triSup;
  std::vector<double> edgeChi2Min;

  BuildTriangleSupportWithTriplets(adj, tracks,
                                 /*chi2PairCut=*/chi2PairCut,
                                 /*chi2TripletMax=*/chi2TripCut,
                                 /*condMax=*/1e6,
                                 triSup, edgeChi2Min);

  // auto triSup = TriangleSupport(adj);
  const int sMin = minSupport;                        // e.g. 1 (or 2 in clutter)
  auto triIdx = [N](size_t a, size_t b){ return (a<b) ? a*N + b : b*N + a; };
  for (size_t u=0; u<N; ++u) {
    auto& nb = adj[u];
    nb.erase(std::remove_if(nb.begin(), nb.end(),
      [&](size_t v){ return (triSup[triIdx(u,v)] < sMin); }), nb.end());
  }
  // symmetrize after pruning
  { std::vector<std::vector<size_t>> sym(N);
    for (size_t u=0; u<N; ++u) for (size_t v: adj[u]) { sym[u].push_back(v); sym[v].push_back(u); }
    adj.swap(sym);
  }

  if (fPruneBridges) {
    // Feed per-edge min triplet χ² so “strong bridges” can be kept.
    PruneUnsupportedBridges(adj, adjW, triSup, sMin + 1, /*strongChi2Cut=*/strongChi2Cut, &edgeChi2Min);
  }

  auto comps = ConnectedComponents(adj);
  std::vector<char> claimed(N,0);


  // Claim threshold (same semantics used elsewhere)
  const double wClaimMin = (fweightCut > 0 ? fweightCut : 0.30);

  auto nC3 = [](size_t m)->size_t { return (m<3)?0u:(m*(m-1)*(m-2))/6u; };
  

  struct Comp {
    std::vector<const Candidate*> tracks; // final track list (after split/assign)
    TVectorD    vBLUE;                    // seed for RobustFit
    Comp(): vBLUE(3) { vBLUE.Zero();  }
  };

  struct TNode {
    TVectorD    v;     // 3x1
    TMatrixDSym C;     // 3x3
    double      chi2;  // triplet chi2/ndf
    double      w;     // quality weight
    std::array<size_t,3> idx; // indices of tracks in 'trks'
    TNode(): v(3), C(3), chi2(0.0), w(0.0), idx{0,0,0} { v.Zero(); C.Zero(); }
  };


  auto trip_maha2 = [this](const TVectorD& va, const TMatrixDSym& Ca,
                         const TVectorD& vb, const TMatrixDSym& Cb)->double {
    TMatrixDSym S(3); S.Zero();
    for (int i=0;i<3;++i) for (int j=0;j<=i;++j) S(i,j)=Ca(i,j)+Cb(i,j);
    TMatrixD Sinv(3,3); Sinv.UnitMatrix();
    { TMatrixDSym tS=S; InvertSPD(tS, Sinv); }
    TVectorD r = va - vb;
    return r * (Sinv * r);
  };

  auto score_to = [this](const Candidate* c, const TVectorD& vB, double chi2_attach_max)->double {
    const ViewEntry& tv = viewOf(c);
    double s=0.0, chi2A=1e99;
    TMatrixDSym Czero(3); Czero.Zero();        // ignore vertex covariance deliberately
    GetMetricPOCAT2V(tv, vB, Czero, s, chi2A);
    if (!std::isfinite(chi2A) || chi2A > chi2_attach_max) return 0.0;
    const double tau = 2.0;                    // smooth but steep
    return std::exp(-chi2A/(2.0*tau));
  };

  // Guardrail: child must be sizeable and internally consistent (triplet completeness)
  FitOpts  tripOpts_okc   = fFitOpts;             
  tripOpts_okc.useWeights = false;
  tripOpts_okc.useTiming  = false;
  const double   chi2Pair_okc   = chi2PairCut;
  const double   chi2Trip_okc   = chi2TripCut;

  auto nC2 = [](size_t m)->size_t { return (m<2)?0u:(m*(m-1))/2u; };

  auto ok_child = [this, nC2, nC3, tripOpts_okc, chi2Pair_okc, chi2Trip_okc]
                  (const std::vector<const Candidate*>& T)->bool {
    const int M = (int)T.size();
    if (M < 3) return false;

    size_t P_ok = 0, P_tot = nC2((size_t)M);
    for (int i=0;i<M;++i)
      for (int j=i+1;j<M;++j) {
        const PairFit& pf = GetPairFit_guarded(T[i], T[j]);
        if (pf.ok && std::isfinite(pf.chi2) && pf.chi2 <= chi2Pair_okc) ++P_ok;
      }

    double p_hat   = (P_tot>0) ? double(P_ok)/double(P_tot) : 0.0;
    double rho_exp = std::clamp(p_hat*p_hat*p_hat, 0.1, 0.95);
    double alpha   = 0.75; // tune 0.6–0.8
    double rho_min = alpha * rho_exp;

    const size_t maxT = nC3((size_t)M);
    if (maxT == 0) return false;

    size_t have = 0;
    for (int i=0;i<M;++i)
      for (int j=i+1;j<M;++j)
        for (int k=j+1;k<M;++k) {
          const TripFit& tf = GetTripFit(T[i], T[j], T[k], tripOpts_okc, chi2Pair_okc);
          if (tf.ok && std::isfinite(tf.chi2) && tf.chi2 <= chi2Trip_okc) ++have;
        }

    const double rho_in = double(have) / double(maxT);
    return (rho_in >= rho_min);   // tune threshold if needed
  };

  //Fill with monomodal components from the pair graph and subgraph on triplets if multimodality has been detected
  std::vector<Comp> components;

  for (const auto& comp : comps) {
    if ((int)comp.size() < fMinTracks) continue;
    if (fRequireSeed){
        bool hasSeed = false;
        for (size_t u : comp) {
          if (isSeed[u]) { hasSeed = true; break; }
        }
        if (!hasSeed) continue;
    }
    std::vector<const Candidate*> trks; trks.reserve(comp.size());
    for (auto u : comp) trks.push_back(tracks[u]);

    // Collect triplets
    std::vector<TNode> trips;

    size_t P_ok = 0, P_tot = nC2(trks.size());
    for (size_t i=0;i<trks.size();++i)
      for (size_t j=i+1;j<trks.size();++j) {
        const PairFit& pf = GetPairFit_guarded(trks[i], trks[j]);
        if (pf.ok && std::isfinite(pf.chi2) && pf.chi2 <= chi2PairCut) ++P_ok;
      }

    double p_hat   = (P_tot>0) ? double(P_ok)/double(P_tot) : 0.0;
    double rho_exp = std::clamp(p_hat*p_hat*p_hat, 0.1, 1.0);
    double alpha   = 0.9; // tune 0.6–0.8


    // multimodality test on component and subgraph on triplets to attempt a split likely multimodal components
    { 
      FitOpts tripopts = fFitOpts;
      tripopts.useWeights = false;
      tripopts.useTiming = false;
      const int M = (int)trks.size();
      trips.reserve((size_t)M*(M-1)*(M-2)/6);
      for (int i=0;i<M;++i) for (int j=i+1;j<M;++j) for (int k=j+1;k<M;++k) {
        const TripFit& tf = GetTripFit(trks[i], trks[j], trks[k], tripopts, chi2PairCut);
        if (!tf.ok || !std::isfinite(tf.chi2) || tf.chi2 > chi2TripCut) continue;
        const double beta_q = 3.0;                   // quality softness
        double wq = std::exp(-tf.chi2/(2.0*beta_q)); // ~1 for small chi2
        TNode n; n.v = tf.v; n.C = tf.Cv; n.chi2 = tf.chi2; n.w = wq;
        n.idx = {(size_t)i,(size_t)j,(size_t)k};
        trips.push_back(std::move(n));
      }
    }

      // quick completeness + dispersion test
    auto nC3_local = nC3; // capture the helper
    bool need_split = false;
    double rho = 1.0, med_d2 = 0.0;
    if (trks.size() >= 10) {
      const size_t maxTrip = nC3_local(trks.size());
      rho = (maxTrip>0) ? double(trips.size())/double(maxTrip) : 0.0;

      if (!trips.empty()) {
        // weighted BLUE of triplets for dispersion check
        std::vector<TVectorD> V; V.reserve(trips.size());
        std::vector<TMatrixDSym> C; C.reserve(trips.size());
        for (auto& t : trips) {
          V.push_back(t.v);
          TMatrixDSym Cs = t.C;                      // scale by 1/w
          const double ww = std::max(1e-6, t.w);
          for (int a=0;a<3;++a) for (int b=0;b<=a;++b) Cs(a,b) /= ww;
          C.push_back(std::move(Cs));
        }
        TVectorD vB(3); vB.Zero();
        TMatrixDSym CB(3); CB.Zero();
        bool blue_ok = BlueCombine(V, C, vB, CB);
        if (!blue_ok) { CB.UnitMatrix(); CB *= 1e2; }
        TMatrixD CBinv(3,3); CBinv.UnitMatrix();
        { TMatrixDSym tCB = CB; InvertSPD(tCB, CBinv); }
        std::vector<double> d2; d2.reserve(trips.size());
        for (auto& t: trips) {
          TVectorD r = t.v - vB;
          d2.push_back(r * (CBinv * r));
        }
        std::nth_element(d2.begin(), d2.begin()+d2.size()/2, d2.end());
        med_d2 = d2[d2.size()/2];
      }

      const double rho_min = alpha * rho_exp;
      const size_t T = trips.size();
      const double med_d2_max = 9.0*(2.0*trks.size() - 3); // (dof counting, self-tuning, ~3σ per dof)
      
      need_split = (rho < rho_min || rho_exp < 0.75) && (med_d2 > med_d2_max && T >=10u);
      // need_split = (rho < rho_min);

      if (fVerbose && need_split) {
        std::cerr << "[TripSplit] comp|T|="<<trks.size()
                  << " trips="<<trips.size()
                  << " max trips="<<nC3_local(trks.size())
                  << " rho_exp="<<std::fixed<<std::setprecision(2)<<rho_exp
                  << " rho="<<std::fixed<<std::setprecision(2)<<rho
                  << " med_d2="<<std::setprecision(2)<<med_d2
                  << " -> "<<(need_split?"try split":"mono")<<"\n";
      }
    }

    if (!need_split || trips.size() < 2u) {
      // Monomodal: compute a BLUE seed from triplets if possible; else from pairs
      Comp cp;
      cp.tracks = trks;
      TVectorD v_seed(3); v_seed.Zero();
      TMatrixDSym C_seed(3); C_seed.Zero();
      bool seeded = SeedFromTriplets(trks, fFitOpts, v_seed, C_seed, chi2TripCut, /*maxTriplets*/trips.size());
      cp.vBLUE = v_seed; // even if Zero; RobustFit can self-seed later
      components.push_back(std::move(cp));
    } else {
        // ===== Conservative split via triplet co-occurrence support ONLY =====
        const size_t Mloc = trks.size();
        const size_t Tn   = trips.size();

        // If for some reason we don't have enough triplets, just fallback to single child
        if (Mloc < 3u || Tn < 3u) {
          Comp cp;
          cp.tracks = trks;
          TVectorD v_seed(3); v_seed.Zero(); TMatrixDSym C_seed(3); C_seed.Zero();
          (void)SeedFromTriplets(trks, fFitOpts, v_seed, C_seed, chi2TripCut, /*maxTriplets*/(int)Tn);
          cp.vBLUE = v_seed;
          components.push_back(std::move(cp));
          continue; // next connected component of the pair graph
        }

        auto idx2 = [Mloc](size_t a, size_t b){ return (a<b) ? a*Mloc + b : b*Mloc + a; };

        //  Build pair support from triplets: number of accepted triplets that contain both i and j
        std::vector<int> pairSup(Mloc*Mloc, 0);
        for (const auto& t : trips) {
          const size_t a=t.idx[0], b=t.idx[1], c=t.idx[2];
          ++pairSup[idx2(a,b)];
          ++pairSup[idx2(a,c)];
          ++pairSup[idx2(b,c)];
        }

        //  Bump support threshold conservatively to avoid over-splitting
        const int sMin_bumped = std::max(minSupport + 2, 2);  
        //  Build adjacency in this local comp by support-only
        std::vector<std::vector<size_t>> adjSup(Mloc);
        for (size_t i=0;i<Mloc;++i) for (size_t j=i+1;j<Mloc;++j) {
          if (pairSup[idx2(i,j)] >= sMin_bumped) {
            adjSup[i].push_back(j);
            adjSup[j].push_back(i);
          }
        }

        //  Connected components under bumped support
        std::vector<char> vis(Mloc,0);
        std::vector<std::vector<size_t>> ccFast;
        std::vector<size_t> st; st.reserve(Mloc);
        for (size_t i=0;i<Mloc;++i){
          if (vis[i]) continue;
          st.clear(); st.push_back(i); vis[i]=1;
          std::vector<size_t> cc; cc.reserve(8);
          while(!st.empty()){
            size_t u = st.back(); st.pop_back();
            cc.push_back(u);
            for (auto v: adjSup[u]) if (!vis[v]){ vis[v]=1; st.push_back(v); }
          }
          ccFast.push_back(std::move(cc));
        }

        // Helper: compute triplet completeness rho for a set of tracks
        auto rho_trip = [&](const std::vector<const Candidate*>& T)->double{
          const int m = (int)T.size();
          if (m < 3) return 0.0;
          size_t ok=0, tot = (size_t)m*(m-1)*(m-2)/6u;
          for (int i=0;i<m;++i)
            for (int j=i+1;j<m;++j)
              for (int k=j+1;k<m;++k){
                const TripFit& tf = GetTripFit(T[i],T[j],T[k], fFitOpts, chi2PairCut);
                if (tf.ok && std::isfinite(tf.chi2) && tf.chi2 <= chi2TripCut) ++ok;
              }
          return (tot>0)? double(ok)/double(tot) : 0.0;
        };

        //  Build candidate children from support CCs and keep only sizeable/consistent ones
        std::vector<Comp> children_fast;
        for (auto& cc : ccFast) {
          if (cc.size() < 2u) continue;
          std::vector<const Candidate*> Ts; Ts.reserve(cc.size());
          for (auto i : cc) Ts.push_back(trks[i]);
          if (ok_child(Ts)) {
            TVectorD v_seed(3); v_seed.Zero(); TMatrixDSym C_seed(3); C_seed.Zero();
            (void)SeedFromTriplets(Ts, fFitOpts, v_seed, C_seed, chi2TripCut, /*maxTriplets*/(int)Tn);
            Comp cp; cp.tracks = std::move(Ts); cp.vBLUE = v_seed;
            children_fast.push_back(std::move(cp));
          }
        }

        // Oversplit guard: if union is very consistent, fall back to single child
        if (!children_fast.empty()) {
          // sort children by size (largest first)
          std::sort(children_fast.begin(), children_fast.end(),
                    [](const Comp& A, const Comp& B){ return A.tracks.size() > B.tracks.size(); });

          // compute union completeness
          std::vector<const Candidate*> unionT;
          {
            std::unordered_set<const Candidate*> seen; seen.reserve(trks.size());
            for (const auto& ch : children_fast)
              for (auto* c : ch.tracks) if (seen.insert(c).second) unionT.push_back(c);
          }
          const double rho_union = rho_trip(unionT);
          const double rho_union_min = 0.85; // conservative; raise to 0.9 if needed

          if (rho_union >= rho_union_min) {
            // prefer single vertex: keep only the largest child as the representative
            if (fVerbose) {
              std::cerr << "[TripSplit][support] union rho=" << std::fixed << std::setprecision(2)
                        << rho_union << " → collapse to 1 child\n";
            }
            children_fast.resize(1u);
          } else {
            // cap at 2 largest children to avoid HF over-splitting
            if (children_fast.size() > 2u) children_fast.resize(2u);
          }

          if (fVerbose) {
            std::cerr << "[TripSplit][support] accepted children: " << children_fast.size()
                      << " (sMin_bumped=" << sMin_bumped << ")\n";
            for (size_t i=0;i<children_fast.size();++i)
              std::cerr << "  child["<<i<<"] |T|="<<children_fast[i].tracks.size() << "\n";
          }

          // append to GLOBAL components and continue
          components.insert(components.end(),
                            std::make_move_iterator(children_fast.begin()),
                            std::make_move_iterator(children_fast.end()));
          continue; // skip any other splitting logic for this component
        }

        // Fallback: nothing decent → keep as single component with triplet BLUE seed
        {
          Comp cp;
          cp.tracks = trks;
          TVectorD v_seed(3); v_seed.Zero(); TMatrixDSym C_seed(3); C_seed.Zero();
          (void)SeedFromTriplets(trks, fFitOpts, v_seed, C_seed, chi2TripCut, /*maxTriplets*/(int)Tn);
          cp.vBLUE = v_seed;
          components.push_back(std::move(cp));
        }
      } // end conservative support-only split
    }//end comp loop

    for (auto comp2 : components){
      std::vector<const Candidate*> trks = comp2.tracks;
      // seedOpt == put here the triplet BLUE of each component
      FitResult fr = RobustFit(trks,  fFitOpts, &comp2.vBLUE);
      if (fr.ndf <= 0) continue;

      const double chi2ndf = fr.chi2 / std::max(1, fr.ndf);
      if (chi2ndf > fMTChi2NDFMax) continue;
      
      Cluster C;

      C.fittedPos    = fr.fittedPos;
      C.fittedCov    = fr.fittedCov;
      C.chi2         = fr.chi2;
      C.ndf          = fr.ndf;
      C.nactive      = fr.nactive;
      C.fittedTime   = fr.fittedTime;
      C.fittedTimeErr= fr.fittedTimeErr;
      C.tracks.reserve(trks.size());
      C.trackWeights.reserve(trks.size());
      C.trackChi2.reserve(trks.size());
      C.trackPhases.reserve(trks.size());

      //Add only tracks that pass the weight cut
      for (size_t i=0; i<trks.size(); ++i) {
        if (fr.trackWeights[i] < wClaimMin) continue; // binary claim

        C.tracks.push_back(trks[i]);
        C.trackWeights.push_back(fr.trackWeights[i]);
        C.trackChi2.push_back(fr.trackChi2[i]);
        C.trackPhases.push_back(fr.trackPhases[i]);
        
        //mark claimed tracks as claimed
        size_t id = (size_t)(std::find(tracks.begin(), tracks.end(), trks[i]) - tracks.begin());
        if (id<claimed.size()) claimed[id]=1;
      }

      if (!C.tracks.empty()) out.push_back(std::move(C));
    }


    // two-track layer: maximum-weight matching on remaining tracks
    if (fMinTracks <= 2 && salvage_pass) {
      std::vector<char> allowed(N, 1);
      for (size_t i=0; i<N; ++i) allowed[i] = !claimed[i];   

      auto pairs = GreedyMaxWeightMatching(adjW, allowed);
      for (auto [u,v] : pairs) {
          if (claimed[u] || claimed[v]) continue;              // redundant now, but harmless
          if (fRequireSeed && !isSeed[u] && !isSeed[v]) continue;

          // TVector3 vfit; TMatrixDSym Cv(3); Cv.Zero(); double chi2 = 0.0;
          PairFit pf = GetPairFit_guarded(tracks[u], tracks[v]);
          if (!pf.ok) continue;
          if (pf.chi2 > chi2PairCut) continue;

          Cluster C;
          C.tracks       = {tracks[u], tracks[v]};
          C.fittedPos    = pf.v;
          C.fittedCov    = pf.Cv;
          C.chi2         = pf.chi2;
          C.ndf          = 1.;
          C.trackWeights = {1.0, 1.0};
          C.trackChi2    = {0.5*pf.chi2, 0.5*pf.chi2};
          C.trackPhases = {pf.l1, pf.l2};

          out.push_back(std::move(C));
          claimed[u] = claimed[v] = 1;                          // mark as used
        }
    }



  if (fVerbose) {
    size_t nMT=0, n2T=0; for (auto& c: out) if (c.nactive>=3) ++nMT; else if (c.nactive==2) ++n2T;
    std::cout << "[Hybrid] multi-track="<<nMT<<", two-track="<<n2T<<"\n";
  }
  return out;
}


// ================== Select Seed tracks ===================

std::vector<const Candidate*> GraphDisplacedVertexFinder::SelectSeeds(
    const std::vector<const Candidate*>& tracks,
    int selectorType,
    double minSeedPT) const
{
    std::vector<const Candidate*> seeds;

    switch (selectorType)
    {
        case 0: // default: select tracks with PT > minSeedPT
            for (const Candidate* trk : tracks) {
                if (trk->PT > minSeedPT) seeds.push_back(trk);
            }
            break;

        case 1: // impact parameter significance (d0 or z0)
            for (const Candidate* trk : tracks) {
                double sig = ipSig_d0z0(trk);
                if (sig > fMinSeedIPSig) {
                    seeds.push_back(trk);
                }
            }
            break;

        case 2: // select tracks with PT > minSeedPT, but seed with leptons only
            {for (const Candidate* trk : tracks) {
                int pdg = std::abs(trk->PID);
                if (trk->PT > minSeedPT && (pdg == 11 || pdg == 13)) seeds.push_back(trk);
            }
            break;
            }


        // Add other selectors here...

        default:
            // fallback default selector (PT)
            for (const Candidate* trk : tracks) {
                if (trk->PT > minSeedPT) seeds.push_back(trk);
            }
            break;
    }

    return seeds;
}


void GraphDisplacedVertexFinder::GetMetricPOCAT2V(const ViewEntry& tv,
                                             const TVectorD& v, 
                                             const TMatrixDSym& Cv,
                                             double& s_out,
                                             double& chi2) const
  {
    TVectorD X(3), t(3), tdot(3), r(3);
    TMatrixDSym Winv(3);
    TMatrixDSym W(3); W.Zero();

    int maxIter = 50;
    double tol = 1e-4;
    double s0 = 0.;
    if (s_out > 0.) s0 = s_out;
    s0 = GetPocaT2V(tv, v, s0, maxIter, tol);
    double s = s0;

    for (int iter = 0; iter < maxIter; ++iter) {
      X = trkX(tv, s);
      t = trkdXds(tv, s);
      tdot = trkd2Xds2(tv, s);
      r = v - X;

      Winv = Cv + trkCx(tv, s);
      (void)InvertSPD(Winv, W);

      double g = - dotD(t, W*r);
      double h = dotD(t, W*t);

      double deltah = std::max(0.0, -dotD(tdot, W*r));
      h += deltah;

      double ds = -g/h;

      s += ds;

      if (std::abs(ds) < tol) break;
    }

    X = trkX(tv, s);
    r = v - X;

    Winv = Cv + trkCx(tv, s);
    (void)InvertSPD(Winv, W);

    s_out = s;
    chi2 = dotD(r, W*r);
  }


// Vertex polish + reconcile: assign leftovers → final polish (+ optional per-track refit)
void GraphDisplacedVertexFinder::PolishOrReconcile(
    std::vector<Cluster>&               clusters,
    const std::vector<const Candidate*>& displacedTracks,
    bool refitTracks) const
{
  if (clusters.empty()) return;


  // GREEDY ASSIGN (leftover tracks)
  const double includeGate   = (fChi2AssociationMax > 0 ? fChi2AssociationMax : 4.0);   // only consider clusters with χ² ≤ includeGate
  const double ambigMargin   = 1.0;    // accept assignment only if (second-best − best) ≥ ambigMargin

  // ------------------ 0) pre-polish each cluster (cheap) ------------------
  for (auto& c : clusters) {
    if (c.tracks.size() < 2) { c.ndf=0; continue; }
    else if (c.tracks.size()==2) RobustFitInPlace(c, fFitOpts);
    if (c.ndf <= 0 || c.chi2/std::max(1, c.ndf) > fMTChi2NDFMax) { continue; }
  }
  double maxchi2 = fMTChi2NDFMax;
  clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
                  [maxchi2](const Cluster& q){ return q.ndf<=0 || q.nactive<2 || q.chi2/std::max(1, q.ndf) > maxchi2; }),
                  clusters.end());
  if (clusters.empty()) return;


  // ------------------  Greedy assignment of leftover tracks ------------------
  // Build set of already-claimed tracks
  std::unordered_set<const Candidate*> claimed;
  claimed.reserve(displacedTracks.size());
  for (const auto& c : clusters) {
    for (size_t i=0; i<c.tracks.size(); ++i) {
      if (c.trackWeights[i] < fweightCut) continue;
      const Candidate* t = c.tracks[i];
      claimed.insert(t);
    }
  }
  // Collect leftovers
  std::vector<const Candidate*> leftovers; leftovers.reserve(displacedTracks.size());
  std::vector<bool> changed(clusters.size(), false);
  for (auto* t : displacedTracks) if (t && !claimed.count(t)) leftovers.push_back(t);
  int nadded = 0;
  for (const Candidate* t : leftovers) {
    double best = 1e12, second = 1e12; int bestK = -1;
    for (size_t k=0; k<clusters.size(); ++k) {
      double c2, s; 
      const ViewEntry& tv = viewOf(t);
      TMatrixDSym nullcov(3); nullcov.Zero();
      GetMetricPOCAT2V(tv, clusters[k].fittedPos, clusters[k].fittedCov, s, c2);

      if (c2 < best) { second = best; best = c2; bestK = (int)k; }
      else if (c2 < second) { second = c2; }
    }
    // confident + gated assignment only
    if (bestK >= 0 && best <= includeGate && (second - best) >= ambigMargin) {
      clusters[bestK].tracks.push_back(t);
      claimed.insert(t);
      ++nadded;
      changed[bestK] = true;
    }
    // else: leave unassigned 
  }
  // Deduplicate tracks per cluster (safety)
  for (auto& c : clusters) {
    std::sort(c.tracks.begin(), c.tracks.end());
    c.tracks.erase(std::unique(c.tracks.begin(), c.tracks.end()), c.tracks.end());
  }
  if (fVerbose) std::cout << "[PolishOrReconcile] Added " << nadded << " leftover tracks to" << std::endl;

  // ------------------ 4) Final polish + optional per-track refits ------------------
  for (size_t i = 0; i < clusters.size(); ++i) {
    if (clusters[i].tracks.size() < 2) { clusters[i].ndf=0; continue; }

    // final robust fit
    if (changed[i]) RobustFitInPlace(clusters[i],  fFitOpts);
    int ndf = clusters[i].ndf;
    if (clusters[i].ndf <= 0 || clusters[i].chi2/std::max(1,ndf) > fMTChi2NDFMax) {  continue; }

    // Optionally refit per-track parameters (normal-plane vertex constraint)
    if (refitTracks) {
      RefitTracksToVertex(clusters[i]);
      }
  }

  // --- exclusivity: one owner per track (shouldn't happen in practice, but let's be safe) ---
  {
    // best cluster for each track
    struct Best { int k; double c2; };
    std::unordered_map<const Candidate*, Best> bestOf;

    // score only tracks a cluster actually claims (weight gate)
    for (int k = 0; k < (int)clusters.size(); ++k) {
      const auto& C = clusters[k];
      for (size_t i = 0; i < C.tracks.size(); ++i) {
        if (i >= C.trackWeights.size() || C.trackWeights[i] < fweightCut) continue;
        const Candidate* t = C.tracks[i];
        double s=0.0, c2=std::numeric_limits<double>::infinity();
        GetMetricPOCAT2V(viewOf(t), C.fittedPos, C.fittedCov, s, c2);
        auto it = bestOf.find(t);
        if (it == bestOf.end() || c2 < it->second.c2) bestOf[t] = {k, c2};
      }
    }

    // helper: slice a vector by indices (keeps alignment)
    auto slice_by_idx = [](auto& vec, const std::vector<size_t>& idxs){
      using T = typename std::decay<decltype(vec)>::type::value_type;
      std::vector<T> out; out.reserve(idxs.size());
      for (size_t i : idxs) if (i < vec.size()) out.push_back(vec[i]);
      vec.swap(out);
    };

    // rebuild each cluster with exclusive members (keep arrays aligned)
    for (int k = 0; k < (int)clusters.size(); ++k) {
      auto& C = clusters[k];

      // collect indices to keep (those for which bestOf says this cluster)
      std::vector<size_t> keep; keep.reserve(C.tracks.size());
      for (size_t i = 0; i < C.tracks.size(); ++i) {
        const Candidate* t = C.tracks[i];
        auto it = bestOf.find(t);
        if (it != bestOf.end() && it->second.k == k) keep.push_back(i);
      }

      // slice all parallel arrays using the same index list
      slice_by_idx(C.tracks,       keep);
      slice_by_idx(C.trackWeights, keep);
      slice_by_idx(C.trackChi2,    keep);
      slice_by_idx(C.trackPhases,  keep);

      // recompute nactive with the current weight gate
      C.nactive = 0;
      for (double w : C.trackWeights) if (w >= fweightCut) ++C.nactive;
    }
  }

  // Drop any invalid/tiny clusters
  const size_t minTracks = static_cast<size_t>(fMinTracks);
    clusters.erase(
    std::remove_if(clusters.begin(), clusters.end(),
        [minTracks](const Cluster& q){
        return q.ndf <=0 || q.nactive < minTracks;
        }),
    clusters.end());

  if (clusters.empty()) return;
}

// ================== Fit Primary Vertex with BeamSpot constraint and sort out prompt/displaced tracks ===================
  double GraphDisplacedVertexFinder::ipSig_d0z0(const Candidate* trk) const {
    // (d0, z0) and their 2x2 covariance with correlation
    TVectorD p  = trkPars(trk);          // (D,phi0,C,z0,ct)
    TMatrixDSym Cp = trkCov(trk);        // 5x5 (PD/regularized by trkCov)

    const double d0 = p(0);
    const double z0 = p(3);

    TMatrixDSym Cdz(2); Cdz.Zero();
    Cdz(0,0) = Cp(0,0);
    Cdz(1,1) = Cp(3,3);
    Cdz(0,1) = Cdz(1,0) = Cp(0,3);

    // invert 2x2 SPD 
    TMatrixD Cinv(2,2);
    {
      TMatrixD I2(2,2); I2.UnitMatrix();
      if (!SolveSPD(Cdz, I2, Cinv)) {
        // tiny regularization as last resort
        TMatrixDSym Cdz2 = Cdz;
        Cdz2(0,0) *= 1.0 + 1e-12;
        Cdz2(1,1) *= 1.0 + 1e-12;
        if (!SolveSPD(Cdz2, I2, Cinv)) return std::numeric_limits<double>::infinity();
      }
    }

    // s² = [d0, z0] C^{-1} [d0, z0]^T  (no abs needed; squared form)
    const double s2 = d0*(Cinv(0,0)*d0 + Cinv(0,1)*z0)
                    + z0*(Cinv(1,0)*d0 + Cinv(1,1)*z0);
    return (s2 >= 0.0 ? std::sqrt(s2) : 0.0);
  };
void GraphDisplacedVertexFinder::FitPrimaryVertex() {
  fDisplacedTracks.clear();
  fPromptTracks.clear();
  fSoftTracks.clear();
  if (fVerbose) std::cout<<"=== FitPrimaryVertex ===\n";


  // ------------ 1) prompt vs displaced preselection ------------
  fItTrack->Reset();
  const Candidate* trk = nullptr;

  while ((trk = static_cast<const Candidate*>(fItTrack->Next()))) {
    if (!trk || trk->Charge == 0) continue;   // only charged

    const double pt = trk->PT;
    const double sIP = ipSig_d0z0(trk);       // correlated (d0,z0) significance

    // Raw component gates (optional but often useful)
    TVectorD p  = trkPars(trk);
    const double d0 = std::fabs(p(0));
    const double z0 = std::fabs(p(3));

    const bool passDisp  = (sIP > fMinsDisp) && (d0 > fMinD0) && (z0 > fMinZ0) && (pt > fMinTrackPT);
    const bool passProm  = ((sIP < fMinsDisp) || (d0 < fMinD0) || (z0 < fMinZ0)) && (pt > fMinTrackPT);

    if (passDisp)      fDisplacedTracks.push_back(trk);
    else if (passProm)      fPromptTracks.push_back(trk);
    else fSoftTracks.push_back(trk);
    // else: unclassified 
    // else               fPromptTracks.push_back(trk);
  }

  // ------------ 2) beam-spot prior (used as PV constraint) ------------
  // cov in mm^2 (transverse & z)
  TMatrixDSym Cbs(3); Cbs.Zero();
  Cbs(0,0) = fBeamSpotSigmaX * fBeamSpotSigmaX;
  Cbs(1,1) = fBeamSpotSigmaY * fBeamSpotSigmaY;
  Cbs(2,2) = fBeamSpotSigmaZ * fBeamSpotSigmaZ;
  TVectorD muBS(3); muBS.Zero();

  // make available to the fitter (it looks up fBeamPos/fBeamCov when UseBeamConstraint=true)
  fBeamPos = muBS;
  fBeamCov = Cbs;

  // ------------ 3) fit PV from prompt tracks ------------
  TVector3 vPV(0,0,0);
  TVectorD muPV(3); muPV.Zero();
  TMatrixDSym CvPV(3); CvPV = Cbs; // default to prior
  bool havePV = false;

  double PVChi2;
  int PVndf;
  if (fPromptTracks.size() >= 2) {
    FitOpts PVopts = fFitOpts;
    PVopts.useBeamConstraint = true;     // activates (fBeamPos,fBeamCov) in fitter

    FitResult fr = RobustFit(fPromptTracks, PVopts, &muBS);

    const double chi2ndfMax = (fPVChi2NDFMax > 0 ? fPVChi2NDFMax : 9.0);
    if (std::isfinite(fr.chi2) && fr.ndf > 0 && (fr.chi2 / std::max(1, fr.ndf)) < chi2ndfMax) {
      muPV  = fr.fittedPos;
      vPV   = TVector3(fr.fittedPos(0), fr.fittedPos(1), fr.fittedPos(2));
      CvPV  = fr.fittedCov;
      havePV = true;
      fPVtime = fr.fittedTime;
      fPVtimeErr = fr.fittedTimeErr;

      PVChi2 = fr.chi2;
      PVndf  = fr.ndf;
    }
  }

  // fallback is beam-spot only (already set: vPV=(0,0,0), CvPV=Cbs)
  fPVPos = vPV;
  fPVCov = CvPV;
  fHasPV = havePV;

    // ------------ 4) optional PV-association veto for displaced list (normal-plane) ------------
    if (fUsePVCut && fHasPV) {
   

    fDisplacedTracks.clear();
    fPromptTracks.clear();
    fSoftTracks.clear();

    fItTrack->Reset();
    const Candidate* trk = nullptr;

    while ((trk = static_cast<const Candidate*>(fItTrack->Next()))) {
        if (!trk) continue;
        double chi2, s;
        GetMetricPOCAT2V(viewOf(trk), muPV, fPVCov, s, chi2);
        // keep as displaced if it does NOT associate to PV
        if (chi2 > fPVCutChi2 && trk->PT > fMinTrackPT) fDisplacedTracks.push_back(trk);
        else if (chi2 < fPVCutChi2) fPromptTracks.push_back(trk);
        else fSoftTracks.push_back(trk);
      }
    }

  // Export Primary Vertex
  Candidate* PV = fFactory->NewCandidate();
  PV->Position.SetXYZT(fPVPos.X(), fPVPos.Y(), fPVPos.Z(), fPVtime);
  PV->PositionError.SetXYZT(std::sqrt(fPVCov(0,0)), std::sqrt(fPVCov(1,1)), std::sqrt(fPVCov(2,2)), fPVtimeErr);
  PV->ErrorXY = fPVCov(0,1);
  PV->ErrorXZ = fPVCov(0,2);
  PV->ErrorYZ = fPVCov(1,2);
  double Lxy = fPVPos.Perp();
  double Lz = fPVPos.Z();
  double Lxyz = fPVPos.Mag();

  PV->Lxy = Lxy;
  PV->Lz = Lz;
  PV->Lxyz = Lxyz;

  PV->vChi2NDF = PVChi2/std::max(1, PVndf);
  PV->vNDF = PVndf;

  double varLxy=0, varLz=0, varLxyz=0;
   {
      const double dx=fPVPos.X(), dy=fPVPos.Y(), dz=fPVPos.Z();

      //Gaussian Error propagation
      if (Lxy>0) {
        const double d2 = std::max(1e-12, dx*dx+dy*dy);
        varLxy = (dx*dx*fPVCov(0,0) + 2*dx*dy*fPVCov(0,1) + dy*dy*fPVCov(1,1)) / d2;
      }
      varLz = fPVCov(2,2);

      if (Lxyz>0) {
        const double d2 = std::max(1e-12, dx*dx+dy*dy+dz*dz);
        varLxyz = ( dx*dx*fPVCov(0,0) + dy*dy*fPVCov(1,1) + dz*dz*fPVCov(2,2)
                  + 2*dx*dy*fPVCov(0,1) + 2*dx*dz*fPVCov(0,2) + 2*dy*dz*fPVCov(1,2) ) / d2;
      }
    }
    PV->ErrorLxy  = (varLxy>0)  ? std::sqrt(varLxy)  : 0.0;
    PV->ErrorLz   = (varLz>0)   ? std::sqrt(varLz)   : 0.0;
    PV->ErrorLxyz = (varLxyz>0) ? std::sqrt(varLxyz) : 0.0;

  

  double charge = 0;
  TLorentzVector PVmom(0,0,0,0); 
  int NTracks = 0;
  int NMuons = 0;
  int NElectrons = 0;
  int NChargedHadrons = 0;

  for (size_t i = 0; i < fPromptTracks.size(); i++) {
    const Candidate* trk = fPromptTracks[i];
    PVmom += trk->Momentum;
    charge += trk->Charge;
    NTracks++;
    int apid = fabs(trk->PID);
    if (apid == 13) NMuons++;
    else if (apid == 11) NElectrons++;
    else if (trk->Charge != 0) NChargedHadrons++;
    PV->AssociatedTracks.Add(const_cast<Candidate*>(trk));
  }

  PV->Momentum = PVmom;
  PV->Charge = charge;
  PV->NTracks = NTracks;
  PV->NMuons = NMuons;
  PV->NElectrons = NElectrons;
  PV->NChargedHadrons = NChargedHadrons;


  PVArray->Add(PV);
    
  if (fVerbose) {
    std::cout << "[FitPrimaryVertex] prompt=" << fPromptTracks.size()
              << " displaced(cut)=" << fDisplacedTracks.size()
              << " soft=" << fSoftTracks.size()
              << "  havePV=" << (fHasPV ? "yes" : "no") << "\n";
    std::cout << "Found Primary Vertex at (" << fPVPos.X() << ", " << fPVPos.Y() << ", " << fPVPos.Z() << ")\n";
  }
}

// ================== Generic Helpers ===================

TVectorD GraphDisplacedVertexFinder::trkPars(const Candidate* trk) const {
    TVectorD par(5);

    par(0) = trk->D0;          // d0 (mm)
    par(1) = trk->Phi;         // phi0 (rad)
    par(2) = trk->C;           // curvature (1/mm)
    par(3) = trk->DZ;          // z0 (mm)
    par(4) = trk->CtgTheta;    // cot(theta) (unitless)
    return par;
}

bool GraphDisplacedVertexFinder::checkPosDef(const TMatrixD& mat) const {
    // Trivial check for empty matrices
    if (mat.GetNrows() == 0 || mat.GetNcols() == 0) return false;

    TDecompChol chol(mat);
    
    // Check if Cholesky decomposition succeeds
    bool isPosDef = chol.Decompose();
    
    if (!isPosDef) {
        std::cerr << "Matrix is not positive definite!" << std::endl;
    }
    
    return isPosDef;
}

TMatrixDSym GraphDisplacedVertexFinder::trkCov(const Candidate* trk) const {
    TMatrixDSym Cv = trk->TrackCovariance; //Covariance Matrix is stored in m for some reason, need to convert to mm
    // std::cout << "Covariance matrix size: " << Cv.GetNrows() << "x" << Cv.GetNcols() << std::endl;

    TMatrixDSym Cmm(5); Cmm.Zero();

    // Fill derivative matrix
    TMatrixD A(5, 5); A.Zero();
    A(0, 0) = 1.0e3;    // D-d0 in mm
    A(1, 1) = 1.0;      // phi0-phi0
    A(2, 2) = 1.0e-3;   // C = q/pT: 1e-4
    A(3, 3) = 1.0e3;    // z0-z0 conversion to mm
    A(4, 4) = 1.0;      // lambda - cot(theta)

    TMatrixD At(5, 5);
    At.Transpose(A);
    Cv.Similarity(At);
    Cmm = Cv;

    // std::cout << "Cmm (covariance matrix) size: " << Cmm.GetNrows() << "x" << Cmm.GetNcols() << std::endl;

    if (!checkPosDef(Cmm)) {
        std::cerr << "Warning: Track Covariance matrix is not positive definite, applying regularization" << std::endl;
        Cmm(0, 0) += 9e-4;       // d0: 30 µm regularize to typical resolution
        Cmm(1, 1) += 1e-6;       // phi: 1 mrad
        Cmm(2, 2) += 1e-8;       // C = q/pT: 1e-4
        Cmm(3, 3) += 2.25e-2;    // dz: 150 µm
        Cmm(4, 4) += 1e-6;       // cot(theta): 0.001   
    }

    // std::cout << "Final Cmm after regularization: " << std::endl;
    // Cmm.Print();

    return Cmm;
}

// === helpers (put in anonymous namespace) ===
namespace {
  inline void Symmetrize(TMatrixD& A){
    const int n=A.GetNrows();
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j){
      const double a = 0.5*(A(i,j)+A(j,i));
      A(i,j)=a; A(j,i)=a;
    }
  }
  inline double MeanDiag(const TMatrixD& A){
    const int n=A.GetNrows(); double s=0.;
    for(int i=0;i<n;++i) s += std::abs(A(i,i));
    return (n>0 && s>0.) ? s/n : 1.0;
  }
  // Robust SPD repair: prefer symmetric-eigen; fallback to SVD if needed.
  inline bool EigenFloorSPD(const TMatrixD& A_in, double floorAbs, TMatrixD& A_spd){
    const int n = A_in.GetNrows();
    if (n == 0) { A_spd.ResizeTo(0,0); return true; }

    //  Symmetrize input and map to TMatrixDSym
    TMatrixD S(A_in);
    // enforce symmetry numerically
    const int m = S.GetNrows();
    for (int i=0;i<m;++i) for (int j=i+1;j<m;++j) {
      const double a = 0.5*(S(i,j)+S(j,i));
      S(i,j)=a; S(j,i)=a;
    }
    TMatrixDSym Ssym(n);
    for (int i=0;i<n;++i) for (int j=0;j<=i;++j) Ssym(i,j) = S(i,j);

    //  Symmetric eigen
    {
      TMatrixDSymEigen eig(Ssym); // doesn’t use nonsymmetric Schur
      TVectorD lam = eig.GetEigenValues();
      TMatrixD V   = eig.GetEigenVectors(); // columns are eigenvectors

      // Check finiteness
      bool finite = V.IsValid();
      for (int i=0;i<n && finite;++i){
        if (!std::isfinite(lam(i))) finite=false;
        for (int j=0;j<n && finite;++j) if (!std::isfinite(V(i,j))) finite=false;
      }
      if (finite){
        for (int i=0;i<n;++i) lam(i) = std::max(lam(i), floorAbs);
        TMatrixD D(n,n); D.Zero(); for (int i=0;i<n;++i) D(i,i) = lam(i);
        A_spd = V * D * V.T();
        // Symmetrize result to kill fp noise
        for (int i=0;i<n;++i) for (int j=i+1;j<n;++j){
          const double a = 0.5*(A_spd(i,j)+A_spd(j,i));
          A_spd(i,j)=a; A_spd(j,i)=a;
        }
        return true;
      }
    }

    // Fallback: SVD on symmetrized S
    {
      TDecompSVD svd(S);
      if (svd.Decompose()){
        TVectorD sig = svd.GetSig();
        TMatrixD V   = svd.GetV(); // for symmetric S, U≈V
        for (int i=0;i<n;++i) sig(i) = std::max(sig(i), floorAbs);
        TMatrixD D(n,n); D.Zero(); for (int i=0;i<n;++i) D(i,i) = sig(i);
        A_spd = V * D * V.T();
        for (int i=0;i<n;++i) for (int j=i+1;j<n;++j){
          const double a = 0.5*(A_spd(i,j)+A_spd(j,i));
          A_spd(i,j)=a; A_spd(j,i)=a;
        }
        return true;
      }
    }

    // Last resort: diagonal shift
    A_spd = S;
    for (int i=0;i<n;++i) A_spd(i,i) += floorAbs;
    return true;
  }

  inline double FroNorm(const TMatrixD& M){
    double s=0.; const int r=M.GetNrows(), c=M.GetNcols();
    for(int i=0;i<r;++i) for(int j=0;j<c;++j){ double v=M(i,j); s+=v*v; }
    return std::sqrt(s);
  }
  inline double TwoNorm(const TVectorD& v){
    double s=0.; for (int i=0;i<v.GetNrows();++i){ double x=v(i); s+=x*x; }
    return std::sqrt(s);
  }
}

// Methods to solve/invert SPD systems/matrices with robust fallbacks
// the covariance/information matrices are often badly conditioned (although they are SPD by construction)
// we add eigenfloors (and repair via residual substractions) to make them invertible via cholesky (usually enough)
// if cholesky on the eigen-floored matrices still doesn't work, we first try BK LDL^T factorization
// if BK doesn't work, we try SVD
// if SVD doesn't work, we give up and cry
bool GraphDisplacedVertexFinder::SolveSPD(const TMatrixDSym& H,
                                          const TMatrixD& rhs,
                                          TMatrixD& sol) const
{
  const int n = H.GetNrows();
  if (rhs.GetNrows()!=n) { std::cerr<<"SolveSPD: dim mismatch\n"; return false; }

  // Keep original H as dense for residuals; build working K for factorization
  TMatrixD Horig(H);
  TMatrixD K(H); Symmetrize(K);

  auto refinement = [&](auto& factorSolve, const char* tag)->bool{
    // initial solution already in 'sol'
    const int maxRefine = 10;
    const double atol = 1e-12, rtol = 1e-10;
    for (int it=0; it<maxRefine; ++it){
      // r = rhs - H * sol
      TMatrixD r(rhs);
      TMatrixD Hx(Horig, TMatrixD::kMult, sol);
      r -= Hx;

      const double rnorm = FroNorm(r);
      const double rhsn  = std::max(1e-30, FroNorm(rhs));
      if (rnorm <= atol + rtol*rhsn){
        // if(it==0) std::cerr<<"SolveSPD["<<tag<<"]: residual OK, no refinement\n";
        // else      std::cerr<<"SolveSPD["<<tag<<"]: refinement converged in "<<it<<" step(s)\n";
        return true;
      }

      // Solve K * delta = r with the SAME factorization
      TMatrixD delta = r;
      bool ok = true;
      for (int j=0;j<delta.GetNcols();++j){
        TMatrixDColumn col(delta, j);
        ok = factorSolve(col) && ok;
      }
      if (!ok){ std::cerr<<"SolveSPD["<<tag<<"]: refinement solve failed\n"; return false; }

      // sol += delta
      sol += delta;
    }
    // std::cerr<<"SolveSPD["<<tag<<"]: refinement reached max iters\n";
    return true;
  };

  // 1) Cholesky
  {
    const double scale = MeanDiag(K);
    const double floorAbs = 1e-8 * scale;
    TMatrixD Kspd; Kspd.ResizeTo(n,n); EigenFloorSPD(K, floorAbs, Kspd);
    TDecompChol chol(Kspd);
    if (chol.Decompose()){
      sol = rhs;
      bool ok = true;
      for (int j=0;j<sol.GetNcols();++j){ TMatrixDColumn col(sol,j); ok = chol.Solve(col) && ok; }
      if (ok){
        // std::cerr<<"SolveSPD: Cholesky success\n";
        auto solveCol = [&](TMatrixDColumn& c){ return chol.Solve(c); };
        return refinement(solveCol, "chol");
      }
      std::cerr<<"SolveSPD: chol Solve failed\n";
    } else std::cerr<<"SolveSPD: chol decompose failed\n";
  }


  // 2) Eigen-floor repair -> chol
  {
    const double scale = MeanDiag(K);
    const double floorAbs = 1e-7 * scale; // tune if needed
    TMatrixD Kspd; Kspd.ResizeTo(n,n); EigenFloorSPD(K, floorAbs, Kspd);
    TDecompChol chol(Kspd);
    if (chol.Decompose()){
      sol = rhs;
      bool ok = true;
      for (int j=0;j<sol.GetNcols();++j){ TMatrixDColumn col(sol,j); ok = chol.Solve(col) && ok; }
      if (ok){
        std::cerr<<"SolveSPD: eigen-floor chol success (λmin="<<floorAbs<<")\n";
        auto solveCol = [&](TMatrixDColumn& c){ return chol.Solve(c); };
        return refinement(solveCol, "eigfloor");
      }
      std::cerr<<"SolveSPD: eigen-floor chol solve failed\n";
    } else std::cerr<<"SolveSPD: eigen-floor chol decompose failed\n";
  }

  // 3) BK LDL^T
  {
    TMatrixDSym Ksym(n); for (int i=0;i<n;++i) for (int j=0;j<=i;++j) Ksym(i,j)=K(i,j);
    TDecompBK bk(Ksym);
    if (bk.Decompose()){
      sol = rhs;
      bool ok = true;
      for (int j=0;j<sol.GetNcols();++j){ TMatrixDColumn col(sol,j); ok = bk.Solve(col) && ok; }
      if (ok){
        std::cerr<<"SolveSPD: BK success\n";
        auto solveCol = [&](TMatrixDColumn& c){ return bk.Solve(c); };
        return refinement(solveCol, "bk");
      }
      std::cerr<<"SolveSPD: BK solve failed\n";
    } else std::cerr<<"SolveSPD: BK decompose failed\n";
  }

  // 4) SVD (pseudo-inverse solve)
  {
    TDecompSVD svd(K);
    if (svd.Decompose()){
      sol = rhs;
      bool ok = true;
      for (int j=0;j<sol.GetNcols();++j){ TMatrixDColumn col(sol,j); ok = svd.Solve(col) && ok; }
      if (ok){
        std::cerr<<"SolveSPD: SVD success\n";
        auto solveCol = [&](TMatrixDColumn& c){ return svd.Solve(c); };
        return refinement(solveCol, "svd");
      }
      std::cerr<<"SolveSPD: SVD solve failed\n";
    } else std::cerr<<"SolveSPD: SVD decompose failed\n";
  }

  std::cerr<<"SolveSPD: all methods failed\n";
  return false;
}


bool GraphDisplacedVertexFinder::SolveSPD(const TMatrixDSym& H,
                                          const TVectorD& rhs,
                                          TVectorD& sol) const
{
  const int n = H.GetNrows();
  if (rhs.GetNrows()!=n) { std::cerr<<"SolveSPD(vec): dim mismatch\n"; return false; }

  TMatrixD Horig(H);
  TMatrixD K(H); Symmetrize(K);

  auto refinement = [&](auto& factorSolve, const char* tag)->bool{
    // residual r = rhs - H*sol
    const int maxRefine = 10;
    const double atol=1e-12, rtol=1e-10;
    for (int it=0; it<maxRefine; ++it){
      TVectorD r = rhs - (Horig * sol); // H*sol
      const double rnorm = TwoNorm(r);
      const double rhsn  = std::max(1e-30, TwoNorm(rhs));
      if (rnorm <= atol + rtol*rhsn){
        // if(it==0) std::cerr<<"SolveSPD(vec)["<<tag<<"]: residual OK, no refinement\n";
        // else      std::cerr<<"SolveSPD(vec)["<<tag<<"]: refinement converged in "<<it<<" step(s)\n";
        return true;
      }
      TVectorD delta(r);
      if (!factorSolve(delta)){ std::cerr<<"SolveSPD(vec)["<<tag<<"]: refinement solve failed\n"; return false; }
      sol += delta;
    }
    // std::cerr<<"SolveSPD(vec)["<<tag<<"]: refinement reached max iters\n";
    return true;
  };

  // 1) Cholesky
  {
    // TDecompChol chol(K);
    const double scale = MeanDiag(K);
    const double floorAbs = 1e-8 * scale;
    TMatrixD Kspd; Kspd.ResizeTo(n,n); EigenFloorSPD(K, floorAbs, Kspd);
    TDecompChol chol(Kspd);
    if (chol.Decompose()){
      TVectorD x(rhs);
      if (chol.Solve(x)){
        sol = x; 
        // std::cerr<<"SolveSPD(vec): Cholesky success\n";
        auto solveVec = [&](TVectorD& v){ return chol.Solve(v); };
        return refinement(solveVec, "chol");
      }
      std::cerr<<"SolveSPD(vec): chol Solve failed\n";
    } else std::cerr<<"SolveSPD(vec): chol decompose failed\n";
  }

  // 2) Eigen-floor repair -> chol
  {
    const double scale = MeanDiag(K);
    const double floorAbs = 1e-7 * scale;
    TMatrixD Kspd; Kspd.ResizeTo(n,n); EigenFloorSPD(K, floorAbs, Kspd);
    TDecompChol chol(Kspd);
    if (chol.Decompose()){
      TVectorD x(rhs);
      if (chol.Solve(x)){
        sol = x; std::cerr<<"SolveSPD(vec): eigen-floor chol success (λmin="<<floorAbs<<")\n";
        auto solveVec = [&](TVectorD& v){ return chol.Solve(v); };
        return refinement(solveVec, "eigfloor");
      }
      std::cerr<<"SolveSPD(vec): eigen-floor chol solve failed\n";
    } else std::cerr<<"SolveSPD(vec): eigen-floor chol decompose failed\n";
  }

  // 3) BK LDL^T
  {
    TMatrixDSym Ksym(n); for(int i=0;i<n;++i) for(int j=0;j<=i;++j) Ksym(i,j)=K(i,j);
    TDecompBK bk(Ksym);
    if (bk.Decompose()){
      TVectorD x(rhs);
      if (bk.Solve(x)){
        sol = x; std::cerr<<"SolveSPD(vec): BK success\n";
        auto solveVec = [&](TVectorD& v){ return bk.Solve(v); };
        return refinement(solveVec, "bk");
      }
      std::cerr<<"SolveSPD(vec): BK solve failed\n";
    } else std::cerr<<"SolveSPD(vec): BK decompose failed\n";
  }

  // 4) SVD
  {
    TDecompSVD svd(K);
    if (svd.Decompose()){
      TVectorD x(rhs);
      if (svd.Solve(x)){
        sol = x; std::cerr<<"SolveSPD(vec): SVD success\n";
        auto solveVec = [&](TVectorD& v){ return svd.Solve(v); };
        return refinement(solveVec, "svd");
      }
      std::cerr<<"SolveSPD(vec): SVD solve failed\n";
    } else std::cerr<<"SolveSPD(vec): SVD decompose failed\n";
  }

  std::cerr<<"SolveSPD(vec): all methods failed\n";
  return false;
}

bool GraphDisplacedVertexFinder::InvertSPD(const TMatrixDSym& H, TMatrixD& Hinv) const
{
  const int n = H.GetNrows();
  if (n == 0) { Hinv.ResizeTo(0,0); return true; }

  TMatrixD K(H); Symmetrize(K);

  auto chol_invert_into_dense = [&](TDecompChol& chol)->bool {
    Hinv.ResizeTo(n,n);
    Hinv.UnitMatrix();                   // start with I
    for (int j=0; j<n; ++j) {            // solve H * X = I  => X = H^{-1}
      TMatrixDColumn col(Hinv, j);
      if (!chol.Solve(col)) return false;
    }
    Symmetrize(Hinv);                    // kill tiny asymmetry
    return true;
  };

  // 1) Cholesky inverse
  {
    const double scale = MeanDiag(K);
    const double floorAbs = 1e-8 * scale;
    TMatrixD Kspd; Kspd.ResizeTo(n,n); EigenFloorSPD(K, floorAbs, Kspd);
    TDecompChol chol(Kspd);
    if (chol.Decompose()){
      if (chol_invert_into_dense(chol)) return true;
      std::cerr<<"InvertSPD: chol Solve(I) failed\n";
    } else {
      std::cerr<<"InvertSPD: chol decompose failed\n";
    }
  }

  // 2) Eigenvalue floor → exact inverse of repaired SPD
  {
    const double scale = MeanDiag(K);
    const double floor = 1e-7*scale;
    TMatrixD Kspd; Kspd.ResizeTo(n,n); EigenFloorSPD(K, floor, Kspd);
    TDecompChol chol(Kspd);
    if (chol.Decompose()){
      if (chol_invert_into_dense(chol)){
        std::cerr<<"InvertSPD: eigen-floor success (λmin="<<floor<<")\n";
        return true;
      }
      std::cerr<<"InvertSPD: eigen-floor chol Solve(I) failed\n";
    } else {
      std::cerr<<"InvertSPD: eigen-floor chol decompose failed\n";
    }
  }

  std::cerr<<"InvertSPD: all methods failed\n";
  return false;
}


bool GraphDisplacedVertexFinder::InvertSPD(const TMatrixDSym& H, TMatrixDSym& Hinv) const
{
  const int n = H.GetNrows();
  if (n == 0) { Hinv.ResizeTo(0,0); std::cerr<<"InvertSPD(sym): empty matrix"<<std::endl; return true; }

  TMatrixD K(H); Symmetrize(K);

  // 1) Cholesky inverse
  {
    const double scale = MeanDiag(K);
    const double floorAbs = 1e-8 * scale;
    TMatrixD Kspd; Kspd.ResizeTo(n,n); EigenFloorSPD(K, floorAbs, Kspd);
    TDecompChol chol(Kspd);
    if (chol.Decompose()){
      Hinv.ResizeTo(n,n);
      if (chol.Invert(Hinv)) return true;
      std::cerr<<"InvertSPD(sym): chol Invert failed\n";
    } else {
      std::cerr<<"InvertSPD(sym): chol decompose failed\n";
    }
  }

  // 2) Eigen-floor → chol invert
  {
    const double scale = MeanDiag(K);
    const double floor = 1e-7*scale;
    TMatrixD Kspd; Kspd.ResizeTo(n,n); EigenFloorSPD(K, floor, Kspd);
    TDecompChol chol(Kspd);
    if (chol.Decompose()){
      Hinv.ResizeTo(n,n);
      if (chol.Invert(Hinv)){ std::cerr<<"InvertSPD(sym): eigen-floor success (λmin="<<floor<<")\n"; return true; }
      std::cerr<<"InvertSPD(sym): eigen-floor chol Invert failed\n";
    } else {
      std::cerr<<"InvertSPD(sym): eigen-floor chol decompose failed\n";
    }
  }

  std::cerr<<"InvertSPD(sym): all methods failed\n";
  return false;
}

// Convert 5x5 perigee covariance from mm-units (fit space) back to Delphes' meters storage.
//
// From C_mm = A * C_m * A^T with A = diag(1e3, 1, 1e-3, 1e3, 1).
// So C_m = A^{-1} * C_mm * A^{-1}^T, with Ainv = diag(1e-3, 1, 1e3, 1e-3, 1).
TMatrixDSym GraphDisplacedVertexFinder::CovPerigee_mm_to_m(const TMatrixDSym& Cmm) const{
  TMatrixDSym out(5); out.Zero();
  TMatrixD Ainv(5,5); Ainv.Zero();
  Ainv(0,0) = 1.0e-3;  // d0
  Ainv(1,1) = 1.0;     // phi
  Ainv(2,2) = 1.0e3;   // curvature
  Ainv(3,3) = 1.0e-3;  // z0
  Ainv(4,4) = 1.0;     // cot(theta)
  out = Cmm;           // copy
  out.Similarity(Ainv); // out := Ainv * Cmm * Ainv^T
  return out;
}
