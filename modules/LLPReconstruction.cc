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

 /**  \class LLPReconstruction
 * 
 * Use the displaced vertices that have been found in GraphDisplacedVertexFinder to reconstruct LLPs
 * Matches jets that are built from the output of GraphDisplacedVertexFinder to the LLPs via pT-weighted track overlap
 * 
 * Inputs: Displaced vertices from GraphDisplacedVertexFinder
 *         Jets that have been built from the displaced (eflow-)tracks (from GraphDisplacedVertexFinder) and neutral eflows
 * 
 * Outputs: LLP candidates
 *          jets tagged as displaced (clones)
 *          electrons and muons tagges as displaced (also clones for now)
 *
 * \author Jonathan Kriewald 
 */

#include "LLPReconstruction.h"
#include "classes/DelphesFactory.h"
#include "TMath.h"
#include "TVector3.h"
#include <map>
#include <set>
#include <algorithm>
#include <iostream> // for std::cout

LLPReconstruction::LLPReconstruction() : DelphesModule()
{
    fOverlapThreshold = 0.1;  // threshold to match a jet to a DV
    fMinLeptonPT = 2.0;       // threshold to report a lepton as coming from a LLP candidate (and not part of a displaced jet)
    fMomentumThreshold = 0.8; // Reject LLP candidates for which the invariant mass based on the jet/lepton momenta is lower than the invariant mass from track-only momenta
    fcosThetaCut = 0.5;       // Reject LLP candidates for which the reconstructed momentum direction is wildly inconsistent with the pointing
    fMaxDeltaR = 0.3;         // Maximum deltaR between the LLP candidate and a jet (no hard cut just used as tie-breaker)
    fPTAlpha = 1.0;           // Weighting factor for the pT of the LLP candidate: if set to 1.0, fully use pT, if set to 0., don't use pT
}

LLPReconstruction::~LLPReconstruction() {}

void LLPReconstruction::Init()
{
    fInputVertices = ImportArray(GetString("InputVertices", "DisplacedVertices"));
    nextDV = fInputVertices->MakeIterator();
    
    fInputJets = ImportArray(GetString("InputJets", "DisplacedJets"));
    nextJet = fInputJets->MakeIterator();

    fOutputLLPs = ExportArray(GetString("OutputLLPs", "LLPCandidates"));
    fOutputJetClones = ExportArray(GetString("OutputJetClones", "LLPJets"));
    fOutputElectrons = ExportArray(GetString("OutputElectrons", "LLPElectrons"));
    fOutputMuons = ExportArray(GetString("OutputMuons", "LLPMuons"));

    fOverlapThreshold = GetDouble("OverlapThreshold", 0.1);
    fMinLeptonPT = GetDouble("MinLeptonPT", 2.0);
    fMomentumThreshold = GetDouble("MomentumThreshold", 0.8);
    fcosThetaCut = GetDouble("cosThetaCut", 0.5);

    fMaxDeltaR = GetDouble("MaxDeltaR", 0.3);
    fPTAlpha = GetDouble("PTAlpha", 1.0);
    fVerbose = GetBool("Verbose", false);

    
}

void LLPReconstruction::Finish() {
    if (nextDV) delete nextDV;
    if (nextJet) delete nextJet;
}

void LLPReconstruction::Process()
{

    // Collect jets into vector
    std::vector<Candidate*> jets;
    jets.clear();
    Candidate* jet = nullptr;
    nextJet->Reset();
    while ((jet = static_cast<Candidate*>(nextJet->Next()))) {
        jets.push_back(jet);
    }

    // Collect DVs into vector 
    std::vector<Candidate*> dvsVec;
    dvsVec.clear();
    Candidate* dv = nullptr;
    nextDV->Reset();
    while ((dv = static_cast<Candidate*>(nextDV->Next()))) {
        dvsVec.push_back(dv);
    }

    // Map jet -> (best DV, score) pairs first
    struct JetDVMatch {
        Candidate* jet;
        Candidate* dv;
        double score;
    };

    std::vector<JetDVMatch> matches;

    for (Candidate* j : jets) {
        auto jetTracks = GetDisplacedJetTracks(j);
        double bestScore = 0.0;
        double bestoverlap = 0.0;
        Candidate* bestDV = nullptr;
        double bestdR = 999.0;

        for (Candidate* v : dvsVec) {
            TLorentzVector dvpos = v->Position;
            TLorentzVector pvpos = v->InitialPosition;

            TVector3 dvdir(dvpos.X() - pvpos.X(), dvpos.Y() - pvpos.Y(), dvpos.Z() - pvpos.Z());
            auto dvTracks = GetDisplacedTracks(v);
            double overlap = ComputePTWeightedOverlap(jetTracks, dvTracks);
            TLorentzVector jetmom = j->Momentum;
            double dR = jetmom.Vect().DeltaR(dvdir);

            // Overlap score with angular consistency as tie-breaker
            // for the match we only use the pT weighted track overlap
            double overlapscore = overlap + std::exp(-dR*dR/(fMaxDeltaR*fMaxDeltaR));

            if (overlapscore > bestScore) {
                bestScore = overlapscore;
                bestoverlap = overlap;
                bestDV = v;
                bestdR = dR;
            }
        }

        if (bestDV && (bestoverlap > fOverlapThreshold)) {
            matches.push_back({j, bestDV, bestScore});
        }
    }

    // Sort matches by decreasing score so higher-overlap jets are assigned first
    std::sort(matches.begin(), matches.end(),
              [](const JetDVMatch& a, const JetDVMatch& b) { return a.score > b.score; });

    // Keep track of jets already assigned
    std::set<Candidate*> assignedJets;

    // Clear jets/leptons association in DVs
    for (Candidate* v : dvsVec) {
        v->AssociatedJets.Clear();
        v->AssociatedLeptons.Clear();
    }

    // Greedily assign each jet to the best DV, skipping if already assigned
    for (const auto& m : matches) {
        if (assignedJets.count(m.jet)) continue;
        m.dv->AssociatedJets.Add(m.jet);
        assignedJets.insert(m.jet);

        // Clone jet with DV position
        Candidate* jetClone = static_cast<Candidate*>(m.jet->Clone());
        jetClone->Position = m.dv->Position;
        jetClone->PositionError = m.dv->PositionError;

        fOutputJetClones->Add(jetClone);
    }

    // Associate leptons not in jets for each DV
    for (Candidate* v : dvsVec) {
        auto dvTracks = GetDisplacedTracks(v);

        std::set<const Candidate*> tracksInJets;
        for (int i = 0; i < v->AssociatedJets.GetEntriesFast(); ++i) {
            Candidate* assignedJet = static_cast<Candidate*>(v->AssociatedJets.At(i));
            auto jetTracks = GetDisplacedJetTracks(assignedJet);
            for (auto jt : jetTracks) tracksInJets.insert(jt);
        }

        for (auto trk : dvTracks) {
            if (trk->Momentum.Pt() < fMinLeptonPT) continue;
            if (tracksInJets.find(trk) != tracksInJets.end()) continue;

            if (fabs(trk->PID) == 11 || fabs(trk->PID) == 13) {
                Candidate* lepClone = static_cast<Candidate*>(trk->Clone());
                lepClone->Position = v->Position;
                lepClone->PositionError = v->PositionError;
                v->AssociatedLeptons.Add(lepClone);
                if (fabs(trk->PID) == 11) fOutputElectrons->Add(lepClone);
                else fOutputMuons->Add(lepClone);
            }
        }
    }

    // Create LLP candidates from DVs with associated jets and leptons
    nextDV->Reset();
    while ((dv = static_cast<Candidate*>(nextDV->Next()))) {
        TLorentzVector sumP4(0, 0, 0, 0);
        int njets = dv->AssociatedJets.GetEntriesFast();
        int nelectrons = 0;
        int nmuons = 0;

        for (int i = 0; i < dv->AssociatedJets.GetEntriesFast(); ++i) {
            Candidate* aj = static_cast<Candidate*>(dv->AssociatedJets.At(i));
            sumP4 += aj->Momentum;
        }
        for (int i = 0; i < dv->AssociatedLeptons.GetEntriesFast(); ++i) {
            Candidate* lep = static_cast<Candidate*>(dv->AssociatedLeptons.At(i));
            sumP4 += lep->Momentum;
            if (fabs(lep->PID) == 11) ++nelectrons;
            if (fabs(lep->PID) == 13) ++nmuons;
        }
        

        Candidate* llpCand = static_cast<Candidate*>(dv->Clone());
        if (sumP4.Vect().Mag() < 1e-3 || sumP4.M() < fMomentumThreshold * dv->Momentum.M()) {
            continue;
        } else {
            llpCand->Momentum = sumP4;
        }
        llpCand->NJets = njets;
        llpCand->NElectrons = nelectrons;
        llpCand->NMuons = nmuons;

        TLorentzVector dvpos = dv->Position;
        TLorentzVector pvpos = dv->InitialPosition;

        TVector3 displacement(dvpos.X() - pvpos.X(), dvpos.Y() - pvpos.Y(), dvpos.Z() - pvpos.Z());
        TVector3 direction = llpCand->Momentum.Vect();

        double ptrans = (sumP4.Vect().Cross(displacement.Unit())).Mag();
        double Mcorr = std::sqrt(sumP4.M2() + ptrans * ptrans) + ptrans;
        llpCand->MassCorr = Mcorr;

        if (fVerbose) std::cout << "DV Momentum PT = " << dv->Momentum.Pt() << ", sum PT = " << sumP4.Pt() << ", njets = "<< njets << std::endl;
        if (fVerbose) std::cout << "DV Momentum M = " << dv->Momentum.M() << ", sum M = " << sumP4.M() << ", Mcorr = " << Mcorr << std::endl;
        double cosTheta = 0.;
        if (displacement.Mag() > 0 && direction.Mag() > 0)
            cosTheta = displacement.Unit() * direction.Unit();

        if (cosTheta < fcosThetaCut) continue; // Reject LLP candidates if their reconstructed momentum is not aligned with the displacement direction

        llpCand->CosThetaDVMom = cosTheta;
        
        fOutputLLPs->Add(llpCand);
    }
}

std::vector<const Candidate*> LLPReconstruction::GetDisplacedTracks(const Candidate* cand) const
{
    std::vector<const Candidate*> tracks;
    for (int i = 0; i < cand->AssociatedTracks.GetEntriesFast(); ++i) {
        const Candidate* trk = static_cast<const Candidate*>(cand->AssociatedTracks.At(i));
        if (!trk) continue;
        tracks.push_back(trk);
    }
    return tracks;
}

std::vector<const Candidate*> LLPReconstruction::GetDisplacedJetTracks(Candidate* cand) const
{
    std::vector<const Candidate*> tracks;
    TObjArray* arr = cand->GetCandidates();
    if (!arr) return tracks;

    // Collect charged tracks only
    for (int i = 0; i < arr->GetEntriesFast(); ++i) {
        const Candidate* trk = static_cast<const Candidate*>(arr->At(i));
        if (trk && trk->Charge != 0)
            tracks.push_back(trk);
    }

    // Reject jets without charged tracks
    if (tracks.size() < 1) tracks.clear();

    return tracks;
}

double LLPReconstruction::ComputeTrackOverlapFraction(const std::vector<const Candidate*>& jetTracks,
                                                     const std::vector<const Candidate*>& dvTracks) const
{
    if (jetTracks.empty() || dvTracks.empty())
        return 0.0;

    std::set<const Candidate*> dvTrackSet(dvTracks.begin(), dvTracks.end());

    int nShared = 0;
    for (auto trk : jetTracks)
        if (dvTrackSet.count(trk)) ++nShared;

    // std::cout << "nShared = " << nShared << ", jetTracks.size() = " << jetTracks.size() << std::endl;
    return 2.*double(nShared) / (double(jetTracks.size()) + double(dvTracks.size()));
}

double LLPReconstruction::ComputePTWeightedOverlap(
    const std::vector<const Candidate*>& jetTracks,
    const std::vector<const Candidate*>& dvTracks) const
{
    if (jetTracks.empty() || dvTracks.empty()) return 0.0;

    std::set<const Candidate*> dvSet(dvTracks.begin(), dvTracks.end());

    auto w = [this](const Candidate* t) {
        // Guard against non-positive pT
        const double pt = std::max(0.0, (double)t->Momentum.Pt());
        return (fPTAlpha == 1.0) ? pt : std::pow(pt, fPTAlpha);
    };

    double sumJet = 0.0, sumDV = 0.0, sumShared = 0.0;

    for (auto* t : jetTracks) {
        const double wt = w(t);
        sumJet += wt;
        if (dvSet.count(t)) sumShared += wt; // shared is defined by pointer identity 
    }
    for (auto* t : dvTracks) sumDV += w(t);

    const double denom = sumJet + sumDV;
    if (denom <= 0.0) return 0.0;

    return 2.0 * sumShared / denom; // pT-weighted Dice/F1
}
