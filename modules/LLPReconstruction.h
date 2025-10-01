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

#ifndef LLPRECONSTRUCTION_H
#define LLPRECONSTRUCTION_H

#include "classes/DelphesModule.h"
#include "classes/DelphesClasses.h"
#include "TIterator.h"

class LLPReconstruction : public DelphesModule
{
public:
  LLPReconstruction();
  ~LLPReconstruction();

  void Init() override;
  void Finish() override;
  void Process() override;

private:
  TObjArray *fInputVertices = nullptr;
  TObjArray *fInputJets = nullptr;

  TObjArray *fOutputLLPs = nullptr;
  TObjArray *fOutputJetClones = nullptr;
  TObjArray *fOutputElectrons = nullptr;
  TObjArray *fOutputMuons = nullptr;

  TIterator *nextDV;
  TIterator *nextJet;

  double fOverlapThreshold;
  double fMinLeptonPT;
  double fMomentumThreshold;
  double fcosThetaCut;
  double fMaxDeltaR;
  bool fVerbose = false;


  double fPTAlpha = 1.0;

  // Private helper methods
  std::vector<const Candidate*> GetDisplacedTracks(const Candidate* cand) const;
  double ComputeTrackOverlapFraction(const std::vector<const Candidate*>& jetTracks,
                                     const std::vector<const Candidate*>& dvTracks) const;

  std::vector<const Candidate*> GetDisplacedJetTracks(Candidate* cand) const ;
  
  double ComputePTWeightedOverlap(
    const std::vector<const Candidate*>& jetTracks,
    const std::vector<const Candidate*>& dvTracks) const;

  ClassDef(LLPReconstruction, 1)
};

#endif
