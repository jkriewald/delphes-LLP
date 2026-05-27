#!/usr/bin/env python3

import ROOT

from argparse import ArgumentParser
from sys import exit


def main():

    parser = ArgumentParser()

    parser.add_argument("--card", help="detector card", required=True)
    parser.add_argument("--format", help="input file format (HepMC2, HepMC3, STDHEP, Pythia8)", required=True)
    parser.add_argument("--input", help="input file name", required=True)
    parser.add_argument("--output", help="output file name", required=True)

    args = parser.parse_args()

    ROOT.gSystem.Load("libDelphes")

    inputFormats = {
        "hepmc2": ROOT.DelphesHepMC2Reader,
        "hepmc3": ROOT.DelphesHepMC3Reader,
        "stdhep": ROOT.DelphesSTDHEPReader,
        "pythia8": ROOT.DelphesPythia8Reader,
    }

    inputFileFormat = args.format.lower()

    if inputFileFormat not in inputFormats:
        parser.print_help()
        exit(1)

    inputFileName = args.input

    reader = inputFormats[inputFileFormat]()

    reader.OpenInputFile(inputFileName)

    cardFileName = args.card

    confReader = ROOT.ExRootConfReader()

    confReader.ReadFile(cardFileName)

    outputFileName = args.output

    outputFile = ROOT.TFile(outputFileName, "CREATE")

    treeWriter = ROOT.ExRootTreeWriter(outputFile, "Delphes")

    branchEvent = treeWriter.NewBranch("Event", ROOT.HepMCEvent.Class())
    branchWeight = treeWriter.NewBranch("Weight", ROOT.Weight.Class())

    modularDelphes = ROOT.Delphes("Delphes")
    modularDelphes.SetConfReader(confReader)
    modularDelphes.SetTreeWriter(treeWriter)

    factory = modularDelphes.GetFactory()

    allParticleOutputArray = modularDelphes.ExportArray("allParticles")
    stableParticleOutputArray = modularDelphes.ExportArray("stableParticles")
    partonOutputArray = modularDelphes.ExportArray("partons")

    readLHEF = inputFileFormat == "pythia8" and reader.ReadLHEF()

    if readLHEF:
        readerLHEF = ROOT.DelphesLHEFReader()
        readerLHEF.OpenInputFile(reader.FileNameLHEF())

        branchEventLHEF = treeWriter.NewBranch("EventLHEF", ROOT.LHEFEvent.Class())
        branchWeightLHEF = treeWriter.NewBranch("WeightLHEF", ROOT.LHEFWeight.Class())

        allParticleOutputArrayLHEF = modularDelphes.ExportArray("allParticlesLHEF")
        stableParticleOutputArrayLHEF = modularDelphes.ExportArray("stableParticlesLHEF")
        partonOutputArrayLHEF = modularDelphes.ExportArray("partonsLHEF")

    modularDelphes.InitTask()

    progressBar = ROOT.ExRootProgressBar(-1)

    eventCounter = 0
    treeWriter.Clear()
    modularDelphes.Clear()
    reader.Clear()
    if readLHEF:
        readerLHEF.Clear()

    while reader.ReadEvent(factory, allParticleOutputArray, stableParticleOutputArray, partonOutputArray):
        eventCounter += 1

        if readLHEF:
            readerLHEF.ReadEvent(factory, allParticleOutputArrayLHEF, stableParticleOutputArrayLHEF, partonOutputArrayLHEF)

        if reader.EventReady():
            modularDelphes.ProcessTask()

            reader.AnalyzeEvent(branchEvent, eventCounter)
            reader.AnalyzeWeight(branchWeight)

            if readLHEF:
                readerLHEF.AnalyzeEvent(branchEventLHEF, eventCounter)
                readerLHEF.AnalyzeWeight(branchWeightLHEF)

            treeWriter.Fill()

        treeWriter.Clear()
        modularDelphes.Clear()
        reader.Clear()
        if readLHEF:
            readerLHEF.Clear()

        progressBar.Update(0, eventCounter)

    progressBar.Update(0, eventCounter, ROOT.kTRUE)
    progressBar.Finish()

    reader.CloseInputFile()

    if readLHEF:
        readerLHEF.CloseInputFile()

    modularDelphes.FinishTask()
    treeWriter.Write()

    del treeWriter


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("** ERROR:", e)
