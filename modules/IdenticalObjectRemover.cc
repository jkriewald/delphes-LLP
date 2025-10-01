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

 /**  \class IdenticalObjectRemover
 * 
 * Removes objects from an input collection/array based on pointer identity
 * 
 * Inputs: 1 Array from which objects should be removed
 *         N Arrays of objects that shall be removed from the first array
 * 
 * Outputs: 1 Array without the removed objects
 *
 * \author Jonathan Kriewald 
 */

#include "IdenticalObjectRemover.h"

#include "classes/DelphesFactory.h"

#include <iostream>

IdenticalObjectRemover::IdenticalObjectRemover() :
    fInputArray(nullptr),
    fInputIter(nullptr),
    fOutputArray(nullptr)
{
}

IdenticalObjectRemover::~IdenticalObjectRemover()
{
    if(fInputIter) {
        delete fInputIter;
        fInputIter = nullptr;
    }
}

void IdenticalObjectRemover::Init()
{
    fInputArray = ImportArray(GetString("InputArray", "EFlowMerger/eflow"));
    if(!fInputArray) {
        std::cerr << "IdenticalObjectRemover: Failed to import InputArray" << std::endl;
        return;
    }
    fInputIter = fInputArray->MakeIterator();

    fOutputArray = ExportArray(GetString("OutputArray", "cleanedeflow"));
    if(!fOutputArray) {
        std::cerr << "IdenticalObjectRemover: Failed to export OutputArray" << std::endl;
        return;
    }

    ExRootConfParam param = GetParam("RemoveArray");
    Long_t size = param.GetSize();
    for(Long_t i = 0; i < size; ++i)
    {
        TObjArray *arr = ImportArray(param[i].GetString());
        if(arr)
        {
            fRemoveArrays.push_back(arr);
        }
        else
        {
            std::cerr << "IdenticalObjectRemover: Failed to import RemoveArray " << param[i].GetString() << std::endl;
        }
    }
}

void IdenticalObjectRemover::Process()
{
    fOutputArray->Delete();

    // Build set of pointers to remove
    std::set<const Candidate*> removeSet;
    for(auto arr : fRemoveArrays)
    {
        if(!arr) continue;
        TIterator* it = arr->MakeIterator();;
        Candidate* obj = nullptr;
        while((obj = static_cast<Candidate*>(it->Next())))
        {
            removeSet.insert(obj);
        }
    }

    fInputIter->Reset();
    Candidate* cand = nullptr;
    while((cand = static_cast<Candidate*>(fInputIter->Next())))
    {
        if(removeSet.find(cand) == removeSet.end())
        {
            fOutputArray->Add(cand);
        }
    }
}

void IdenticalObjectRemover::Finish()
{
    if(fInputIter)
    {
        fInputIter->Reset();
    }
}
