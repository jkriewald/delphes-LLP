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
 
#ifndef IdenticalObjectRemover_h
#define IdenticalObjectRemover_h

#include "classes/DelphesModule.h"
#include "classes/DelphesClasses.h"
#include "TIterator.h"
#include "TObjArray.h"
#include <vector>
#include <set>

class IdenticalObjectRemover : public DelphesModule
{
public:
    IdenticalObjectRemover();
    ~IdenticalObjectRemover();

    void Init() override;
    void Process() override;
    void Finish() override;

private:
    TObjArray *fInputArray;          // Main input collection
    TIterator *fInputIter;           // Iterator over main input

    TObjArray *fOutputArray;         // Output cleaned collection

    std::vector<TObjArray*> fRemoveArrays;  // Arrays of objects to remove
};

#endif
