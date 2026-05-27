
/** \class ExRootTask
 *
 *  Class handling output ROOT tree
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve
 *
 */

#include "ExRootAnalysis/ExRootTask.h"
#include "ExRootAnalysis/ExRootConfReader.h"

#include "TClass.h"
#include "TFolder.h"
#include "TList.h"
#include "TROOT.h"
#include "TString.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

static const Int_t kINIT = 0;
static const Int_t kPROCESS = 1;
static const Int_t kFINISH = 2;

using namespace std;

ExRootTask::ExRootTask() :
  TNamed("", "")
{
  fTasks = new TList();
  fTasks->SetOwner(kTRUE);

  fItTasks = fTasks->MakeIterator();
}

//------------------------------------------------------------------------------

ExRootTask::~ExRootTask()
{
  delete fItTasks;
  delete fTasks;
}

//------------------------------------------------------------------------------

void ExRootTask::Init()
{
}

//------------------------------------------------------------------------------

void ExRootTask::Process()
{
}

//------------------------------------------------------------------------------

void ExRootTask::Finish()
{
}

//------------------------------------------------------------------------------

void ExRootTask::ExecuteTask(Int_t option)
{
  ExRootTask *task;

  if(option == kINIT)
  {
    cout << left;
    cout << setw(30) << "** INFO: initializing module";
    cout << setw(25) << GetName() << endl;
    Init();
  }
  else if(option == kPROCESS)
  {
    Process();
  }
  else if(option == kFINISH)
  {
    Finish();
  }

  fItTasks->Reset();
  while((task = static_cast<ExRootTask *>(fItTasks->Next())))
  {
    task->ExecuteTask(option);
  }
}

//------------------------------------------------------------------------------

void ExRootTask::InitTask()
{
  ExecuteTask(kINIT);
}

//------------------------------------------------------------------------------

void ExRootTask::ProcessTask()
{
  ExecuteTask(kPROCESS);
}

//------------------------------------------------------------------------------

void ExRootTask::FinishTask()
{
  ExecuteTask(kFINISH);
}

//------------------------------------------------------------------------------

void ExRootTask::Add(ExRootTask *task)
{
  stringstream message;

  if(!task) return;

  if(!task->IsA()->InheritsFrom(ExRootTask::Class()))
  {
    message << "task '" << task->IsA()->GetName();
    message << "' does not inherit from ExRootTask";
    throw runtime_error(message.str());
  }

  fTasks->Add(task);
}

//------------------------------------------------------------------------------

ExRootTask *ExRootTask::NewTask(TClass *cl, const char *taskName)
{
  stringstream message;

  if(!cl) return 0;

  if(!cl->InheritsFrom(ExRootTask::Class()))
  {
    message << "task '" << cl->GetName();
    message << "' does not inherit from ExRootTask";
    throw runtime_error(message.str());
  }

  ExRootTask *task = static_cast<ExRootTask *>(cl->New());
  task->SetName(taskName);
  task->SetFolder(fFolder);
  task->SetConfReader(fConfReader);

  return task;
}

//------------------------------------------------------------------------------

ExRootTask *ExRootTask::NewTask(const char *className, const char *taskName)
{
  stringstream message;
  TClass *cl = gROOT->GetClass(className);
  if(!cl)
  {
    message << "can't find class '" << className << "'";
    throw runtime_error(message.str());
  }

  return NewTask(cl, taskName);
}

//------------------------------------------------------------------------------

ExRootConfParam ExRootTask::GetParam(const char *name)
{
  if(fConfReader)
  {
    return fConfReader->GetParam(TString(GetName()) + "::" + name);
  }
  else
  {
    return ExRootConfParam(TString(GetName()) + "::" + name, 0, 0);
  }
}

//------------------------------------------------------------------------------

int ExRootTask::GetInt(const char *name, int defaultValue, int index)
{
  if(fConfReader)
  {
    return fConfReader->GetInt(TString(GetName()) + "::" + name, defaultValue, index);
  }
  else
  {
    return defaultValue;
  }
}

//------------------------------------------------------------------------------

long ExRootTask::GetLong(const char *name, long defaultValue, int index)
{
  if(fConfReader)
  {
    return fConfReader->GetLong(TString(GetName()) + "::" + name, defaultValue, index);
  }
  else
  {
    return defaultValue;
  }
}

//------------------------------------------------------------------------------

double ExRootTask::GetDouble(const char *name, double defaultValue, int index)
{
  if(fConfReader)
  {
    return fConfReader->GetDouble(TString(GetName()) + "::" + name, defaultValue, index);
  }
  else
  {
    return defaultValue;
  }
}

//------------------------------------------------------------------------------

bool ExRootTask::GetBool(const char *name, bool defaultValue, int index)
{
  if(fConfReader)
  {
    return fConfReader->GetBool(TString(GetName()) + "::" + name, defaultValue, index);
  }
  else
  {
    return defaultValue;
  }
}

//------------------------------------------------------------------------------

const char *ExRootTask::GetString(const char *name, const char *defaultValue, int index)
{
  if(fConfReader)
  {
    return fConfReader->GetString(TString(GetName()) + "::" + name, defaultValue, index);
  }
  else
  {
    return defaultValue;
  }
}

//------------------------------------------------------------------------------

TFolder *ExRootTask::NewFolder(const char *name)
{
  stringstream message;
  TFolder *folder;
  folder = static_cast<TFolder *>(GetObject(name, TFolder::Class()));
  if(!folder) folder = GetFolder()->AddFolder(name, "");
  if(!folder)
  {
    message << "can't create folder '" << name << "'";
    throw runtime_error(message.str());
  }
  folder = folder->AddFolder(GetName(), GetTitle());
  if(!folder)
  {
    message << "can't create folder '";
    message << name << "/" << GetName() << "'";
    throw runtime_error(message.str());
  }
  return folder;
}

//------------------------------------------------------------------------------

TObject *ExRootTask::GetObject(const char *name, TClass *cl)
{
  stringstream message;
  TObject *object = GetFolder()->FindObject(name);
  if(object && object->IsA() != cl)
  {
    message << "object '" << name;
    message << "' is not of class '" << cl->GetName() << "'";
    throw runtime_error(message.str());
  }
  return object;
}
