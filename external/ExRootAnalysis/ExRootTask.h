#ifndef ExRootTask_h
#define ExRootTask_h

/** \class ExRootTask
 *
 *  Class handling output ROOT tree
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve
 *
 */

#include "TNamed.h"

#include "ExRootAnalysis/ExRootConfReader.h"

class TClass;
class TFolder;
class TIterator;
class TList;

class ExRootTask: public TNamed
{
public:
  ExRootTask();
  virtual ~ExRootTask();

  virtual void Init();
  virtual void Process();
  virtual void Finish();

  virtual void InitTask();
  virtual void ProcessTask();
  virtual void FinishTask();

  void Add(ExRootTask *task);

  ExRootTask *NewTask(TClass *cl, const char *name);
  ExRootTask *NewTask(const char *className, const char *taskName);

  int GetInt(const char *name, int defaultValue, int index = -1);
  long GetLong(const char *name, long defaultValue, int index = -1);
  double GetDouble(const char *name, double defaultValue, int index = -1);
  bool GetBool(const char *name, bool defaultValue, int index = -1);
  const char *GetString(const char *name, const char *defaultValue, int index = -1);
  ExRootConfParam GetParam(const char *name);

  void SetFolder(TFolder *folder) { fFolder = folder; }
  void SetConfReader(ExRootConfReader *conf) { fConfReader = conf; }

protected:
  TFolder *GetFolder() const { return fFolder; }
  ExRootConfReader *GetConfReader() const { return fConfReader; }

  TFolder *NewFolder(const char *name);
  TObject *GetObject(const char *name, TClass *cl);

private:
  void ExecuteTask(Int_t option);

  TIterator *fItTasks = nullptr; //!
  TList *fTasks = nullptr; //!
  TFolder *fFolder = nullptr; //!
  ExRootConfReader *fConfReader = nullptr; //!

  ClassDef(ExRootTask, 1)
};

#endif /* ExRootTask */
