include_directories(
  ${CMAKE_SOURCE_DIR}
  ${ROOT_INCLUDE_DIRS}
  ${DelphesExternals_INCLUDE_DIR}
)

file(GLOB sources *.cc)
file(GLOB headers *.h)
list(REMOVE_ITEM headers ${CMAKE_CURRENT_SOURCE_DIR}/ModulesLinkDef.h)
list(REMOVE_ITEM headers ${CMAKE_CURRENT_SOURCE_DIR}/ModulesFastJetLinkDef.h)
list(REMOVE_ITEM headers ${CMAKE_CURRENT_SOURCE_DIR}/ModulesPythia8LinkDef.h)

DELPHES_GENERATE_DICTIONARY(ModulesDict ${headers} LINKDEF ModulesLinkDef.h)
DELPHES_GENERATE_DICTIONARY(ModulesFastJetDict ${headers} LINKDEF ModulesFastJetLinkDef.h)

if(PYTHIA8_FOUND)
  DELPHES_GENERATE_DICTIONARY(ModulesPythia8Dict ${headers} LINKDEF ModulesPythia8LinkDef.h)
  add_library(modules OBJECT ${sources} ModulesDict.cxx ModulesFastJetDict.cxx ModulesPythia8Dict.cxx)
else()
  list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/PileUpMergerPythia8.cc)
  add_library(modules OBJECT ${sources} ModulesDict.cxx ModulesFastJetDict.cxx)
endif()

# install headers
install(FILES ${headers} DESTINATION include/modules)

# install pcms if they are created
if (NOT ${ROOT_VERSION} VERSION_LESS "6.0.0")
  install(FILES
    ${PROJECT_BINARY_DIR}/modules/libModulesDict_rdict.pcm
    ${PROJECT_BINARY_DIR}/modules/libModulesFastJetDict_rdict.pcm
    DESTINATION lib)
  if(PYTHIA8_FOUND)
    install(FILES
      ${PROJECT_BINARY_DIR}/modules/libModulesPythia8Dict_rdict.pcm
      DESTINATION lib)
  endif()
endif()
