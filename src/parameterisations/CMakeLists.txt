set(parameterisations_source
  NNADparameterisation.cc
  LHAPDFparameterisation.cc
  tools.cc
  )

add_library(parameterisations OBJECT ${parameterisations_source})
