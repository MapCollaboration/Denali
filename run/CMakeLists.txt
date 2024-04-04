# build exacutable
add_executable(ComputeChi2s ComputeChi2s.cc)
target_link_libraries(ComputeChi2s Denali)

add_executable(Optimize Optimize.cc)
target_link_libraries(Optimize Denali)

add_executable(LHAPDFGrid LHAPDFGrid.cc)
target_link_libraries(LHAPDFGrid Denali)

add_executable(Predictions Predictions.cc)
target_link_libraries(Predictions Denali)
