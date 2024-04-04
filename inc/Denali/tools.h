//
// Authors: Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//          Amedeo Chiefa
//

#pragma once

#include <LHAPDF/LHAPDF.h>
#include <vector>
#include <math.h>

namespace Denali
{
  /**
   * @name Tools
   * Collection of useful tools.
   */

  /**
   * @brief Utility function used by the NN parametrisation
   * to add the standard deviation to the central replica
   * @param x: Bjorken's x
   * @param Q: Virtuality
   * @param UnpPDF: LHAPDF set
   * @param AddMean: whether or not to add the mean value to
   * the standard deviation (e.g. in NNADparameterisation)
   * @param r: Multiplicative factor
   * @return a map<int, double> containig the std for each
   * PDF, evaluated at (x,Q)
   */
  std::map<int, double> Compute_Std(double const& x, double const& Q, std::vector<LHAPDF::PDF*> UnpPDF, bool const& AddMean = false, double const& r = 1);
}