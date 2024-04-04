//
// Authors: Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#pragma once

#include "Denali/NNADparameterisation.h"

#include <NNAD/FeedForwardNN.h>
#include <yaml-cpp/yaml.h>
#include <NangaParbat/parameterisation.h>
#include <LHAPDF/LHAPDF.h>

namespace Denali
{
  class NNADparameterisation: public NangaParbat::Parameterisation
  {
  public:
    /**
     * @brief The "NNADparameterisation" constructor
     */
    NNADparameterisation(YAML::Node const &config, std::shared_ptr<const apfel::Grid> g, std::vector<LHAPDF::PDF*> UnpPDF, double const& mu0, int const& irep, bool const& IncludeStd = true, double const& r = 1.);

    std::vector<double> GetParameters() const { return _pars; }
    int GetParameterNumber()            const { return _Np; }

    void SetParameters(std::vector<double> const &pars)
    {
      this->_pars = pars;
      _NN->SetParameters(pars);
    };

    void EvaluateOnGrid();
    void DeriveOnGrid();

    std::function<apfel::Set<apfel::Distribution>(double const&)> DistributionFunction() const;
    std::function<apfel::Set<apfel::Distribution>(double const&)> DistributionDerivative(int ipar) const;

  private:
    std::vector<int>                             _NNarchitecture;
    nnad::FeedForwardNN<double>                 *_NN;
    int                                          _Nout;
    int                                          _Np;
    std::shared_ptr<const apfel::Grid>           _g;
    std::vector<apfel::Set<apfel::Distribution>> _NNderivativeSets;
    std::map<int, apfel::Distribution>           _UnpPDF;
    int                                          _FlavourMap;
  };
}
