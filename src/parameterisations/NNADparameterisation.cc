//
// Authors: Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include <algorithm>
#include "Denali/NNADparameterisation.h"
#include "Denali/tools.h"

namespace Denali
{
  //_________________________________________________________________________
  NNADparameterisation::NNADparameterisation(YAML::Node const &config, std::shared_ptr<const apfel::Grid> g, std::vector<LHAPDF::PDF*> UnpPDF, double const& mu0, int const& irep, bool const& IncludeStd, double const& r):
    NangaParbat::Parameterisation("NNAD", 2, {}, false),
              _NNarchitecture(config["architecture"].as<std::vector<int>>()),
              _NN(new nnad::FeedForwardNN<double>(_NNarchitecture, config["seed"].as<int>(), nnad::OutputFunction::ACTIVATION)),
              _Nout(_NNarchitecture.back()),
              _Np(_NN->GetParameterNumber()),
              _g(g),
              _NNderivativeSets(_Np + 1, apfel::Set<apfel::Distribution> {apfel::DiagonalBasis{13}, std::map<int, apfel::Distribution>{}}),
              _FlavourMap( (config["flavour map"] ? config["flavour map"].as<int>() : 1) )
  {
    // Get NN parameters
    this->_pars = _NN->GetParameters();

    // Check if the number of output nodes matches
    // the number of parametrised flavours
    switch (_FlavourMap)
      {
        case 0: // s != sbar
          if (_NNarchitecture.back() != 7)
            {
              std::cerr << "NNAD : The output layer must contain 7 nodes." << std::endl;
              exit(-1);
            }
            break;
        case 1: // s = sbar
          if (_NNarchitecture.back() != 6)
            {
              std::cerr << "NNAD : The output layer must contain 6 nodes." << std::endl;
              exit(-1);
            }
            break;
        default:
          std::cerr << "NNAD : Unknown flavour map" << std::endl;
          exit(-1);
      }

    // Allocate unpolarised PDFs on grid
    if (IncludeStd)
      _UnpPDF = apfel::DistributionMap(*_g, [=] (double const& x) -> std::map<int, double>
      {
        std::map<int, double> PDF_Std = Denali::Compute_Std(x, mu0, UnpPDF, true, r);
        return PDF_Std;
      });
    else
      _UnpPDF = apfel::DistributionMap(*_g, [=] (double const& x) -> std::map<int, double>
      {
        std::map<int, double> f = UnpPDF[irep]->xfxQ(x, mu0);
        std::for_each(f.begin(), f.end(), [] (std::pair<const int, double>& p)
        {
          p.second = std::abs(p.second);
        });
        return f;
      });

    // Fill in grid
    EvaluateOnGrid();
  }

  //_________________________________________________________________________
  void NNADparameterisation::EvaluateOnGrid()
  {
    const std::function<std::map<int, double>(double const&)> NormNN = [=] (double const& x) -> std::map<int, double>
    {
      // Vector of indices to zip with nnx
      std::vector<int> ind{-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6};

      // Get NN at x
      const std::vector<double> fnnx = _NN->Evaluate({x});

      // Resize to get 13 entries and set c = cbar = b =
      // bbar = t = tbar = 0.
      std::vector<double> nnx(13, 0.);
      switch (_FlavourMap)
        {
          case 0: // sbar != s
            nnx[3] = fnnx[0]; // sbar
            nnx[4] = fnnx[1]; // ubar
            nnx[5] = fnnx[2]; // dbar
            nnx[6] = fnnx[3]; // g
            nnx[7] = fnnx[4]; // d
            nnx[8] = fnnx[5]; // u
            nnx[9] = fnnx[6]; // s
            break;

          case 1: // sbar = s
            nnx[3] = fnnx[0]; // sbar
            nnx[4] = fnnx[1]; // ubar
            nnx[5] = fnnx[2]; // dbar
            nnx[6] = fnnx[3]; // g
            nnx[7] = fnnx[4]; // d
            nnx[8] = fnnx[5]; // u
            nnx[9] = fnnx[0]; // s
            break;
        }

      // Zip indices and NN into a map
      std::map<int, double> nnxMapPhys;
      std::transform(ind.begin(), ind.end(), nnx.begin(), std::inserter(nnxMapPhys, nnxMapPhys.end()), [] (int const& s, double const& i)
      {
        return std::make_pair(s, i);
      });

      // Enforce positivity in the phsysical basis, i.e.:
      //
      // - |f_i(x, m0)| < \Delta f_i(x, mu0) < |f_i(x, m0)|,
      //
      // assuming that the initial scale is below the charm threshold
      // and no intrinsic heavy flavours. This is done exploiting the
      // fact that the activation function of the output nodes of the
      // NN is a sigmoid that is bound between 0 and 1. Therefore, the
      // positivity bound can be enforced as:
      //
      // \Delta f_i(x, mu0) = ( 2 * NN(x) - 1 ) * |f_i(x, m0)|.
      //
      std::map<int, double> xDeltafPhys{{-6, 0}, {-5, 0}, {-4, 0}, {-3, 0}, {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}};
      for (int i = -3; i <= 3; i++)
        xDeltafPhys.at(i) = ( 2 * nnxMapPhys.at(i) - 1 ) * _UnpPDF.at((i == 0 ? 21 : i)).Evaluate(x);

      // Rotate back to the QCD evolution basis and return
      return apfel::PhysToQCDEv(xDeltafPhys);
    };

    // Construct set of distributions using the the function
    _NNderivativeSets[0].SetObjects(DistributionMap(*_g, NormNN));
  }

  //_________________________________________________________________________
  void NNADparameterisation::DeriveOnGrid()
  {
    for (int ip = 0; ip < _Np + 1; ip++)
      {
        const std::function<std::map<int, double>(double const&)> NormNN = [=] (double const& x) -> std::map<int, double>
        {
          // Vector of indices to zip with nnx
          std::vector<int> ind{-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6};

          // Get NN and its derivatives at x
          const std::vector<double> dnnx = _NN->Derive({x});

          // Resize to get 13 entries and set c = cbar = b =
          // bbar = t = tbar = 0.
          std::vector<double> nnx(13, 0.);
          switch (_FlavourMap)
           {
            case 0: // s != sbar
              nnx[3] = dnnx[0 + ip * _Nout]; // sbar
              nnx[4] = dnnx[1 + ip * _Nout]; // ubar
              nnx[5] = dnnx[2 + ip * _Nout]; // dbar
              nnx[6] = dnnx[3 + ip * _Nout]; // g
              nnx[7] = dnnx[4 + ip * _Nout]; // d
              nnx[8] = dnnx[5 + ip * _Nout]; // u
              nnx[9] = dnnx[6 + ip * _Nout]; // s
              break;

            case 1: // s = sbar
              nnx[3] = dnnx[0 + ip * _Nout]; // sbar
              nnx[4] = dnnx[1 + ip * _Nout]; // ubar
              nnx[5] = dnnx[2 + ip * _Nout]; // dbar
              nnx[6] = dnnx[3 + ip * _Nout]; // g
              nnx[7] = dnnx[4 + ip * _Nout]; // d
              nnx[8] = dnnx[5 + ip * _Nout]; // u
              nnx[9] = dnnx[0 + ip * _Nout]; // s
              break;
           }

          // Zip indices and NN into a map
          std::map<int, double> nnxMapPhys;
          std::transform(ind.begin(), ind.end(), nnx.begin(), std::inserter(nnxMapPhys, nnxMapPhys.end()), [] (int const& s, double const& i)
          {
            return std::make_pair(s, i);
          });

          // Enforce positivity in the phsysical basis
          std::map<int, double> xDeltafPhys{{-6, 0}, {-5, 0}, {-4, 0}, {-3, 0}, {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}};
          for (int i = -3; i <= 3; i++)
            // Include -1 only if ip = 0, i.e. when the NN is not
            // derived.
            xDeltafPhys[i] = ( 2 * nnxMapPhys.at(i) - (ip == 0 ? 1 : 0) ) * _UnpPDF.at((i == 0 ? 21 : i)).Evaluate(x);

          // Rotate back to the QCD evolution basis and return
          return apfel::PhysToQCDEv(xDeltafPhys);
        };

        // Construct set of distributions using the the function
        _NNderivativeSets[ip].SetObjects(DistributionMap(*_g, NormNN));
      }
  }

  //_________________________________________________________________________
  std::function<apfel::Set<apfel::Distribution>(double const&)> NNADparameterisation::DistributionFunction() const
  {
    return [=] (double const &) -> apfel::Set<apfel::Distribution> { return _NNderivativeSets[0]; };
  }

  //_________________________________________________________________________
  std::function<apfel::Set<apfel::Distribution>(double const&)> NNADparameterisation::DistributionDerivative(int ipar) const
  {
    return [=] (double const &) -> apfel::Set<apfel::Distribution> { return _NNderivativeSets[ipar+1]; };
  }
}
