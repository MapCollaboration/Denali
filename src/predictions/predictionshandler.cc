//
// Authors: Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "Denali/predictionshandler.h"

#include <LHAPDF/LHAPDF.h>
#include <apfel/SIDISpol.h>
#include <numeric>

namespace Denali
{
  //_________________________________________________________________________
  PredictionsHandler::PredictionsHandler(YAML::Node                                     const& config,
                                         NangaParbat::DataHandler                       const& DH,
                                         std::shared_ptr<const apfel::Grid>             const& g,
                                         std::vector<std::shared_ptr<NangaParbat::Cut>> const& cuts):
    NangaParbat::ConvolutionTable{},
    _mu0(config["mu0"].as<double>()),
    _Thresholds(config["thresholds"].as<std::vector<double>>()),
    _g(g),
    _obs(DH.GetObservable()),
    _bins(DH.GetBinning()),
    _qTfact(DH.GetKinematics().qTfact),
    _iso(DH.GetTargetIsoscalarity()),
    _cmap(apfel::DiagonalBasis{13})
  {
    // Set silent mode for both apfel++ and LHAPDF;
    apfel::SetVerbosityLevel(0);
    LHAPDF::setVerbosity(0);

    // Get unpolarised PDF set
    const LHAPDF::PDF* PDFs = LHAPDF::mkPDF(config["Sets"]["unpolarised pdfset"]["name"].as<std::string>(), config["Sets"]["unpolarised pdfset"]["member"].as<int>());

    // Perturbative order
    const int PerturbativeOrder = config["perturbative order"].as<int>();

    // Alpha_s
    const apfel::TabulateObject<double> TabAlphas{*(new apfel::AlphaQCD
      {config["alphas"]["aref"].as<double>(), config["alphas"]["Qref"].as<double>(), _Thresholds, PerturbativeOrder}), 100, 0.9, 1001, 3};
    const auto Alphas = [&] (double const& mu) -> double{ return TabAlphas.Evaluate(mu); };

    // Initialise QCD space-like polarised evolution operators and
    // tabulated them
    const std::unique_ptr<const apfel::TabulateObject<apfel::Set<apfel::Operator>>> TabGammaij{new const apfel::TabulateObject<apfel::Set<apfel::Operator>>
      {*(BuildDglap(InitializeDglapObjectsQCDpol(*_g, _Thresholds, true), _mu0, PerturbativeOrder, Alphas)), 100, 1, 100, 3}};

    // Zero operator
    const apfel::Operator Zero{*_g, apfel::Null{}};

    // Identity operator
    const apfel::Operator Id{*_g, apfel::Identity{}};

    // Set cuts in the mother class
    this->_cuts = cuts;

    // Compute total cut mask as a product of single masks
    _cutmask.resize(_bins.size(), true);
    for (auto const& c : _cuts)
      _cutmask *= c->GetMask();

    // Center of mass energy
    //const double Vs = DH.GetKinematics().Vs;

    // Overall prefactor
    const double pref = DH.GetPrefactor();

    // Adjust PDFs to account for the isoscalarity
    const std::function<std::map<int, double>(double const&, double const&)> tPDFs = [&] (double const& x, double const& Q) -> std::map<int, double>
    {
      const std::map<int, double> pr = PDFs->xfxQ(x, Q);
      std::map<int, double> tg = pr;
      tg.at(1)  = _iso * pr.at(1)  + ( 1 - _iso ) * pr.at(2);
      tg.at(2)  = _iso * pr.at(2)  + ( 1 - _iso ) * pr.at(1);
      tg.at(-1) = _iso * pr.at(-1) + ( 1 - _iso ) * pr.at(-2);
      tg.at(-2) = _iso * pr.at(-2) + ( 1 - _iso ) * pr.at(-1);
      return tg;
    };

    // Rotate input PDF set into the QCD evolution basis
    const auto RotPDFs = [=] (double const& x, double const& mu) -> std::map<int, double> { return apfel::PhysToQCDEv(tPDFs(x, mu)); };

    // Get electroweak charges
    const std::function<std::vector<double>(double const&)> fBq = [=] (double const& Q) -> std::vector<double> { return apfel::ElectroWeakCharges(Q, false); };

    // Initialise inclusive structure functions
    const auto F2 = BuildStructureFunctions(InitializeF2NCObjectsZM(*_g, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fBq);
    const auto FL = BuildStructureFunctions(InitializeFLNCObjectsZM(*_g, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fBq);

    if (DH.GetProcess() == NangaParbat::DataHandler::Process::pDIS)
      {
        // Get g1 objects
        const auto g1Obj = apfel::Initializeg1NCObjectsZM(*_g, _Thresholds);

        // Loop over the data points
        for (int i = 0; i < (int) _bins.size(); i++)
          {
            // Get virtuality
            const double Q = _bins[i].Qav;

            // Get Bjorken's x
            const double xb = _bins[i].xav;

            // Get the strong coupling
            const double as = Alphas(Q);

            // Get evolution-operator objects
            std::map<int, apfel::Operator> Gammaij = TabGammaij->Evaluate(Q).GetObjects();

            // Get g1 coefficient functions at Q
            const apfel::StructureFunctionObjects g1 = g1Obj(Q, fBq(Q));

            // Get skip vector
            const std::vector<int> skip = g1.skip;

            // Combine perturbative contributions to the coefficient
            // functions
            apfel::Set<apfel::Operator> Ki = g1.C0.at(0);
            if (PerturbativeOrder > 0)
              Ki += ( as / apfel::FourPi ) * g1.C1.at(0);
            if (PerturbativeOrder > 1)
              Ki += pow(as / apfel::FourPi, 2) * g1.C2.at(0);

            // Intialise container for the FK table
            std::map<int, apfel::Operator> Cj;
            for (int j = 0; j < 13; j++)
              Cj.insert({j, Zero});

            // Convolute coefficient functions with the evolution
            // operators
            for (int j = 0; j < 13; j++)
              {
                std::map<int, apfel::Operator> gj;
                for (int i = 0; i < 13; i++)
                  if (apfel::Gkj.count({i, j}) == 0 || (std::find(skip.begin(), skip.end(), i) != skip.end()))
                    gj.insert({i, Zero});
                  else
                    gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                // Convolute distributions, combine them and return.
                Cj.at(j) += (Ki * apfel::Set<apfel::Operator> {g1.ConvBasis.at(0), gj}).Combine();
              }

            // Compute denominator
            double denom = 1;
            if (_obs == NangaParbat::DataHandler::Observable::g1_F1)
              // Construct F1
              denom = ( F2.at(0).Evaluate(xb, Q) - FL.at(0).Evaluate(xb, Q) ) / 2 / xb;

            // Put together FK tables
            if (!_cutmask[i])
              _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
            else
              _FKt.push_back(apfel::Set<apfel::Operator> {(pref / denom / 2 / xb) * apfel::Set<apfel::Operator>{_cmap, Cj}});
          }
      }
    else if (DH.GetProcess() ==  NangaParbat::DataHandler::Process::Sum_rules)
      {
        if (DH.GetObservable() == NangaParbat::DataHandler::Observable::a3 || DH.GetObservable() == NangaParbat::DataHandler::Observable::a8)
          {
            for (int i = 0; i < (int) _bins.size(); i++)
              {
                // Get virtuality
                const double Q = _bins[i].Qav;

                // Get evolution-operator objects
                std::map<int, apfel::Operator> Gammaij = TabGammaij->Evaluate(Q).GetObjects();

                // Fill the coefficients Ki with Id for the selected distribution
                // and zero otherwise
                std::map<int, apfel::Operator> Ki;
                for (int j = 0; j < 13; j++)
                  if (DH.GetObservable() == NangaParbat::DataHandler::Observable::a3 && j == 3)
                    Ki.insert({j, Id});
                  else if (DH.GetObservable() == NangaParbat::DataHandler::Observable::a8 && j == 5)
                    Ki.insert({j, Id});
                  else
                    Ki.insert({j, Zero});

                // Intialise container for the FK table
                std::map<int, apfel::Operator> Cj;
                for (int j = 0; j < 13; j++)
                  Cj.insert({j, Zero});

                // Convolute coefficient functions with the evolution
                // operators
                for (int j = 0; j < 13; j++)
                  {
                    std::map<int, apfel::Operator> gj;
                    for (int i = 0; i < 13; i++)
                      if (apfel::Gkj.count({i, j}) == 0)
                        gj.insert({i, Zero});
                      else
                        gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                    // Convolute distributions, combine them and return.
                    Cj.at(j) += (apfel::Set<apfel::Operator> {_cmap, Ki} * apfel::Set<apfel::Operator> {_cmap, gj}).Combine();
                  }

                // Put together FK tables
                if (!_cutmask[i])
                  _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
                else
                  _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, Cj});
              }
          }
      }
    else if (DH.GetProcess() == NangaParbat::DataHandler::Process::pSIDIS)
      {
        // PDF set
        const LHAPDF::PDF* PDFs = LHAPDF::mkPDF(config["Sets"]["unpolarised pdfset"]["name"].as<std::string>(), config["Sets"]["unpolarised pdfset"]["member"].as<int>());

        // Target isoscalarity
        const double iso = DH.GetTargetIsoscalarity();

        // Adjust PDFs to account for the isoscalarity
        const std::function<std::map<int, double>(double const&, double const&)> tPDFs = [&] (double const& x, double const& Q) -> std::map<int, double>
        {
          const std::map<int, double> pr = PDFs->xfxQ(x, Q);
          std::map<int, double> tg = pr;
          tg.at(1)  = iso * pr.at(1)  + ( 1 - iso ) * pr.at(2);
          tg.at(2)  = iso * pr.at(2)  + ( 1 - iso ) * pr.at(1);
          tg.at(-1) = iso * pr.at(-1) + ( 1 - iso ) * pr.at(-2);
          tg.at(-2) = iso * pr.at(-2) + ( 1 - iso ) * pr.at(-1);
          return tg;
        };

        // FF set
        const std::string hadron = DH.GetHadron() + (DH.GetCharge() == 1 ? "plus" : "minus");
        const LHAPDF::PDF* FFs = LHAPDF::mkPDF(config["Sets"][hadron + " ffset"]["name"].as<std::string>(), config["Sets"][hadron + " ffset"]["member"].as<int>());
        const std::function<std::map<int, double>(double const&, double const&)> tFFs = [&] (double const& x, double const& Q) -> std::map<int, double> { return FFs->xfxQ(x, Q); };

        // EW charges. Set to zero charges of the flavours that are
        // not tagged.
        std::function<std::vector<double>(double const&)> fBq = [=] (double const& Q) -> std::vector<double> { return apfel::ElectroWeakCharges(Q, false); };

        // Initialize unpolarised SIDIS objects.
        const apfel::SidisObjects sounp = InitializeSIDIS(*_g, _Thresholds);

        // Semi-inclusive F1 differential in x, Q, and z as a
        // Set<DoubleObject<Operator, Distribution>> function of Q.
        const std::function<apfel::DoubleObject<apfel::Distribution>(double const&)> SIF1 = [&] (double const& Q) -> apfel::DoubleObject<apfel::Distribution>
        {
          // Strong coupling
          const double as  = Alphas(Q) / apfel::FourPi;
          const double as2 = as * as;

          // Number of active flavours
          const int nf = apfel::NF(Q, _Thresholds);

          // Get charges
          const std::vector<double> Bq = fBq(Q);

          // Produce a map of distributions out of the PDFs and FFs in
          // the physical basis
          const std::map<int, apfel::Distribution> dPDF = apfel::DistributionMap(*_g, tPDFs, Q);
          const std::map<int, apfel::Distribution> dFF  = apfel::DistributionMap(*_g, tFFs,  Q);

          apfel::DoubleObject<apfel::Distribution> distqq;
          apfel::DoubleObject<apfel::Distribution> distgq;
          apfel::DoubleObject<apfel::Distribution> distqg;
          for (auto j = - nf; j <= nf; j++)
            {
              // Skip the gluon
              if (j == 0)
                continue;

              distqq.AddTerm({Bq[abs(j)-1], dPDF.at(j),  dFF.at(j)});
              distgq.AddTerm({Bq[abs(j)-1], dPDF.at(j),  dFF.at(21)});
              distqg.AddTerm({Bq[abs(j)-1], dPDF.at(21), dFF.at(j)});
            }

          // Return F1 as ( F2 - FL ) / 2x
          return ( sounp.C20qq + as * sounp.C21qq + as2 * sounp.C22qq.at(nf) ) * distqq + as * ( sounp.C21gq * distgq + sounp.C21qg * distqg )
                 - as * ( sounp.CL1qq * distqq + sounp.C21gq * distgq + sounp.C21qg * distqg );
        };
        const apfel::TabulateObject<apfel::DoubleObject<apfel::Distribution>> TabSIF1{SIF1, 100, 1, 10, 3, _Thresholds};

        // Rotation matrix from evolution to physical basis
        std::map<int, std::map<int, double>> Tqi;
        for (int q = -6; q <= 6; q++)
          {
            if (q == 0)
              continue;
            Tqi.insert({q, std::map<int, double>{}});
            for (int j = 1; j <= 12; j++)
              Tqi[q].insert({j, apfel::RotQCDEvToPhysFull[q+6][j]});
          }

        // Initialize polarised SIDIS objects.
        const apfel::SidisPolObjects so = InitializeSIDISpol(*_g, _Thresholds);

        // Semi-inclusive g1 differential in x, Q, and z as a
        // Set<DoubleObject<Operator, Distribution>> function of Q.
        const std::function<apfel::Set<apfel::DoubleObject<apfel::Operator, apfel::Distribution>>(double const&)> Ki = [&] (double const& Q) ->
                                                                                                                       apfel::Set<apfel::DoubleObject<apfel::Operator, apfel::Distribution>>
        {
          // Strong coupling
          const double as  = Alphas(Q) / apfel::FourPi;
          const double as2 = as * as;

          // Number of active flavours
          const int nf = apfel::NF(Q, _Thresholds);

          // Get charges
          const std::vector<double> Bq = fBq(Q);

          // Produce a map of distributions out of the FFs in the
          // physical basis
          const std::map<int, apfel::Distribution> DistFFs = apfel::DistributionMap(*_g, tFFs, Q);

          // Initialise a map of double objects to be used to construct
          // a set
          std::map<int, apfel::DoubleObject<apfel::Operator, apfel::Distribution>> KiMap{};

          // Start with the gluon channel and construct the relevant
          // combination of FFs
          apfel::Distribution eqdq = Bq[0] * ( DistFFs.at(1) + DistFFs.at(-1) );
          for (int q = 2; q <= 5; q++)
            eqdq += Bq[q-1] * ( DistFFs.at(q) + DistFFs.at(-q) );

          // NLO gq for g1
          apfel::DoubleObject<apfel::Operator, apfel::Distribution> D0t;
          for (auto const& t : so.G11qg.GetTerms())
            D0t.AddTerm({as * t.coefficient, t.object1, t.object2 * eqdq});
          KiMap.insert({0, D0t});

          // Now run over the quark evolution basis
          for (int i = 1; i < 13; i++)
            {
              apfel::DoubleObject<apfel::Operator, apfel::Distribution> Dit;

              // Useful combination
              apfel::Distribution eqdqTqi = Bq[0] * ( DistFFs.at(1) * Tqi.at(1).at(i) + DistFFs.at(-1) * Tqi.at(-1).at(i) );
              for (int q = 2; q <= 5; q++)
                eqdqTqi += Bq[q-1] * ( DistFFs.at(q) * Tqi.at(q).at(i) + DistFFs.at(-q) * Tqi.at(-q).at(i) );

              // LO
              for (auto const& t: so.G10qq.GetTerms())
                Dit.AddTerm({t.coefficient, t.object1, t.object2 * eqdqTqi});

              // NLO qq
              for (auto const& t: so.G11qq.GetTerms())
                Dit.AddTerm({as * t.coefficient, t.object1, t.object2 * eqdqTqi});

              // NLO qg
              double eqTqi = 0;
              for (int q = 1; q <= 5; q++)
                eqTqi += Bq[q-1] * ( Tqi.at(q).at(i) + Tqi.at(-q).at(i) );

              if (eqTqi != 0)
                {
                  for (auto const& t: so.G11gq.GetTerms())
                    Dit.AddTerm({as * eqTqi * t.coefficient, t.object1, t.object2 * DistFFs.at(21)});
                }

              // NNLO qq for F2
              if (PerturbativeOrder > 1)
                for (auto const& t: so.G12qq.at(nf).GetTerms())
                  Dit.AddTerm({as2 * t.coefficient, t.object1, t.object2 * eqdqTqi});

              KiMap.insert({i, Dit});
            }
          return apfel::Set<apfel::DoubleObject<apfel::Operator, apfel::Distribution>>{KiMap};
        };

        // Tabulate semi-inclusive cross sections in Q
        const apfel::TabulateObject<apfel::Set<apfel::DoubleObject<apfel::Operator, apfel::Distribution>>> TabKi{Ki, 50, 1, 10, 3, _Thresholds};

        // Pointers to the tabulated functions
        std::unique_ptr<apfel::TabulateObject<apfel::Set<apfel::Operator>>> TabSemiIncQIntegrand;

        // Integrate inclusive cross sections and store them
        for (int i = 0; i < (int) _bins.size(); i++)
          {
            // If the point does not obey the cut, set FK table to zero and continue
            if (!_cutmask[i])
              {
                _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
                continue;
              }

            // Tabulate Q integrand for the semi-inclusive cross section
            const std::function<apfel::Set<apfel::Operator>(double const&)> Nj = [&] (double const& Q) -> apfel::Set<apfel::Operator>
            {
              // Get Ki objects at the scale Q
              const std::map<int, apfel::DoubleObject<apfel::Operator, apfel::Distribution>> Ki = TabKi.Evaluate(Q).GetObjects();

              // Compute integral of Ki in x and construct a set
              std::map<int, apfel::Operator> IntKi;
              for (auto const& tms : Ki)
                {
                  apfel::Operator cumulant = Zero;
                  for (auto const& t : tms.second.GetTerms())
                    cumulant += ( t.coefficient * (_bins[i].Intz ? t.object2.Integrate(_bins[i].zmin, _bins[i].zmax) : t.object2.Evaluate(_bins[i].zav)) ) * t.object1;

                  IntKi.insert({tms.first, cumulant});
                };

              // Get evolution operator
              apfel::Set<apfel::Operator> Gammaij = TabGammaij->Evaluate(Q);

              // Intialise container for the FK table
              std::map<int, apfel::Operator> Nj;
              for (int j = 0; j < 13; j++)
                Nj.insert({j, Zero});

              // Compute the product of Ki and Gammaij and adjust the
              // convolution basis
              for (int j = 0; j < 13; j++)
                for (int i = 0; i < 13; i++)
                  if (apfel::Gkj.count({i, j}) != 0)
                    Nj.at(j) += IntKi.at(i) * Gammaij.at(apfel::Gkj.at({i, j}));

              // Return the result
              return apfel::Set<apfel::Operator> {_cmap, Nj};
            };

            // Compute denominator
            double denom;
            if (_bins[i].Intz)
              denom = TabSIF1.Evaluate(_bins[i].Qav).Integrate2(_bins[i].zmin, _bins[i].zmax).Evaluate(_bins[i].xav);
            else
              denom = TabSIF1.Evaluate(_bins[i].Qav).Evaluate(_bins[i].xav, _bins[i].zav);

            // Push back multiplicities
            _FKt.push_back(apfel::Set<apfel::Operator> {( pref / denom ) * Nj(_bins[i].Qav)});
          }
      }
    else
      throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Process.");
  }

  //_________________________________________________________________________
  PredictionsHandler::PredictionsHandler(PredictionsHandler                             const& PH,
                                         std::vector<std::shared_ptr<NangaParbat::Cut>> const& cuts):
    NangaParbat::ConvolutionTable{},
    _mu0(PH._mu0),
    _Thresholds(PH._Thresholds),
    _g(PH._g),
    _obs(PH._obs),
    _bins(PH._bins),
    _qTfact(PH._qTfact),
    _iso(PH._iso),
    _cmap(PH._cmap)
  {
    // Set cuts in the mather class
    _cuts = PH._cuts;

    // Compute total cut mask as a product of single masks
    _cutmask = PH._cutmask;
    for (auto const& c : cuts)
      _cutmask *= c->GetMask();

    // Impose new cuts
    _FKt.resize(_bins.size());
    for (int i = 0; i < (int) _bins.size(); i++)
      _FKt[i] = (_cutmask[i] ? PH._FKt[i] : apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
  }

  //_________________________________________________________________________
  void PredictionsHandler::SetInputPDFs(std::function<apfel::Set<apfel::Distribution>(double const&)> const& InDistFunc)
  {
    // Construct set of distributions taking into account the target
    // isoscalarity.
    const apfel::Set<apfel::Distribution> SF = InDistFunc(_mu0);
    std::map<int, apfel::Distribution> F = SF.GetObjects();
    F.at(3) *= 2 * _iso - 1;
    F.at(4) *= 2 * _iso - 1;
    _D = apfel::Set<apfel::Distribution> {SF.GetMap(), F};
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&)> const&) const
  {
    // Initialise vector of predictions
    std::vector<double> preds(_bins.size());

    // Compute predictions by convoluting the precomputed kernels with
    // the initial-scale PDFs and then perform the integration in x.
    for (int id = 0; id < (int) _bins.size(); id++)
      if (_bins[id].Intx)
        preds[id] = (_cutmask[id] ? ((_FKt[id] * _D).Combine() * [] (double const& x) -> double{ return 1 / x; }).Integrate(_bins[id].xmin, _bins[id].xmax) : 0);
      else
        preds[id] = (_cutmask[id] ? (_FKt[id] * _D).Combine().Evaluate(_bins[id].xav) : 0);

    return preds;
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&)> const&,
                                                         std::function<double(double const&, double const&, double const&)> const&) const
  {
    return PredictionsHandler::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&, int const&)> const&) const
  {
    return PredictionsHandler::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&, int const&)> const&,
                                                         std::function<double(double const&, double const&, double const&, int const&)> const&) const
  {
    return PredictionsHandler::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }
}
