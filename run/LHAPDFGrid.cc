//
// Authors: Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "Denali/tools.h"

#include <apfel/apfelxx.h>
#include <LHAPDF/LHAPDF.h>
#include <yaml-cpp/yaml.h>
#include <NNAD/FeedForwardNN.h>

#include <functional>
#include <fstream>
#include <algorithm>
#include <tuple>

typedef std::tuple<std::vector<double>, double, int> PVD;
typedef std::vector<PVD> VPVD;

bool wayToSort(PVD i, PVD j)
{
  return std::get<1>(i) < std::get<1>(j);  //ascending order
}

int main(int argc, char *argv[])
{
  if (argc < 2)
    {
      std::cerr << "Usage: " << argv[0] << " <path to fit folder> [<set name> (default: LHAPDFSet)] [<Nmembers> (default: all)]" << std::endl;
      exit(-1);
    }

  // Path to result folder
  const std::string ResultFolder = argv[1];

  std::string OutName = "LHAPDFSet";
  if (argc >= 3)
    OutName = argv[2];

  // Read Input Card
  YAML::Node config = YAML::LoadFile(ResultFolder + "/config.yaml");

  // Retrive best-fit paramaters
  YAML::Node bestfits = YAML::LoadFile(ResultFolder + "/BestParameters.yaml");

  VPVD AllPars;

  // Load all parameters and pair them with the total chi2 value for sorting
  for (auto const &rep : bestfits)
    AllPars.push_back(std::make_tuple(rep["parameters"].as<std::vector<double>>(), rep["total chi2"].as<double>(), rep["unpolarised pdfset"].as<int>()));

  // Sort sets of parameters according to the chi2
  sort(AllPars.begin(), AllPars.end(), wayToSort);

  int Nmembers = 0;
  if (argc >= 4)
    {
      Nmembers = std::stoi(argv[3]);
      if (Nmembers <= 0)
	{
	  std::cout << "Number of availablereplicas = " << AllPars.size() << std::endl;
	  exit(-1);
	}
      std::cout << "Nmembers requested = " << Nmembers << std::endl;
    }
  else
    Nmembers = AllPars.size();

  // Pick the lowest Nmembers of the chi2-sorted replicas
  int i = 0;
  std::vector<std::vector<double>> BestPars;
  std::vector<int>                 PDFreps;
  for (auto const &rep : AllPars)
    {
      if (i == Nmembers)
        break;

      BestPars.push_back(std::get<0>(rep));
      PDFreps.push_back(std::get<2>(rep));
      i++;
    }
  if (i < Nmembers)
    {
      std::cerr << "Requested more replicas than available." << std::endl;
      exit(-1);
    }

  // Get NN architecture
  const std::vector<int> Architecture = config["NNAD"]["architecture"].as<std::vector<int>>();

  // Initialise neural network
  nnad::FeedForwardNN<double> NN{Architecture, 0, nnad::OutputFunction::ACTIVATION};

  // Get unpolarised PDFs
  const std::vector<LHAPDF::PDF*> UnpPDF = LHAPDF::mkPDFs(config["Predictions"]["Sets"]["unpolarised pdfset"]["name"].as<std::string>());

  // APFEL++ EvolutionSetup object
  apfel::EvolutionSetup es{};

  // Adjust evolution parameters
  es.Virtuality        = apfel::EvolutionSetup::Virtuality::SPACE;
  es.EvolPolarisation  = apfel::EvolutionSetup::EvolPolarisation::POL;
  es.Q0                = config["Predictions"]["mu0"].as<double>();
  es.PerturbativeOrder = config["Predictions"]["perturbative order"].as<int>();
  es.QQCDRef           = config["Predictions"]["alphas"]["Qref"].as<double>();
  es.AlphaQCDRef       = config["Predictions"]["alphas"]["aref"].as<double>();
  es.Thresholds        = config["Predictions"]["thresholds"].as<std::vector<double>>();
  es.Masses            = es.Thresholds;
  es.Qmin              = 1;
  es.Qmax              = 1000;
  es.name              = OutName;
  es.GridParameters    = {{200, 1e-5, 3}, {100, 1e-1, 3}, {100, 5e-1, 3}, {100, 8e-1, 3}};
  es.InSet.clear();

  // Gather the flavour map from the config file
  int FlavourMap = (config["NNAD"]["flavour map"] ? config["NNAD"]["flavour map"].as<int>() : 1);

  // Check if the number of output nodes matches
  // the number of parametrised flavours
  switch (FlavourMap)
    {
      case 0: // s != sbar
        if (Architecture.back() != 7)
          {
            std::cerr << "NNAD : The output layer must contain 7 nodes." << std::endl;
            exit(-1);
          }
          break;
      case 1: // s = sbar
        if (Architecture.back() != 6)
          {
            std::cerr << "NNAD : The output layer must contain 6 nodes." << std::endl;
            exit(-1);
          }
          break;
      default:
        std::cerr << "NNAD : Unknown flavour map" << std::endl;
        exit(-1);
    }

  // NN Parameterisation. First compute the average.
  std::vector<std::function<std::map<int, double>(double const&, double const&)>> sets
  {
    [&] (double const& x, double const&) -> std::map<int, double>
    {
      // Initialise map in the QCD evolution basis
      std::map<int, double> PhysMap{{-6, 0}, {-5, 0}, {-4, 0}, {-3, 0}, {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}};

      // Number of replicas used for the average
      const int nr = BestPars.size();

      // Initialise map in the LHAPDf noomenclature
      std::map<int, double> Std{{3,0}, {2,0}, {1,0}, {21,0}, {-1,0}, {-2,0}, {-3,0}};

      // Compute PDFcentral + Std if requested
      if ((config["Predictions"]["Include Std"] ? config["Predictions"]["Include Std"].as<bool>() : true))
        Std = Denali::Compute_Std(x, config["Predictions"]["mu0"].as<double>(), UnpPDF, false, (config["Predictions"]["Multiplicative factor"] ? config["Predictions"]["Multiplicative factor"].as<double>() : 1.));

      // Now run over all replicas and accumulate
      for (int i = 0; i < (int) BestPars.size(); i++)
        {
          // Get central replica of unpolarised PDFs
          const std::map<int, double> xf = UnpPDF[PDFreps[i]]->xfxQ(x, config["Predictions"]["mu0"].as<double>());

          // Set NN parameters
          NN.SetParameters(BestPars[i]);

          // Call NN at x
          std::vector<double> nnx = NN.Evaluate({x});

          // Construct PDFs
          switch (FlavourMap)
            {
              case 0: // s != sbar
                PhysMap[-3] += ( 2 * nnx[0] - 1 ) * (std::abs(xf.at(-3)) + Std.at(-3)) / nr;
                PhysMap[-2] += ( 2 * nnx[1] - 1 ) * (std::abs(xf.at(-2)) + Std.at(-2)) / nr;
                PhysMap[-1] += ( 2 * nnx[2] - 1 ) * (std::abs(xf.at(-1)) + Std.at(-1)) / nr;
                PhysMap[0]  += ( 2 * nnx[3] - 1 ) * (std::abs(xf.at(21)) + Std.at(21)) / nr;
                PhysMap[1]  += ( 2 * nnx[4] - 1 ) * (std::abs(xf.at(1)) + Std.at(1))  / nr;
                PhysMap[2]  += ( 2 * nnx[5] - 1 ) * (std::abs(xf.at(2)) + Std.at(2))  / nr;
                PhysMap[3]  += ( 2 * nnx[6] - 1 ) * (std::abs(xf.at(3)) + Std.at(3))  / nr;
                break;

              case 1: // s = sbar
                PhysMap[-3] += ( 2 * nnx[0] - 1 ) * (std::abs(xf.at(3)) + Std.at(3)) / nr;
                PhysMap[-2] += ( 2 * nnx[1] - 1 ) * (std::abs(xf.at(-2)) + Std.at(-2)) / nr;
                PhysMap[-1] += ( 2 * nnx[2] - 1 ) * (std::abs(xf.at(-1)) + Std.at(-1)) / nr;
                PhysMap[0]  += ( 2 * nnx[3] - 1 ) * (std::abs(xf.at(21)) + Std.at(21)) / nr;
                PhysMap[1]  += ( 2 * nnx[4] - 1 ) * (std::abs(xf.at(1)) + Std.at(1))  / nr;
                PhysMap[2]  += ( 2 * nnx[5] - 1 ) * (std::abs(xf.at(2)) + Std.at(2))  / nr;
                PhysMap[3]  += ( 2 * nnx[0] - 1 ) * (std::abs(xf.at(3)) + Std.at(3))  / nr;
                break;

              default:
                std::cerr << "NNAD : Unknown flavour map" << std::endl;
                exit(-1);
            }
        }

      // Rotate into the QCD evolution basis and return
      return apfel::PhysToQCDEv(PhysMap);
    }
  };
  // Now run over replicas
  //for (auto p: BestPars)
  for (int i = 0; i < (int) BestPars.size(); i++)
    {
      sets.push_back([&,BestPars,i] (double const& x, double const&) -> std::map<int, double>
      {
        // Initialise map in the QCD evolution basis
        std::map<int, double> PhysMap{{-6, 0}, {-5, 0}, {-4, 0}, {-3, 0}, {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}};

        // Initialise map in the LHAPDf noomenclature
        std::map<int, double> Std{{3,0}, {2,0}, {1,0}, {21,0}, {-1,0}, {-2,0}, {-3,0}};

        // Compute PDFcentral + Std if requested
        if ((config["Predictions"]["Include Std"] ? config["Predictions"]["Include Std"].as<bool>() : true))
          Std = Denali::Compute_Std(x, config["Predictions"]["mu0"].as<double>(), UnpPDF, false, (config["Predictions"]["Multiplicative factor"] ? config["Predictions"]["Multiplicative factor"].as<double>() : 1.));

        // Get central replica of unpolarised PDFs
        const std::map<int, double> xf = UnpPDF[PDFreps[i]]->xfxQ(x, config["Predictions"]["mu0"].as<double>());

        // Set NN parameters
        NN.SetParameters(BestPars[i]);

        // Call NN at x
        std::vector<double> nnx = NN.Evaluate({x});

        // Construct PDFs
        switch (FlavourMap)
          {
            case 0: // s != sbar
              PhysMap[-3] += ( 2 * nnx[0] - 1 ) * (std::abs(xf.at(-3)) + Std.at(-3));
              PhysMap[-2] += ( 2 * nnx[1] - 1 ) * (std::abs(xf.at(-2)) + Std.at(-2));
              PhysMap[-1] += ( 2 * nnx[2] - 1 ) * (std::abs(xf.at(-1)) + Std.at(-1));
              PhysMap[0]  += ( 2 * nnx[3] - 1 ) * (std::abs(xf.at(21)) + Std.at(21));
              PhysMap[1]  += ( 2 * nnx[4] - 1 ) * (std::abs(xf.at(1)) + Std.at(1));
              PhysMap[2]  += ( 2 * nnx[5] - 1 ) * (std::abs(xf.at(2)) + Std.at(2));
              PhysMap[3]  += ( 2 * nnx[6] - 1 ) * (std::abs(xf.at(3)) + Std.at(3));
              break;

            case 1: // s = sbar
              PhysMap[-3] += ( 2 * nnx[0] - 1 ) * (std::abs(xf.at(3)) + Std.at(3));
              PhysMap[-2] += ( 2 * nnx[1] - 1 ) * (std::abs(xf.at(-2)) + Std.at(-2));
              PhysMap[-1] += ( 2 * nnx[2] - 1 ) * (std::abs(xf.at(-1)) + Std.at(-1));
              PhysMap[0]  += ( 2 * nnx[3] - 1 ) * (std::abs(xf.at(21)) + Std.at(21));
              PhysMap[1]  += ( 2 * nnx[4] - 1 ) * (std::abs(xf.at(1)) + Std.at(1));
              PhysMap[2]  += ( 2 * nnx[5] - 1 ) * (std::abs(xf.at(2)) + Std.at(2));
              PhysMap[3]  += ( 2 * nnx[0] - 1 ) * (std::abs(xf.at(3)) + Std.at(3));
              break;

            default:
              std::cerr << "NNAD : Unknown flavour map" << std::endl;
              exit(-1);
          }

        // Rotate into the QCD evolution basis and return
        return apfel::PhysToQCDEv(PhysMap);
      });
    }
  es.InSet = sets;

  // Custom LHAPDF-grid header
  std::string GridHeader = "SetDesc: '" + es.name + " ";
  GridHeader += "proton polarised PDF fit at " + std::string(es.PerturbativeOrder, 'N') + "LO - mem=0 => average over replicas, ";
  GridHeader += "mem=1-" + std::to_string(Nmembers) + " => Monte Carlo replicas - set generated with APFEL++'\n";
  GridHeader += "SetIndex: 0000000\n";
  GridHeader += "Authors: V. Bertone, A. Chiefa, E. R. Nocera\n";
  GridHeader += "Reference: arXiv:xxxx.xxxxx\n";
  GridHeader += "Format: lhagrid1\n";
  GridHeader += "DataVersion: 1\n";
  GridHeader += "Particle: 2212\n";
  GridHeader += "FlavorScheme: variable\n";
  GridHeader += "ErrorType: replicas";

  // Feed it to the initialisation class of APFEL++ and create a grid
  apfel::InitialiseEvolution evpdf{es, true, GridHeader};

  // Move set into the result folder if the set does not exist yet
  std::rename(OutName.c_str(), (ResultFolder + "/" + OutName).c_str());

  return 0;
}
