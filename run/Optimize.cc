//
// Authors: Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "Denali/predictionshandler.h"
#include "Denali/AnalyticChiSquare.h"
#include "Denali/IterationCallBack.h"
#include "Denali/NNADparameterisation.h"
#include "Denali/LHAPDFparameterisation.h"

// LHAPDF
#include <LHAPDF/LHAPDF.h>

// NangaParbat
#include <NangaParbat/chisquare.h>
#include <NangaParbat/cutfactory.h>
#include <NangaParbat/Trainingcut.h>
#include <NangaParbat/direxists.h>

// CERES
#include <ceres/ceres.h>

// GSL
#include <gsl/gsl_randist.h>

// C++
#include <unistd.h>
#include <getopt.h>
#include <sys/stat.h>

#define DEBUG false

int main(int argc, char *argv[])
{
  const char* const short_opts = "s";
  const option long_opts[] =
  {
    {"separate_replica_results", no_argument, nullptr, 's'},
  };

  bool separate_replica_results = false;

  while (true)
    {
      const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

      if (opt == -1)
        break;

      switch (opt)
        {
        case 's':
          separate_replica_results = true;
          break;
        case '?': // Unrecognized option
        default:  // Unhandled option
          std::cerr << "Usage: " << argv[0] << " [-s|--separate_replica_results] <replica index> <input card> <path to data> <output folder>" << std::endl;
          exit(-1);
        }
    }

  if ((argc - optind) != 4)
    {
      std::cerr << "Usage: " << argv[0] << " [-s|--separate_replica_results] <replica index> <input card> <path to data> <output folder>" << std::endl;
      exit(-1);
    }

  // Input information
  int replica = atoi(argv[optind]);
  const std::string InputCardPath = argv[optind + 1];
  const std::string DataFolder = (std::string) argv[optind + 2] + "/";
  const std::string ResultFolder = argv[optind + 3];

  // Assign to the fit the name of the input card
  const std::string OutputFolder = ResultFolder + "/" + InputCardPath.substr(InputCardPath.find_last_of("/") + 1, InputCardPath.find(".yaml") - InputCardPath.find_last_of("/") - 1);

  // Require that the result folder exists. If not throw an exception.
  if (!NangaParbat::dir_exists(ResultFolder))
    throw std::runtime_error("Result folder does not exist.");

  // Timer
  apfel::Timer t;

  // Read Input Card
  YAML::Node config = YAML::LoadFile(InputCardPath);

  // Set the seeds based on replica wanted
  config["NNAD"]["seed"] = config["NNAD"]["seed"].as<int>() + replica;

  if (replica == 0)
    {
      // Set all replica members to zero (central value)
      for (const auto& Set_type : config["Predictions"]["Sets"])
        {
          YAML::Node temp = Set_type.second;
          temp["member"] = 0;
        }
    }
  else
    {
      if (config["Predictions"]["Replica settings"].as<int>() == -1)
        {
          // Initialise GSL random-number generator
          gsl_rng *pdf_rng = gsl_rng_alloc(gsl_rng_ranlxs2);

          // Loop over the node "Sets" in the config file
          int i=0;
          for (const auto& Set_type : config["Predictions"]["Sets"])
            {
              // Seed the GSL random-number generator
              gsl_rng_set(pdf_rng, config["NNAD"]["seed"].as<int>()+i);

              // Store the YAML::Node
              YAML::Node temp = Set_type.second;

              // Open the set to check it is a Monte Carlo set
              const LHAPDF::PDFSet set(temp["name"].as<std::string>());
              if (set.get_entry_as<std::string>("ErrorType") != "replicas")
                throw std::runtime_error("The chosen set is not a Monte Carlo set.");

              // Extract replica from the GSL random-number.
              temp["member"] = int(gsl_ran_flat(pdf_rng, 1, set.get_entry_as<int>("NumMembers")));
              i++;
            }
        }
      else if (config["Predictions"]["Replica settings"].as<int>() == 0)
        {
          // Set all replica members to zero (central value)
          for (const auto& Set_type : config["Predictions"]["Sets"])
            {
              YAML::Node temp = Set_type.second;
              temp["member"] = 0;
            }
        }
      else if (config["Predictions"]["Replica settings"].as<int>() == 1)
        {
          // Initialise GSL random-number generator
          gsl_rng *pdf_rng = gsl_rng_alloc(gsl_rng_ranlxs2);

          // Loop over the node "Sets" in the config file
          int i=0;
          for (const auto& Set_type : config["Predictions"]["Sets"])
            {
              // Seed the GSL random-number generator
              gsl_rng_set(pdf_rng, config["NNAD"]["seed"].as<int>()+i);

              // Store the YAML::Node
              YAML::Node temp = Set_type.second;

              // Open the set to check it is a Monte Carlo set
              const LHAPDF::PDFSet set(temp["name"].as<std::string>());
              if (set.get_entry_as<std::string>("ErrorType") != "replicas")
                throw std::runtime_error("The chosen set is not a Monte Carlo set.");

              // Check if the member index is negative. If this is
              // the case, then generate a random number
              if (Set_type.second["member"].as<int>() < 0)
                temp["member"] = int(gsl_ran_flat(pdf_rng, 1, set.get_entry_as<int>("NumMembers")));

              i++;
            }
        }
      else
        throw std::runtime_error("Replica settings must be 0, 1 or -1. See the config file for details.");
    }

  // If "Include Std" is True, set the unpolarized PDF member to 0
  // regardless the member previosly selected.
  if ((config["Predictions"]["Include Std"] ? config["Predictions"]["Include Std"].as<bool>() : true))
    config["Predictions"]["Sets"]["unpolarised pdfset"]["member"] = 0;

#if DEBUG == true
  for (const auto& Set_type : config["Predictions"]["Sets"])
    std::cout << Set_type.second["name"].as<std::string>() << " : " << Set_type.second["member"].as<int>() << std::endl;
#endif

  // Set silent mode for APFEL++
  apfel::SetVerbosityLevel(0);

  // APFEL++ x-space grid
  std::vector<apfel::SubGrid> vsg;
  for (auto const &sg : config["Predictions"]["xgrid"])
    vsg.push_back({sg[0].as<int>(), sg[1].as<double>(), sg[2].as<int>()});
  const std::shared_ptr<const apfel::Grid> g(new const apfel::Grid{vsg});

  // Initialise GSL random-number generator
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs2);
  gsl_rng_set(rng, config["Data"]["seed"].as<int>());

  // Vectors of DataHandler-ConvolutionTable pairs to be fed to the chi2
  std::vector<std::pair<NangaParbat::DataHandler*, NangaParbat::ConvolutionTable*>> DSVect;
  std::vector<std::pair<NangaParbat::DataHandler*, NangaParbat::ConvolutionTable*>> DSVectt;
  std::vector<std::pair<NangaParbat::DataHandler*, NangaParbat::ConvolutionTable*>> DSVectv;

  // Vector of file names
  std::vector<std::string> filenames;

  int MinTrainingSize = (config["Data"]["minimum training size"] ? config["Data"]["minimum training size"].as<int>() : 10);

  // Run over the data set
  for (auto const &ds : config["Data"]["sets"])
    {
      std::cout << "Initialising " << ds["name"].as<std::string>() << (config["Data"]["closure_test"] ? " for closure test level " + config["Data"]["closure_test"]["level"].as<std::string>() : "" ) + "...\n";

      // Push back file name
      filenames.push_back(ds["file"].as<std::string>());

      // Load dataset and fluctuate the points
      // EXCEPT if replica is 0, in which case data is not fluctuated and *DH == *DHc
      NangaParbat::DataHandler *DH = new NangaParbat::DataHandler{ds["name"].as<std::string>(), YAML::LoadFile(DataFolder + ds["file"].as<std::string>()), rng, replica};

      // Load dataset without fluctuating the points to be used at the
      // end of the fit to compute the final chi2 over the entire and
      // unfluctuated data set.
      NangaParbat::DataHandler *DHc = new NangaParbat::DataHandler{ds["name"].as<std::string>(), YAML::LoadFile(DataFolder + ds["file"].as<std::string>())};

      // Accumulate kinematic cuts
      std::vector<std::shared_ptr<NangaParbat::Cut> > cuts;
      for (auto const &c : ds["cuts"])
        cuts.push_back(NangaParbat::CutFactory::GetInstance(*DH, c["name"].as<std::string>(), c["min"].as<double>(), c["max"].as<double>(),
                                                            (c["pars"] ? c["pars"].as<std::vector<double>>() : std::vector<double> {})));

      // Compute predictions within kinematic cuts
      Denali::PredictionsHandler PH{config["Predictions"], *DH, g, cuts};

      // Training fraction
      double TrainingFraction = ds["training fraction"].as<double>();

      // Compute training and validation cuts
      NangaParbat::TrainingCut *TrainingCut = new NangaParbat::TrainingCut{*DH, cuts, TrainingFraction, rng, MinTrainingSize};
      NangaParbat::TrainingCut *ValidationCut = new NangaParbat::TrainingCut{*TrainingCut, true, cuts};

      // Push back DataHandler-PredictionHandler pair of objects using
      // the DH and PH objects defined above (tables are not
      // recomputed) for the total dataset (used at the end of the fit
      // to compute the optimal chi2) and the training and validation
      // subsets.
      DSVect.push_back(std::make_pair(DHc, new Denali::PredictionsHandler{PH}));
      DSVectt.push_back(std::make_pair(DH, new Denali::PredictionsHandler{PH, {std::shared_ptr<NangaParbat::Cut>(new NangaParbat::TrainingCut{*TrainingCut})}}));
      DSVectv.push_back(std::make_pair(DH, new Denali::PredictionsHandler{PH, {std::shared_ptr<NangaParbat::Cut>(new NangaParbat::TrainingCut{*ValidationCut})}}));
    }

  // NN Parameterisation
  // If "Include set" is true, the parametrization uses the
  // average over the replicas, regardless the replica
  // configuration provided in the runcard.
  NangaParbat::Parameterisation *NN_pPDFs = new Denali::NNADparameterisation(config["NNAD"], g,
                                                                             LHAPDF::mkPDFs(config["Predictions"]["Sets"]["unpolarised pdfset"]["name"].as<std::string>()),
                                                                             config["Predictions"]["mu0"].as<double>(), config["Predictions"]["Sets"]["unpolarised pdfset"]["member"].as<int>(),
                                                                             (config["Predictions"]["Include Std"] ? config["Predictions"]["Include Std"].as<bool>() : true), (config["Predictions"]["Multiplicative factor"] ? config["Predictions"]["Multiplicative factor"].as<double>() : 1.));

  // Initialiase chi2 objects for training and validation
  Denali::AnalyticChiSquare *chi2t = new Denali::AnalyticChiSquare{DSVectt, NN_pPDFs};
  Denali::AnalyticChiSquare *chi2v = new Denali::AnalyticChiSquare{DSVectv, NN_pPDFs};

  // Put initial parameters in a vector<double*> for initialising the
  // ceres solver.
  const int np = NN_pPDFs->GetParameterNumber();
  std::vector<double> pars = NN_pPDFs->GetParameters();
  std::vector<double *> initPars(np);
  for (int ip = 0; ip < np; ip++)
    initPars[ip] = new double(pars[ip]);

  // Allocate a "ceres::Problem" instance
  ceres::Problem problem;

  // Allocate a "Chi2CostFunction" instance to be fed to ceres for
  // minimisation
  problem.AddResidualBlock(chi2t, NULL, initPars);

  // Ceres-solver options
  ceres::Solver::Options options;
  options.max_num_iterations = config["Optimizer"]["max_num_iterations"].as<int>();
  options.minimizer_progress_to_stdout = true;
  if (config["Optimizer"]["use_nonmonotonic_steps"])
    options.use_nonmonotonic_steps = config["Optimizer"]["use_nonmonotonic_steps"].as<bool>();

  // Set all tolerances to zero to ensure that all fits get to the
  // maximum number of iterations without stopping.
  options.function_tolerance  = 0;
  options.gradient_tolerance  = 0;
  options.parameter_tolerance = 0;

  // Iteration callback
  options.update_state_every_iteration = true;
  Denali::IterationCallBack *callback = new Denali::IterationCallBack(true, OutputFolder, replica, initPars, chi2t, chi2v);
  options.callbacks.push_back(callback);

  // Summary object
  ceres::Solver::Summary summary;

  // Run the fit
  Solve(options, &problem, &summary);

  // Print summary to screen
  std::cout << summary.FullReport() << "\n" << std::scientific;

  // Get Best fit parameters determined by look-back.
  const std::vector<double> BestParameters = callback->GetBestParameters();
  const int BestIteration = callback->GetBestIteration();
  const double BestValidationChi2 = callback->GetBestValidationChi2();
  std::cout << "Best iteration = " << callback->GetBestIteration() << std::endl;
  std::cout << "Best validation chi2 = " << callback->GetBestValidationChi2() << std::endl;

  // Compute final chi2 and predictions
  Denali::AnalyticChiSquare *chi2f = new Denali::AnalyticChiSquare{DSVect, NN_pPDFs};
  chi2f->SetParameters(BestParameters);
  const double final_chi2 = chi2f->NangaParbat::ChiSquare::Evaluate();
  std::cout << "Final chi2 = " << final_chi2 << std::endl;

  // Output parameters into yaml file
  chi2v->SetParameters(BestParameters);
  double best_chi2v = chi2v->NangaParbat::ChiSquare::Evaluate();
  chi2t->SetParameters(BestParameters);
  double best_chi2t = chi2t->NangaParbat::ChiSquare::Evaluate();

  if (final_chi2 <= config["Optimizer"]["chi2_tolerance"].as<double>())
    {
      YAML::Emitter emitter;
      emitter << YAML::BeginSeq;
      emitter << YAML::Flow << YAML::BeginMap;
      emitter << YAML::Key << "replica" << YAML::Value << replica;
      for (const auto& Set_type : config["Predictions"]["Sets"])
        emitter << YAML::Key << Set_type.first.as<std::string>() << YAML::Value << Set_type.second["member"].as<int>();
      emitter << YAML::Key << "total chi2" << YAML::Value << final_chi2;
      emitter << YAML::Key << "training chi2" << YAML::Value << best_chi2t;
      emitter << YAML::Key << "validation chi2" << YAML::Value << best_chi2v;
      emitter << YAML::Key << "best iteration" << YAML::Value << BestIteration;
      emitter << YAML::Key << "best validation chi2" << YAML::Value << BestValidationChi2;
      emitter << YAML::Key << "Total time in second" << YAML::Value << summary.total_time_in_seconds;
      emitter << YAML::Key << "parameters" << YAML::Value << YAML::Flow << BestParameters;
      emitter << YAML::EndMap;
      emitter << YAML::EndSeq;
      emitter << YAML::Newline;
      std::ofstream fout;
      if (separate_replica_results)
        fout = std::ofstream(OutputFolder + "/SeparateBestParameters/BestParameters_" + std::to_string(replica) + ".yaml", std::ios::out | std::ios::app);
      else
        fout = std::ofstream(OutputFolder + "/BestParameters.yaml", std::ios::out | std::ios::app);

      fout << emitter.c_str();
      fout.close();
    }

  // Copy input card into the result folder
  system(("cp " + InputCardPath + " " + OutputFolder + "/config.yaml").c_str());

  // Create data folder in the result folder (if it does not exist)
  // and fill it with the data files that have been fitted. In the
  // case of closure tests the central value will correspond to the
  // predictions w/ or w/o fluctuaction according to the level.
  if (!NangaParbat::dir_exists(OutputFolder + "/data"))
    system(("mkdir " + OutputFolder + "/data").c_str());

  for (int i = 0; i < (int) DSVectt.size(); i++)
    {
      std::ofstream fout(OutputFolder + "/data/" + filenames[i]);
      fout << DSVectt[i].first->GetDataFile() << std::endl;;
      fout.close();
    }

  t.stop(true);
  return 0;
}
