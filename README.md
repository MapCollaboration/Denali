[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10933177.svg)](https://doi.org/10.5281/zenodo.10933177)

![alt text](resources/Denali.jpg "Denali")

# Denali

`Denali` is a code devoted to the extraction of collinear longitudinally polarised parton distribution functions.

## Requirements

In order for the code to pe compiled, the following dependencies need to be preinstalled:

- [`NangaParbat`](https://github.com/MapCollaboration/NangaParbat)
- [`apfelxx`](https://github.com/vbertone/apfelxx)
- [`NNAD`](https://github.com/rabah-khalek/NNAD)
- [`ceres-solver`](http://ceres-solver.org)
- [`LHAPDF`](https://lhapdf.hepforge.org)
- [`yaml-cpp`](https://github.com/jbeder/yaml-cpp)
- [`GSL`](https://www.gnu.org/software/gsl/)

## Compilation and installation

The `Denali` library only relies on `cmake` for configuration and installation. This is done by following the standard procedure:
```
mkdir build
cd build
cmake ..
make -j
make install
```
The library can be uninstalled by running:
```
make clean
xargs rm < install_manifest.txt
```

## Usage

The relevant source code to perform a fit and analyse the results can be found in the `run/` folder. However, in the following we assume to be in the `build/run/` folder that will be created after the `cmake` procedure detailed above and that contains the executables. In this folder we need to create a subfolder called `fit/` that will be used to store the results. A short description of each code is as follows:

1. `Optimize`: this code is responsible for performing the fit. An example of the usage of this code is:
    ```
    ./Optimize 1 ../../config/example.yaml ../../data/ fit/
    ```
    The first argument indicates the Monte Carlo replica index, the second points to the input card containing the main parameters of the fit as well as the data sets to be fitted (see [here](config/example.yaml) for a commented example, make sure to have the LHAPDF sets corresponding to the unpolarised PDFs and FFs needed to run the fit locally installed), the third points to the folder where the data files are contained, and the last argument is the folder where the results of the fit will be dumped. This code produces in the folder `fit/` a file called `BestParameters.yaml` that contains the best fit parameters of the NN along with some additional information such as the training, validation, and global χ<sup>2</sup>'s. In addition, this code will place in the `fit/` folder two additional subfolders, `log/` and `data/`, containing respectively the log file of the fit and the data files for the fitted experimental sets. If a new fit with a different Monte Carlo replica index is run specifying the `fit/` as a destination for the results, the best fit parameters of this new fit will be appended to the `BestParameters.yaml` file and a new log file will be created in the `fit/log/` subfolder. Notice that Monte Carlo replica indices equal or larger than one correspond to actual random fluctuations of the central values of the experimental data, while the index 0 corresponds to a fit to the central values, i.e. no fluctuations are performed. If the code `Optimize` is run without any arugments it will prompt a short usage description.

2. `LHAPDFGrid`: this code produces an LHAPDF grid for a given fit. In order to produce a grid for the fit in the `fit/` folder, the syntax is:
    ```
    ./LHAPDFGrid fit/
    ```
    The produced grid can be found in the `fit/` folder under the default name `LHAPDFSet`. This set will eventually be used for analysing the results. It possible to customise the output by providing the script with additional options. Specifically, it possible to change the default name and to specificy the number of replicas to be produced. The last option is applicable only when more fits have been run in the `fit/` folder and the number of user-provided replicas does not exceed the number of fits. For example, assuming to have performed 120 fits, the following:
    ```
    ./LHAPDFGrid fit/ MySetForPolPDFs 100
    ```
    will produce a set named `MySetForPolPDFs` with 101 replicas, where the zero-replica is the average over the following 100. In addition, the `LHAPDFGrid` code sorts the replicas in the global χ<sup>2</sup> from the smallest to the largest. Therefore, the resulting set will containg the 100 replicas out of 120 with best global χ<sup>2</sup>'s. Also in this case, if the code `LHAPDFGrid` is run without any arugments it will prompt a short usage description.

3. `ComputeChi2s`: as the name says, the code computes the χ<sup>2</sup>'s using the fit results. The syntax is:
    ```
    ./ComputeChi2s fit/
    ```
    This code relies on the presence of an LHAPDF grid in the fit folder named `LHAPDFSet` and will result in the creation of the file `fit/Chi2s.yaml` containing the χ<sup>2</sup> for the single experiments included in the fit. It is also possible to change the name of the polarised PDF set to be used to compute the χ<sup>2</sup>'s. For example:
    ```
    ./ComputeChi2s fit/ MySetForPolPDFs
    ```
    will compute the χ<sup>2</sup>'s using the `MySetForPolPDFs` set that has to be either in the `fit/` folder or in the LHAPDF data directory (that can be retrieved by running the command `lhapdf-config --datadir` from shell).

4. `Predictions`: this code computes the predictions for all the points included in the fit. It is used as:
    ```
    ./Predictions fit/
    ```
    Also this code relies on the presence of an LHAPDF grid in the fit folder named `LHAPDFSet` and will produce the file `fit/Predictions.yaml`. Again, it is possible to use a different name for the polarised PDF set to be used to compute the χ<sup>2</sup>'s. For example:
    ```
    ./Predictions fit/ MySetForPolPDFs
    ```
    will compute the predictions using the `MySetForPolPDFs` set that has to be either in the `fit/` folder or in the LHAPDF data directory.

The results produced by the codes described above can finally be visualised by copying  into the `fit/` folder and running the template `jupyter` notebook [`AnalysePredictions.ipynb`](analysis/Analysis_template/AnalysePredictions.ipynb) that is in the `analysis/` folder. This is exactly how the fit of polarised PDFs documented in the reference below has been obtained and any user should be able to reproduce it by following the steps above. For reference, we have linked the folder of the baseline fits [here](Results) along with the corresponding `jupyter` notebooks.

## Reference

If you use this code or the PDF sets listed below, please refer to and cite the following reference:

- Valerio Bertone, Amedeo Chiefa, Emanuele R. Nocera, "Helicity-dependent parton distribution functions at next-to-next-to-leading order accuracy from inclusive and semi-inclusive deep-inelastic scattering data", arXiv:2404.04712.

The baseline fits in the LHAPDF format produced with this publications can be found in [this](PDFSets/2404.04712) folder. The additional variant PDF sets discussed in this paper are also available from the authors upon request.

## Contacts

For additional information or questions, contact us using the email adresses below:

- Valerio Bertone: valerio.bertone@cern.ch
- Amedeo Chiefa: amedeo.chiefa@ed.ac.uk
- Emanuele R. Nocera: emanueleroberto.nocera@unito.it

