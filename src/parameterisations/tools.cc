//
// Authors: Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//          Amedeo Chiefa
//

#include "Denali/tools.h"

namespace Denali
{
  //_________________________________________________________________________
  std::map<int, double> Compute_Std(double const& x, double const& Q, std::vector<LHAPDF::PDF*> UnpPDF, bool const& AddMean, double const& r)
  {
    std::map<int, double> Std  = UnpPDF[1]->xfxQ(x, Q);
    std::map<int, double> av2 = Std;
    std::for_each(av2.begin(), av2.end(), [] (std::pair<const int, double>& p)
    {
      p.second = pow(p.second, 2);
    });
    for (int i = 2; i < (int) UnpPDF.size(); i++)
      {
        const std::map<int, double> f = UnpPDF[i]->xfxQ(x, Q);
        std::for_each(Std.begin(), Std.end(), [&] (std::pair<const int, double>& p)
        {
          p.second += f.at(p.first);
        });
        std::for_each(av2.begin(), av2.end(), [&] (std::pair<const int, double>& p)
        {
          p.second += pow(f.at(p.first), 2);
        });
      }
    const int nrep = UnpPDF.size() - 1;
    std::for_each(Std.begin(), Std.end(), [&] (std::pair<const int, double>& p)
    {
      p.second = std::abs((AddMean ? p.second / nrep : 0)) + r * sqrt( av2.at(p.first) / nrep - pow(p.second / nrep, 2) );
    });

    return Std;
  }
}