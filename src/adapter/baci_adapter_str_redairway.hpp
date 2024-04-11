/*----------------------------------------------------------------------*/
/*! \file

\brief Structure field adapter for coupling with reduced-D airway trees


\level 3

*/

/*----------------------------------------------------------------------*/
/* macros */


#ifndef FOUR_C_ADAPTER_STR_REDAIRWAY_HPP
#define FOUR_C_ADAPTER_STR_REDAIRWAY_HPP
/*----------------------------------------------------------------------*/
/* headers */
#include "baci_config.hpp"

#include "baci_adapter_str_wrapper.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

BACI_NAMESPACE_OPEN


namespace DRT
{
  class Condition;
}


namespace ADAPTER
{
  class StructureRedAirway : public StructureWrapper
  {
   public:
    /// Constructor
    StructureRedAirway(Teuchos::RCP<Structure> stru);

    /// set pressure calculated from reduced-d airway tree
    void SetPressure(Teuchos::RCP<Epetra_Vector> couppres);

    /// calculate outlet fluxes for reduced-d airway tree
    void CalcFlux(
        Teuchos::RCP<Epetra_Vector> coupflux, Teuchos::RCP<Epetra_Vector> coupvol, double dt);

    /// calculate volume
    void CalcVol(std::map<int, double>& V);

    /// calculate initial volume
    void InitVol();

    //! (derived)
    void Update() override;

   private:
    /// map between coupling ID and conditions on structure
    std::map<int, DRT::Condition*> coupcond_;

    /// map of coupling IDs
    Teuchos::RCP<Epetra_Map> coupmap_;

    std::map<int, double> Vn_;
    std::map<int, double> Vnp_;
  };

}  // namespace ADAPTER
BACI_NAMESPACE_CLOSE

#endif