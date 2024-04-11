/*----------------------------------------------------------------------*/
/*! \file

\brief singleton class holding all static electrochemistry parameters required for element
evaluation

This singleton class holds all static electrochemistry parameters required for element evaluation.
All parameters are usually set only once at the beginning of a simulation, namely during
initialization of the global time integrator, and then never touched again throughout the
simulation. This parameter class needs to coexist with the general parameter class holding all
general static parameters required for scalar transport element evaluation.

\level 2

*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_SCATRA_ELE_PARAMETER_ELCH_HPP
#define FOUR_C_SCATRA_ELE_PARAMETER_ELCH_HPP

#include "baci_config.hpp"

#include "baci_inpar_elch.hpp"
#include "baci_scatra_ele_parameter_base.hpp"

BACI_NAMESPACE_OPEN

namespace DRT
{
  namespace ELEMENTS
  {
    // class implementation
    class ScaTraEleParameterElch : public ScaTraEleParameterBase
    {
     public:
      //! singleton access method
      static ScaTraEleParameterElch* Instance(
          const std::string& disname  //!< name of discretization
      );

      //! return flag for coupling of lithium-ion flux density and electric current density at
      //! Dirichlet and Neumann boundaries
      bool BoundaryFluxCoupling() const { return boundaryfluxcoupling_; };

      //! set parameters
      void SetParameters(Teuchos::ParameterList& parameters  //!< parameter list
          ) override;

      //! return type of closing equation for electric potential
      INPAR::ELCH::EquPot EquPot() const { return equpot_; };

      //! return Faraday constant
      double Faraday() const { return faraday_; };

      //! return the (universal) gas constant
      double GasConstant() const { return gas_constant_; };

      //! return dielectric constant
      double Epsilon() const { return epsilon_; };

      //! return constant F/RT
      double FRT() const { return frt_; };

      //! return the homogeneous temperature in the scatra field (can be time dependent)
      double Temperature() const { return temperature_; }

     private:
      //! private constructor for singletons
      ScaTraEleParameterElch(const std::string& disname  //!< name of discretization
      );

      //! flag for coupling of lithium-ion flux density and electric current density at Dirichlet
      //! and Neumann boundaries
      bool boundaryfluxcoupling_;

      //! equation used for closing of the elch-system
      enum INPAR::ELCH::EquPot equpot_;

      //! Faraday constant
      double faraday_;

      //! (universal) gas constant
      double gas_constant_;

      //! dielectric constant
      const double epsilon_;

      //! pre-calculation of regularly used constant F/RT
      //! (a division is much more expensive than a multiplication)
      double frt_;

      //! homogeneous temperature within the scalar transport field (can be time dependent)
      double temperature_;
    };
  }  // namespace ELEMENTS
}  // namespace DRT
BACI_NAMESPACE_CLOSE

#endif