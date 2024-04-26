/*---------------------------------------------------------------------*/
/*! \file
\brief Concrete mplementation of all the %NOX::NLN::CONSTRAINT::Interface::Required
       (pure) virtual routines.

\level 3


*/
/*---------------------------------------------------------------------*/

#ifndef FOUR_C_CONTACT_MESHTYING_NOXINTERFACE_HPP
#define FOUR_C_CONTACT_MESHTYING_NOXINTERFACE_HPP


#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_constraint_interface_preconditioner.hpp"
#include "4C_solver_nonlin_nox_constraint_interface_required.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

FOUR_C_NAMESPACE_OPEN


namespace CONTACT
{
  class MtNoxInterface : public NOX::NLN::CONSTRAINT::Interface::Required
  {
   public:
    /// constructor
    MtNoxInterface();

    /// initialize important member variables
    void Init(const Teuchos::RCP<STR::TIMINT::BaseDataGlobalState>& gstate_ptr);

    /** \brief Setup important new member variables
     *
     *  Supposed to be overloaded by derived classes. */
    virtual void Setup();

    /// @name Supported basic interface functions
    /// @{
    //! Returns the constraint right-hand-side norms [derived]
    double GetConstraintRHSNorms(const Epetra_Vector& F,
        NOX::NLN::StatusTest::QuantityType checkQuantity, ::NOX::Abstract::Vector::NormType type,
        bool isScaled) const override;

    /// Returns the root mean square (abbr.: RMS) of the Lagrange multiplier updates [derived]
    double GetLagrangeMultiplierUpdateRMS(const Epetra_Vector& xNew, const Epetra_Vector& xOld,
        double aTol, double rTol, NOX::NLN::StatusTest::QuantityType checkQuantity,
        bool disable_implicit_weighting) const override;

    /// Returns the increment norm of the largange multiplier DoFs
    double GetLagrangeMultiplierUpdateNorms(const Epetra_Vector& xNew, const Epetra_Vector& xOld,
        NOX::NLN::StatusTest::QuantityType checkQuantity, ::NOX::Abstract::Vector::NormType type,
        bool isScaled) const override;

    /// Returns the previous solution norm of the largange multiplier DoFs
    double GetPreviousLagrangeMultiplierNorms(const Epetra_Vector& xOld,
        NOX::NLN::StatusTest::QuantityType checkQuantity, ::NOX::Abstract::Vector::NormType type,
        bool isScaled) const override;
    /// @}

   protected:
    /// get the init indicator state
    inline const bool& IsInit() const { return isinit_; };

    /// get the setup indicator state
    inline const bool& IsSetup() const { return issetup_; };

    /// Check if Init() has been called
    inline void CheckInit() const
    {
      if (not IsInit()) FOUR_C_THROW("Call Init() first!");
    };

    /// Check if Init() and Setup() have been called, yet.
    inline void CheckInitSetup() const
    {
      if (not IsInit() or not IsSetup()) FOUR_C_THROW("Call Init() and Setup() first!");
    };


   protected:
    /// flag indicating if Init() has been called
    bool isinit_;

    /// flag indicating if Setup() has been called
    bool issetup_;

   private:
    //! global state data container
    Teuchos::RCP<STR::TIMINT::BaseDataGlobalState> gstate_ptr_;
  };

}  // namespace CONTACT


FOUR_C_NAMESPACE_CLOSE

#endif