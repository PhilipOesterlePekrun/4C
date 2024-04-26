/*---------------------------------------------------------------------*/
/*! \file
\brief This class provides the functionality to use contact with Lagrangian
multipliers for monolithical coupled multifield problems!
Therefore ApplyForceStiffCmt() & Recover() are overloaded by this class and
do nothing, as they are called directly in the structure. To use the contact
the additional methods ApplyForceStiffCmtCoupled() & RecoverCoupled() have
to be called!

\level 3


*/
/*---------------------------------------------------------------------*/
#ifndef FOUR_C_CONTACT_MONOCOUPLED_LAGRANGE_STRATEGY_HPP
#define FOUR_C_CONTACT_MONOCOUPLED_LAGRANGE_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_contact_lagrange_strategy.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace ADAPTER
{
  class Coupling;
}

namespace CONTACT
{
  /*!
   \brief Contact solving strategy with (standard/dual) Lagrangian multipliers.

   This is a specialization of the abstract contact algorithm as defined in AbstractStrategy.
   For a more general documentation of the involved functions refer to CONTACT::AbstractStrategy.

   */
  class MonoCoupledLagrangeStrategy : public LagrangeStrategy
  {
   public:
    /*!
    \brief Standard Constructor

    */
    MonoCoupledLagrangeStrategy(const Teuchos::RCP<CONTACT::AbstractStratDataContainer>& data_ptr,
        const Epetra_Map* DofRowMap, const Epetra_Map* NodeRowMap, Teuchos::ParameterList params,
        std::vector<Teuchos::RCP<CONTACT::Interface>> interface, int dim,
        Teuchos::RCP<Epetra_Comm> comm, double alphaf, int maxdof);


    // Overload CONTACT::AbstractStrategy::ApplyForceStiffCmt as this is called in the structure
    // --> to early for monolithically coupled algorithms!
    void ApplyForceStiffCmt(Teuchos::RCP<Epetra_Vector> dis,
        Teuchos::RCP<CORE::LINALG::SparseOperator>& kt, Teuchos::RCP<Epetra_Vector>& f,
        const int step, const int iter, bool predictor) override
    {
      if (has_to_evaluate_ && 0)
        FOUR_C_THROW(
            "MonoCoupledLagrangeStrategy::You have to call ApplyForceStiffCmtCoupled() for Contact "
            "Evaluation!");  // what to do in the predictor?
      has_to_evaluate_ = true;
      return;
    };

    // Overload CONTACT::LagrangeStrategy::Recover as this is called in the structure --> no
    // enought information available for monolithically coupled algorithms!
    void Recover(Teuchos::RCP<Epetra_Vector> disi) override
    {
      if (has_to_recover_ && 0)
        FOUR_C_THROW(
            "MonoCoupledLagrangeStrategy::You have to call RecoverCoupled() for Contact Recovery!");
      has_to_recover_ = true;
      return;
    };

    //! @name Access methods

    //@}

    //! @name Evaluation methods

    // Alternative Method to CONTACT::AbstractStrategy::ApplyForceStiffCmt for monolithically
    // coupled algorithms
    virtual void ApplyForceStiffCmtCoupled(Teuchos::RCP<Epetra_Vector> dis,
        Teuchos::RCP<CORE::LINALG::SparseOperator>& k_ss,
        std::map<int, Teuchos::RCP<CORE::LINALG::SparseOperator>*> k_sx,
        Teuchos::RCP<Epetra_Vector>& rhs_s, const int step, const int iter, bool predictor);

    // Alternative Method to CONTACT::AbstractStrategy::ApplyForceStiffCmt for monolithically
    // coupled algorithms
    virtual void ApplyForceStiffCmtCoupled(Teuchos::RCP<Epetra_Vector> dis,
        Teuchos::RCP<CORE::LINALG::SparseOperator>& k_ss,
        Teuchos::RCP<CORE::LINALG::SparseOperator>& k_sx, Teuchos::RCP<Epetra_Vector>& rhs_s,
        const int step, const int iter, bool predictor);

    // Alternative Method to CONTACT::LagrangeStrategy::Recover as this is called in the structure
    // --> no enought information available for monolithically coupled algorithms!
    /*!
    \brief Recovery method

    We only recover the Lagrange multipliers here, which had been
    statically condensated during the setup of the global problem!
    Optionally satisfaction or violation of the contact boundary
    conditions can be checked, too.*/
    virtual void RecoverCoupled(
        Teuchos::RCP<Epetra_Vector> disi, std::map<int, Teuchos::RCP<Epetra_Vector>> inc);

    virtual void RecoverCoupled(Teuchos::RCP<Epetra_Vector> disi, Teuchos::RCP<Epetra_Vector> inc);

    void EvaluateOffDiagContact(Teuchos::RCP<CORE::LINALG::SparseOperator>& kteff,
        int Column_Block_Id);  // condensation for all off diagonal matrixes k_s? in monolithically
                               // coupled problems!

   protected:
    // don't want = operator and cctor
    MonoCoupledLagrangeStrategy operator=(const MonoCoupledLagrangeStrategy& old) = delete;
    MonoCoupledLagrangeStrategy(const MonoCoupledLagrangeStrategy& old) = delete;

    void SaveCouplingMatrices(Teuchos::RCP<CORE::LINALG::SparseMatrix> dhat,
        Teuchos::RCP<CORE::LINALG::SparseMatrix> mhataam,
        Teuchos::RCP<CORE::LINALG::SparseMatrix> invda) override;

    std::map<int, Teuchos::RCP<CORE::LINALG::SparseOperator>>
        csx_s_;  // offdiagonal coupling stiffness blocks on slave side!

    Teuchos::RCP<CORE::LINALG::SparseMatrix> dhat_;
    Teuchos::RCP<CORE::LINALG::SparseMatrix> mhataam_;
    Teuchos::RCP<CORE::LINALG::SparseMatrix> invda_;

    Teuchos::RCP<Epetra_Vector>
        lambda_;  // current vector of Lagrange multipliers(for poro no pen.) at t_n+1
    Teuchos::RCP<Epetra_Vector>
        lambdaold_;  // old vector of Lagrange multipliers(for poro no pen.) at t_n

    //! pure useage safty flags
    bool has_to_evaluate_;  // checks if ApplyForceStiffCmtCoupled() after every call of
                            // ApplyForceStiffCmt()
    bool has_to_recover_;   // checks if RecoverCoupled() after every call of Recover()

  };  // class MonoCoupledLagrangeStrategy
}  // namespace CONTACT


FOUR_C_NAMESPACE_CLOSE

#endif