/*----------------------------------------------------------------------*/
/*! \file
\brief Evaluation of off-diagonal blocks for monolithic STI

\level 2


 */
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_STI_MONOLITHIC_EVALUATE_OFFDIAG_HPP
#define FOUR_C_STI_MONOLITHIC_EVALUATE_OFFDIAG_HPP

#include "baci_config.hpp"

#include "baci_adapter_scatra_base_algorithm.hpp"
#include "baci_inpar_s2i.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_ParameterList.hpp>

BACI_NAMESPACE_OPEN

namespace ADAPTER
{
  class ScaTraBaseAlgorithm;
  class Coupling;
}  // namespace ADAPTER

namespace CORE::LINALG
{
  class SparseOperator;
  class MultiMapExtractor;
}  // namespace CORE::LINALG

namespace SCATRA
{
  class MeshtyingStrategyS2I;
  class ScaTraTimIntImpl;
}  // namespace SCATRA

namespace STI
{
  //! base class for evaluation of scatra-thermo off-diagonal blocks
  class ScatraThermoOffDiagCoupling
  {
   public:
    //! constructor
    explicit ScatraThermoOffDiagCoupling(
        Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo,
        Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo_interface,
        Teuchos::RCP<const Epetra_Map> full_map_scatra,
        Teuchos::RCP<const Epetra_Map> full_map_thermo,
        Teuchos::RCP<const Epetra_Map> interface_map_scatra,
        Teuchos::RCP<const Epetra_Map> interface_map_thermo, bool isale,
        Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> meshtying_strategy_scatra,
        Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> meshtying_strategy_thermo,
        Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra,
        Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> thermo);

    //! destructor
    virtual ~ScatraThermoOffDiagCoupling() = default;

    //! evaluation of domain contributions to scatra-thermo OD block
    void EvaluateOffDiagBlockScatraThermoDomain(
        Teuchos::RCP<CORE::LINALG::SparseOperator> scatrathermoblock);

    //! evaluation of interface contributions to scatra-thermo OD block
    virtual void EvaluateOffDiagBlockScatraThermoInterface(
        Teuchos::RCP<CORE::LINALG::SparseOperator> scatrathermoblockinterface) = 0;

    //! evaluation of domain contributions to thermo-scatra OD block
    void EvaluateOffDiagBlockThermoScatraDomain(
        Teuchos::RCP<CORE::LINALG::SparseOperator> thermoscatrablock);

    //! evaluation of interface contributions to thermo-scatra OD block
    virtual void EvaluateOffDiagBlockThermoScatraInterface(
        Teuchos::RCP<CORE::LINALG::SparseOperator> thermoscatrablockinterface) = 0;

   protected:
    //! map extractor associated with all degrees of freedom inside temperature field
    Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> BlockMapThermo() const
    {
      return block_map_thermo_;
    }

    //! map extractor associated with degrees of freedom on interface of temperature field
    Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> BlockMapThermoInterface() const
    {
      return block_map_thermo_interface_;
    }

    //! map extractor associated with all degrees of freedom inside scatra field
    Teuchos::RCP<const Epetra_Map> FullMapScatra() const { return full_map_scatra_; }

    //! map extractor associated with all degrees of freedom inside thermo field
    Teuchos::RCP<const Epetra_Map> FullMapThermo() const { return full_map_thermo_; }

    //! map associated with all degrees of freedom on thermo interface
    Teuchos::RCP<const Epetra_Map> InterfaceMapScaTra() const { return interface_map_scatra_; }

    //! map associated with all degrees of freedom on scatra interface
    Teuchos::RCP<const Epetra_Map> InterfaceMapThermo() const { return interface_map_thermo_; }

    //! problem with deforming mesh
    bool IsAle() const { return isale_; }

    //! meshtying strategy for scatra-scatra interface coupling on scatra discretization
    Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> MeshtyingStrategyScaTra() const
    {
      return meshtying_strategy_scatra_;
    }

    //! meshtying strategy for scatra-scatra interface coupling on scatra discretization
    Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> MeshtyingStrategyThermo() const
    {
      return meshtying_strategy_thermo_;
    }

    //! ScaTra subproblem
    Teuchos::RCP<SCATRA::ScaTraTimIntImpl> ScaTraField() { return scatra_->ScaTraField(); }

    //! Thermo subproblem
    Teuchos::RCP<SCATRA::ScaTraTimIntImpl> ThermoField() { return thermo_->ScaTraField(); }

   private:
    //! map extractor associated with all degrees of freedom inside temperature field
    Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo_;

    //! map extractor associated with degrees of freedom on interface of temperature field
    Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo_interface_;

    //! map extractor associated with all degrees of freedom inside scatra field
    Teuchos::RCP<const Epetra_Map> full_map_scatra_;

    //! map extractor associated with all degrees of freedom inside thermo field
    Teuchos::RCP<const Epetra_Map> full_map_thermo_;

    //! map associated with all degrees of freedom on thermo interface
    Teuchos::RCP<const Epetra_Map> interface_map_scatra_;

    //! map associated with all degrees of freedom on scatra interface
    Teuchos::RCP<const Epetra_Map> interface_map_thermo_;

    //! flag, if mesh deforms
    const bool isale_;

    //! meshtying strategy for scatra-scatra interface coupling on scatra discretization
    Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> meshtying_strategy_scatra_;

    //! meshtying strategy for scatra-scatra interface coupling on scatra discretization
    Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> meshtying_strategy_thermo_;

    //! ScaTra subproblem
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra_;

    //! Thermo subproblem
    Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> thermo_;
  };

  //! evaluation of scatra-thermo off-diagonal blocks for matching nodes
  class ScatraThermoOffDiagCouplingMatchingNodes : public ScatraThermoOffDiagCoupling
  {
   public:
    //! constructor
    explicit ScatraThermoOffDiagCouplingMatchingNodes(
        Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo,
        Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo_interface,
        Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo_interface_slave,
        Teuchos::RCP<const Epetra_Map> full_map_scatra,
        Teuchos::RCP<const Epetra_Map> full_map_thermo,
        Teuchos::RCP<const Epetra_Map> interface_map_scatra,
        Teuchos::RCP<const Epetra_Map> interface_map_thermo, bool isale,
        Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> meshtying_strategy_scatra,
        Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> meshtying_strategy_thermo,
        Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra,
        Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> thermo);


    //! evaluation of interface contributions to scatra-thermo off-diagonal block
    void EvaluateOffDiagBlockScatraThermoInterface(
        Teuchos::RCP<CORE::LINALG::SparseOperator> scatrathermoblockinterface) override;

    //! evaluation of interface contributions to thermo-scatra off-diagonal block
    void EvaluateOffDiagBlockThermoScatraInterface(
        Teuchos::RCP<CORE::LINALG::SparseOperator> thermoscatrablockinterface) override;

   private:
    //! map extractor associated with degrees of freedom on interface (slave side) of temperature
    //! field
    Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> BlockMapThermoInterfaceSlave() const
    {
      return block_map_thermo_interface_slave_;
    }

    //! copy slave entries to master side scaled by -1.0
    void CopySlaveToMasterScatraThermoInterface(
        Teuchos::RCP<const CORE::LINALG::SparseOperator> slavematrix,
        Teuchos::RCP<CORE::LINALG::SparseOperator>& mastermatrix);

    //! evaluate condition on slave side
    void EvaluateScatraThermoInterfaceSlaveSide(
        Teuchos::RCP<CORE::LINALG::SparseOperator> slavematrix);

    //! map extractor associated with degrees of freedom on interface (slave side) of temperature
    //! field
    Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo_interface_slave_;
  };

  //! evaluation of scatra-thermo off-diagonal blocks for standard Mortar on scatra discretization
  //! and condensed Bubnov Mortar on thermo discretization
  class ScatraThermoOffDiagCouplingMortarStandard : public ScatraThermoOffDiagCoupling
  {
   public:
    //! constructor
    explicit ScatraThermoOffDiagCouplingMortarStandard(
        Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo,
        Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo_interface,
        Teuchos::RCP<const Epetra_Map> full_map_scatra,
        Teuchos::RCP<const Epetra_Map> full_map_thermo,
        Teuchos::RCP<const Epetra_Map> interface_map_scatra,
        Teuchos::RCP<const Epetra_Map> interface_map_thermo, bool isale,
        Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> meshtying_strategy_scatra,
        Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> meshtying_strategy_thermo,
        Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra,
        Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> thermo);

    //! evaluation of interface contributions to scatra-thermo off-diagonal block
    void EvaluateOffDiagBlockScatraThermoInterface(
        Teuchos::RCP<CORE::LINALG::SparseOperator> scatrathermoblockinterface) override;

    //! evaluation of interface contributions to thermo-scatra off-diagonal block
    void EvaluateOffDiagBlockThermoScatraInterface(
        Teuchos::RCP<CORE::LINALG::SparseOperator> thermoscatrablockinterface) override;
  };

  //! build specific off diagonal coupling object
  Teuchos::RCP<STI::ScatraThermoOffDiagCoupling> BuildScatraThermoOffDiagCoupling(
      const INPAR::S2I::CouplingType& couplingtype,
      Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo,
      Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo_interface,
      Teuchos::RCP<const CORE::LINALG::MultiMapExtractor> block_map_thermo_interface_slave,
      Teuchos::RCP<const Epetra_Map> full_map_scatra,
      Teuchos::RCP<const Epetra_Map> full_map_thermo,
      Teuchos::RCP<const Epetra_Map> interface_map_scatra,
      Teuchos::RCP<const Epetra_Map> interface_map_thermo, bool isale,
      Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> meshtying_strategy_scatra,
      Teuchos::RCP<const SCATRA::MeshtyingStrategyS2I> meshtying_strategy_thermo,
      Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> scatra,
      Teuchos::RCP<ADAPTER::ScaTraBaseAlgorithm> thermo);

}  // namespace STI
BACI_NAMESPACE_CLOSE

#endif