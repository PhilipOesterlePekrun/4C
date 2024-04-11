/*-----------------------------------------------------------*/
/*! \file

\brief Class to assemble the fbi coupling contributions


\level 1

*/
/*-----------------------------------------------------------*/


#ifndef FOUR_C_FBI_FLUID_ASSEMBLY_STRATEGY_HPP
#define FOUR_C_FBI_FLUID_ASSEMBLY_STRATEGY_HPP

#include "baci_config.hpp"

#include "baci_utils_exceptions.hpp"

#include <Epetra_FEVector.h>
#include <Teuchos_RCP.hpp>

#include <vector>

// Forward declarations.
class Epetra_FEVector;
class Epetra_Vector;

BACI_NAMESPACE_OPEN

namespace CORE::LINALG
{
  class SparseMatrix;
  class SparseOperator;
  class SerialDenseVector;
  class SerialDenseMatrix;
}  // namespace CORE::LINALG
namespace DRT
{
  class Discretization;
}
namespace BEAMINTERACTION
{
  class BeamContactPair;
}

namespace FBI
{
  namespace UTILS
  {
    /**
     * \brief This class assembles the contributions of fluid beam mesh tying pairs into the global
     * matrices in the standard case of a fluid without internal mesh tying.
     *
     * The form of the fluid matrix and in an extension the required assembly method
     * depend on the fluid problem, particularly if mesh tying is used.
     */
    class FBIAssemblyStrategy
    {
     public:
      /**
       * \brief Destructor.
       */
      virtual ~FBIAssemblyStrategy() = default;

      /**
       * \brief Calls the correct assembly method for the used global fluid matrix depending on the
       * fluid problem
       *
       * \param[in, out] cff fluid coupling matrix
       * \param[in] eid element gid
       * \param[in] Aele dense matrix to be assembled
       * \param[in] lmrow vector with row gids
       * \param[in] lmrowowner vector with owner procs of row gids
       * \param[in] lmcol vector with column gids
       */
      virtual void AssembleFluidMatrix(Teuchos::RCP<CORE::LINALG::SparseOperator> cff, int elegid,
          const std::vector<int>& lmstride, const CORE::LINALG::SerialDenseMatrix& elemat,
          const std::vector<int>& lmrow, const std::vector<int>& lmrowowner,
          const std::vector<int>& lmcol);

      /**
       * \brief Assembles element coupling contributions into global coupling matrices and force
       * vectors needed for partitioned algorithms
       *
       * \param[in] discretization1 discretization to the first field
       * \param[in] discretization2 discretization to the second field
       * \param[in] elegids vector of length 2 containing the global IDs of the interacting elements
       * \param[in] elevec vector of length 2 containing the discrete element residual vectors of
       * the interacting elements
       * \param[in, out] c22 coupling matrix relating DOFs in the second
       * discretization to each other
       * \param[in, out] c11 coupling matrix relating DOFs in the first
       * discretization to each other
       * \param[in, out] c12 coupling matrix relating DOFs in the
       * second discretization to DOFs in the first discretization
       * \param[in, out] c21 coupling
       * matrix relating DOFs in the first discretization to DOFs in the second discretization
       *
       */
      virtual void Assemble(const DRT::Discretization& discretization1,
          const DRT::Discretization& discretization2, std::vector<int> const& elegid,
          std::vector<CORE::LINALG::SerialDenseVector> const& elevec,
          std::vector<std::vector<CORE::LINALG::SerialDenseMatrix>> const& elemat,
          Teuchos::RCP<Epetra_FEVector>& f1, Teuchos::RCP<Epetra_FEVector>& f2,
          Teuchos::RCP<CORE::LINALG::SparseMatrix>& c11,
          Teuchos::RCP<CORE::LINALG::SparseOperator> c22,
          Teuchos::RCP<CORE::LINALG::SparseMatrix>& c12,
          Teuchos::RCP<CORE::LINALG::SparseMatrix>& c21);
    };
  }  // namespace UTILS
}  // namespace FBI

BACI_NAMESPACE_CLOSE

#endif