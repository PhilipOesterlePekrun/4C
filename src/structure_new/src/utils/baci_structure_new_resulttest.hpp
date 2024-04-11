/*-----------------------------------------------------------*/
/*! \file
\brief Structure specific result test class


\level 3

*/
/*-----------------------------------------------------------*/


#ifndef FOUR_C_STRUCTURE_NEW_RESULTTEST_HPP
#define FOUR_C_STRUCTURE_NEW_RESULTTEST_HPP

#include "baci_config.hpp"

#include "baci_lib_resulttest.hpp"
#include "baci_utils_exceptions.hpp"

#include <Epetra_Vector.h>

#include <optional>

BACI_NAMESPACE_OPEN

// forward declarations
namespace DRT
{
  class Discretization;
}  // namespace DRT

namespace CORE::LINALG
{
  class Solver;
}  // namespace CORE::LINALG

namespace IO
{
  class DiscretizationWriter;
}  // namespace IO

namespace STR
{
  namespace TIMINT
  {
    class BaseDataGlobalState;
  }  // namespace TIMINT
  namespace MODELEVALUATOR
  {
    class Data;
  }  // namespace MODELEVALUATOR

  /*! \brief Structure specific result test class */
  class ResultTest : public DRT::ResultTest
  {
    /// possible status flag for the result test
    enum class Status : char
    {
      evaluated,
      unevaluated
    };

   public:
    //! Constructor for time integrators of general kind
    //! \author bborn \date 06/08 (originally)
    ResultTest();

    //! initialization of class variables
    virtual void Init(
        const STR::TIMINT::BaseDataGlobalState& gstate, const STR::MODELEVALUATOR::Data& data);

    //! setup of class variables
    virtual void Setup();

    //! \brief structure version of nodal value tests
    //!
    //! Possible position flags are "dispx", "dispy", "dispz",
    //!                             "velx", "vely", "velz",
    //!                             "accx", "accy", "accz"
    //!                             "stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_xz",
    //!                             "stress_yz"
    //!
    //! \note The type of stress that is used for testing has to be specified in IO->STRUCT_STRESS
    void TestNode(INPUT::LineDefinition& res, int& nerr, int& test_count) override;

    /*! \brief test special quantity not associated with a particular element or node
     *
     *  \param[in] res          input file line containing result test specification
     *  \param[out] nerr        updated number of failed result tests
     *  \param[out] test_count  updated number of result tests
     *  \param[out] uneval_test_count  updated number of unevaluated tests
     *
     *  \author hiermeier \date 11/17 */
    void TestSpecial(
        INPUT::LineDefinition& res, int& nerr, int& test_count, int& uneval_test_count) override;

   protected:
    /// get the indicator state
    inline const bool& IsInit() const { return isinit_; };

    /// get the indicator state
    inline const bool& IsSetup() const { return issetup_; };

    /// Check if Init() and Setup() have been called
    inline void CheckInitSetup() const
    {
      dsassert(IsInit() and IsSetup(), "Call Init() and Setup() first!");
    }

    /// Check if Init() has been called
    inline void CheckInit() const { dsassert(IsInit(), "Call Init() first!"); }

   private:
    /** \brief Get the result of the special structural quantity
     *
     *  The %special_status flag is used to identify circumstances where an
     *  evaluation of the status test is not possible, because the quantity
     *  is not accessible. One example is the number of nonlinear iterations
     *  for a step which is not a part of the actual simulation. Think of a
     *  restart scenario.
     *
     *  \param[in]  quantity        name of the special quantity
     *  \param[out] special_status  status of the specual result test
     *  \return  The value for the subsequent comparison.
     *
     *  \author hiermeier \date 11/17 */
    std::optional<double> GetSpecialResult(
        const std::string& quantity, Status& special_status) const;

    /** \brief Get the last number of linear iterations
     *
     * If the number of iterations for the desired step is accessible, it will
     *  be returned and the special_status flag is set to evaluated. Note that
     *  the step number is part of the quantity name and will be automatically
     *  extracted. The used format is
     *
     *                       lin_iter_step_<INT>
     *
     *  The integer <INT> must be at the very last position separated by an
     *  underscore.
     *
     *  \param[in]  quantity        name of the special quantity
     *  \param[out] special_status  status of the special result test
     *  \return  The number of linear iterations, if possible. Otherwise -1.
     *
     */
    std::optional<int> GetLastLinIterationNumber(
        const std::string& quantity, Status& special_status) const;

    /** \brief Get the number of nonlinear iterations
     *
     *  If the number of iterations for the desired step is accessible, it will
     *  be returned and the special_status flag is set to evaluated. Note that
     *  the step number is part of the quantity name and will be automatically
     *  extracted. The used format is
     *
     *                       num_iter_step_<INT>
     *
     *  The integer <INT> must be at the very last position separated by an
     *  underscore.
     *
     *  \param[in]  quantity        name of the special quantity
     *  \param[out] special_status  status of the special result test
     *  \return  The number of nonlinear iterations, if possible. Otherwise -1.
     *
     *  \author hiermeier \date 11/17 */
    std::optional<int> GetNlnIterationNumber(
        const std::string& quantity, Status& special_status) const;

    std::optional<int> GetNodesPerProcNumber(
        const std::string& quantity, Status& special_status) const;


    /** \brief Get the value for a specific energy (internal, kinetic, total, etc.)
     *
     *  If the energy is accessible, it will be returned and special_status flag is set to
     *  evaluated. If not, error is thrown in STR::MODELEVALUATOR::Data
     *
     *  \param[in]  quantity        name of the energy
     *  \param[out] special_status  status of the special result test
     *  \return     The requested energy
     *
     *  \author kremheller \date 11/19 */
    std::optional<double> GetEnergy(const std::string& quantity, Status& special_status) const;

   protected:
    //! flag which indicates if the Init() routine has already been called
    bool isinit_;

    //! flag which indicates if the Setup() routine has already been called
    bool issetup_;

   private:
    //! our discretisation
    Teuchos::RCP<const DRT::Discretization> strudisc_;
    // our solution
    //! global displacement DOFs
    Teuchos::RCP<const Epetra_Vector> disn_;
    //! global material displacement DOFs
    Teuchos::RCP<const Epetra_Vector> dismatn_;
    //! global velocity DOFs
    Teuchos::RCP<const Epetra_Vector> veln_;
    //! global acceleration DOFs
    Teuchos::RCP<const Epetra_Vector> accn_;
    //! global reaction DOFs
    Teuchos::RCP<const Epetra_Vector> reactn_;
    /* NOTE: these have to be present explicitly
     * as they are not part of the problem instance like in fluid3
     */

    //! pointer to the global state object of the structural time integration
    Teuchos::RCP<const STR::TIMINT::BaseDataGlobalState> gstate_;
    //! pointer to the data container of the structural time integration
    Teuchos::RCP<const STR::MODELEVALUATOR::Data> data_;
  };  // class ResultTest

  /*----------------------------------------------------------------------------*/
  /** \brief Get the integer at the very last position of a name string
   *
   *  \pre The integer must be separated by an underscore from the prefix, e.g.
   *              any-name-even_with_own_underscores_3
   *       The method will return 3 in this case.
   *
   *  \param[in] name  string name to extract from
   *  \return Extracted integer at the very last position of the name.
   *
   *  \author hiermeier \date 11/17 */
  int GetIntegerNumberAtLastPositionOfName(const std::string& quantity);

}  // namespace STR


BACI_NAMESPACE_CLOSE

#endif