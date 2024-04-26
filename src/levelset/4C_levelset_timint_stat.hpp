/*----------------------------------------------------------------------*/
/*! \file

\brief stationary time integration scheme for level-set problems (for coupled problems only)
       just a dummy

\level 2


*----------------------------------------------------------------------*/

#ifndef FOUR_C_LEVELSET_TIMINT_STAT_HPP
#define FOUR_C_LEVELSET_TIMINT_STAT_HPP

#include "4C_config.hpp"

#include "4C_levelset_algorithm.hpp"
#include "4C_scatra_timint_stat.hpp"

FOUR_C_NAMESPACE_OPEN


namespace SCATRA
{
  class LevelSetTimIntStationary : public LevelSetAlgorithm, public TimIntStationary
  {
   public:
    /// Standard Constructor
    LevelSetTimIntStationary(Teuchos::RCP<DRT::Discretization> dis,
        Teuchos::RCP<CORE::LINALG::Solver> solver, Teuchos::RCP<Teuchos::ParameterList> params,
        Teuchos::RCP<Teuchos::ParameterList> sctratimintparams,
        Teuchos::RCP<Teuchos::ParameterList> extraparams,
        Teuchos::RCP<IO::DiscretizationWriter> output);


    /// initialize time-integration scheme
    void Init() override;

    /// setup time-integration scheme
    void Setup() override;

    /// read restart data
    void ReadRestart(const int step, Teuchos::RCP<IO::InputControl> input = Teuchos::null) override
    {
      FOUR_C_THROW("You should not need this function!");
      return;
    };

    /// redistribute the scatra discretization and vectors according to nodegraph
    void Redistribute(const Teuchos::RCP<Epetra_CrsGraph>& nodegraph)
    {
      FOUR_C_THROW("You should not need this function!");
      return;
    };

   protected:
    /// update state vectors
    /// current solution becomes old solution of next time step
    void UpdateState() override
    {
      FOUR_C_THROW("You should not need this function!");
      return;
    };

    /// update the solution after Solve()
    /// extended version for coupled level-set problems including reinitialization
    void Update() override
    {
      FOUR_C_THROW("You should not need this function!");
      return;
    };

    /// update phi within the reinitialization loop
    void UpdateReinit() override
    {
      FOUR_C_THROW("You should not need this function!");
      return;
    };

   private:
  };  // class TimIntStationary

}  // namespace SCATRA

FOUR_C_NAMESPACE_CLOSE

#endif