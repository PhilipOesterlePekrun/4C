/*-----------------------------------------------------------*/
/*! \file

\brief factory for time integration base strategy and data container


\level 3

*/
/*-----------------------------------------------------------*/


#include "baci_structure_new_timint_factory.H"

#include "baci_inpar_structure.H"
#include "baci_lib_globalproblem.H"
#include "baci_lib_prestress_service.H"
#include "baci_structure_new_timint_basedatasdyn.H"
#include "baci_structure_new_timint_explicit.H"
#include "baci_structure_new_timint_implicit.H"
#include "baci_structure_new_timint_loca_continuation.H"

#include <Teuchos_ParameterList.hpp>


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
STR::TIMINT::Factory::Factory()
{
  // empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::Base> STR::TIMINT::Factory::BuildStrategy(
    const Teuchos::ParameterList& sdyn) const
{
  Teuchos::RCP<STR::TIMINT::Base> ti_strategy = Teuchos::null;

  const enum INPAR::STR::IntegrationStrategy intstrat =
      DRT::INPUT::IntegralValue<INPAR::STR::IntegrationStrategy>(sdyn, "INT_STRATEGY");

  switch (intstrat)
  {
    case INPAR::STR::int_standard:
    {
      // Check first if a implicit integration strategy is desired
      ti_strategy = BuildImplicitStrategy(sdyn);
      // If there was no suitable implicit time integrator check for the
      // explicit case
      if (ti_strategy.is_null()) ti_strategy = BuildExplicitStrategy(sdyn);
      break;
    }
    case INPAR::STR::int_loca:
      ti_strategy = Teuchos::rcp(new STR::TIMINT::LOCAContinuation());
      break;
    default:
      dserror("Unknown integration strategy!");
      break;
  }

  return ti_strategy;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::Base> STR::TIMINT::Factory::BuildImplicitStrategy(
    const Teuchos::ParameterList& sdyn) const
{
  Teuchos::RCP<STR::TIMINT::Base> ti_strategy = Teuchos::null;

  // get the dynamic type
  const enum INPAR::STR::DynamicType dyntype =
      DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP");

  if (::UTILS::PRESTRESS::IsAny() or dyntype == INPAR::STR::dyna_statics or  // dynamic type
      dyntype == INPAR::STR::dyna_genalpha or dyntype == INPAR::STR::dyna_genalpha_liegroup or
      dyntype == INPAR::STR::dyna_onesteptheta or dyntype == INPAR::STR::dyna_gemm)
    ti_strategy = Teuchos::rcp(new STR::TIMINT::Implicit());

  return ti_strategy;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::Base> STR::TIMINT::Factory::BuildExplicitStrategy(
    const Teuchos::ParameterList& sdyn) const
{
  Teuchos::RCP<STR::TIMINT::Base> ti_strategy = Teuchos::null;

  // what's the current problem type?
  ProblemType probtype = DRT::Problem::Instance()->GetProblemType();

  if (probtype == ProblemType::fsi or probtype == ProblemType::fsi_redmodels or
      probtype == ProblemType::fsi_lung or probtype == ProblemType::gas_fsi or
      probtype == ProblemType::ac_fsi or probtype == ProblemType::biofilm_fsi or
      probtype == ProblemType::thermo_fsi)
    dserror("No explicit time integration with fsi");

  const enum INPAR::STR::DynamicType dyntype =
      DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP");

  if (dyntype == INPAR::STR::dyna_expleuler or dyntype == INPAR::STR::dyna_centrdiff or
      dyntype == INPAR::STR::dyna_ab2)
    //    ti_strategy = Teuchos::rcp(new STR::TIMINT::Explicit());
    dserror("Explicit time integration scheme is not yet implemented!");

  return ti_strategy;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::BaseDataSDyn> STR::TIMINT::Factory::BuildDataSDyn(
    const Teuchos::ParameterList& sdyn) const
{
  Teuchos::RCP<STR::TIMINT::BaseDataSDyn> sdyndata_ptr = Teuchos::null;

  const enum INPAR::STR::DynamicType dyntype =
      DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(sdyn, "DYNAMICTYP");

  switch (dyntype)
  {
    case INPAR::STR::dyna_genalpha:
    case INPAR::STR::dyna_genalpha_liegroup:
      sdyndata_ptr = Teuchos::rcp(new STR::TIMINT::GenAlphaDataSDyn());
      break;
    case INPAR::STR::dyna_onesteptheta:
      sdyndata_ptr = Teuchos::rcp(new STR::TIMINT::OneStepThetaDataSDyn());
      break;
    default:
      sdyndata_ptr = Teuchos::rcp(new STR::TIMINT::BaseDataSDyn());
      break;
  }

  return sdyndata_ptr;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::BaseDataGlobalState> STR::TIMINT::Factory::BuildDataGlobalState() const
{
  return Teuchos::rcp(new STR::TIMINT::BaseDataGlobalState());
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::Base> STR::TIMINT::BuildStrategy(const Teuchos::ParameterList& sdyn)
{
  Factory factory;
  return factory.BuildStrategy(sdyn);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::BaseDataSDyn> STR::TIMINT::BuildDataSDyn(
    const Teuchos::ParameterList& sdyn)
{
  Factory factory;
  return factory.BuildDataSDyn(sdyn);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Teuchos::RCP<STR::TIMINT::BaseDataGlobalState> STR::TIMINT::BuildDataGlobalState()
{
  Factory factory;
  return factory.BuildDataGlobalState();
}