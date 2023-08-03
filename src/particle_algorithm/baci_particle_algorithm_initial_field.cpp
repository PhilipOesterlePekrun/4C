/*---------------------------------------------------------------------------*/
/*! \file
\brief initial field handler for particle simulations
\level 2
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_particle_algorithm_initial_field.H"

#include "baci_lib_globalproblem.H"
#include "baci_particle_algorithm_utils.H"
#include "baci_particle_engine_container.H"
#include "baci_particle_engine_container_bundle.H"
#include "baci_particle_engine_enums.H"
#include "baci_particle_engine_interface.H"

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEALGORITHM::InitialFieldHandler::InitialFieldHandler(const Teuchos::ParameterList& params)
    : params_(params)
{
  // empty constructor
}

void PARTICLEALGORITHM::InitialFieldHandler::Init()
{
  // get control parameters for initial/boundary conditions
  const Teuchos::ParameterList& params_conditions =
      params_.sublist("INITIAL AND BOUNDARY CONDITIONS");

  // relate particle state to input name
  std::map<std::string, PARTICLEENGINE::StateEnum> initialfieldtostateenum = {
      std::make_pair("INITIAL_TEMP_FIELD", PARTICLEENGINE::Temperature),
      std::make_pair("INITIAL_VELOCITY_FIELD", PARTICLEENGINE::Velocity),
      std::make_pair("INITIAL_ACCELERATION_FIELD", PARTICLEENGINE::Acceleration)};

  // iterate over particle states
  for (auto& stateIt : initialfieldtostateenum)
  {
    // get reference to sub-map
    std::map<PARTICLEENGINE::TypeEnum, int>& currentstatetypetofunctidmap =
        statetotypetofunctidmap_[stateIt.second];

    // read parameters relating particle types to values
    PARTICLEALGORITHM::UTILS::ReadParamsTypesRelatedToValues(
        params_conditions, stateIt.first, currentstatetypetofunctidmap);
  }
}

void PARTICLEALGORITHM::InitialFieldHandler::Setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;
}

void PARTICLEALGORITHM::InitialFieldHandler::SetInitialFields()
{
  // get particle container bundle
  PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle =
      particleengineinterface_->GetParticleContainerBundle();

  for (auto& stateIt : statetotypetofunctidmap_)
  {
    // get state of particles
    PARTICLEENGINE::StateEnum particleState = stateIt.first;

    // iterate over particle types
    for (auto& initialFieldIt : stateIt.second)
    {
      // get type of particles
      PARTICLEENGINE::TypeEnum particleType = initialFieldIt.first;

      // get container of owned particles of current particle type
      PARTICLEENGINE::ParticleContainer* container =
          particlecontainerbundle->GetSpecificContainer(particleType, PARTICLEENGINE::Owned);

      // get number of particles stored in container
      const int particlestored = container->ParticlesStored();

      // no owned particles of current particle type
      if (particlestored <= 0) continue;

      // get id of function
      const int functid = initialFieldIt.second;

      // get reference to function
      DRT::UTILS::FunctionOfSpaceTime& function =
          DRT::Problem::Instance()->FunctionById<DRT::UTILS::FunctionOfSpaceTime>(functid - 1);

      // get pointer to particle states
      const double* pos = container->GetPtrToState(PARTICLEENGINE::Position, 0);
      double* state = container->GetPtrToState(particleState, 0);

      // get particle state dimensions
      int posstatedim = container->GetStateDim(PARTICLEENGINE::Position);
      int statedim = container->GetStateDim(particleState);

      // safety check
      if (static_cast<std::size_t>(statedim) != function.NumberComponents())
        dserror("dimensions of function defining initial field and of state '%s' not matching!",
            PARTICLEENGINE::EnumToStateName(particleState).c_str());

      // iterate over owned particles of current type
      for (int i = 0; i < particlestored; ++i)
      {
        // evaluate function to set initial field
        for (int dim = 0; dim < statedim; ++dim)
          state[statedim * i + dim] = function.Evaluate(&(pos[posstatedim * i]), 0.0, dim);
      }
    }
  }
}