/*---------------------------------------------------------------------------*/
/*! \file
\brief momentum handler for smoothed particle hydrodynamics (SPH) interactions
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_MOMENTUM_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_MOMENTUM_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_config.hpp"

#include "baci_inpar_particle.hpp"
#include "baci_particle_engine_enums.hpp"
#include "baci_particle_engine_typedefs.hpp"

BACI_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace PARTICLEENGINE
{
  class ParticleEngineInterface;
  class ParticleContainerBundle;
}  // namespace PARTICLEENGINE

namespace PARTICLEWALL
{
  class WallHandlerInterface;
}

namespace PARTICLEINTERACTION
{
  class SPHKernelBase;
  class MaterialHandler;
  class InteractionWriter;
  class SPHEquationOfStateBundle;
  class SPHNeighborPairs;
  class SPHVirtualWallParticle;
  class SPHMomentumFormulationBase;
  class SPHArtificialViscosity;
}  // namespace PARTICLEINTERACTION

namespace MAT
{
  namespace PAR
  {
    class ParticleMaterialSPHFluid;
  }
}  // namespace MAT

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace PARTICLEINTERACTION
{
  class SPHMomentum final
  {
   public:
    //! constructor
    explicit SPHMomentum(const Teuchos::ParameterList& params);

    /*!
     * \brief destructor
     *
     * \author Sebastian Fuchs \date 09/2018
     *
     * \note At compile-time a complete type of class T as used in class member
     *       std::unique_ptr<T> ptr_T_ is required
     */
    ~SPHMomentum();

    //! init momentum handler
    void Init();

    //! setup momentum handler
    void Setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
        const std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface,
        const std::shared_ptr<PARTICLEINTERACTION::SPHKernelBase> kernel,
        const std::shared_ptr<PARTICLEINTERACTION::MaterialHandler> particlematerial,
        const std::shared_ptr<PARTICLEINTERACTION::InteractionWriter> particleinteractionwriter,
        const std::shared_ptr<PARTICLEINTERACTION::SPHEquationOfStateBundle> equationofstatebundle,
        const std::shared_ptr<PARTICLEINTERACTION::SPHNeighborPairs> neighborpairs,
        const std::shared_ptr<PARTICLEINTERACTION::SPHVirtualWallParticle> virtualwallparticle);

    //! insert momentum evaluation dependent states
    void InsertParticleStatesOfParticleTypes(
        std::map<PARTICLEENGINE::TypeEnum, std::set<PARTICLEENGINE::StateEnum>>&
            particlestatestotypes) const;

    //! add momentum contribution to acceleration field
    void AddAccelerationContribution() const;

   private:
    //! init momentum formulation handler
    void InitMomentumFormulationHandler();

    //! init artificial viscosity handler
    void InitArtificialViscosityHandler();

    //! setup particle interaction writer
    void SetupParticleInteractionWriter();

    //! momentum equation (particle contribution)
    void MomentumEquationParticleContribution() const;

    //! momentum equation (particle-boundary contribution)
    void MomentumEquationParticleBoundaryContribution() const;

    //! momentum equation (particle-wall contribution)
    void MomentumEquationParticleWallContribution() const;

    //! smoothed particle hydrodynamics specific parameter list
    const Teuchos::ParameterList& params_sph_;

    //! interface to particle engine
    std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface_;

    //! particle container bundle
    PARTICLEENGINE::ParticleContainerBundleShrdPtr particlecontainerbundle_;

    //! interface to particle wall handler
    std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface_;

    //! kernel handler
    std::shared_ptr<PARTICLEINTERACTION::SPHKernelBase> kernel_;

    //! particle material handler
    std::shared_ptr<PARTICLEINTERACTION::MaterialHandler> particlematerial_;

    //! particle interaction writer
    std::shared_ptr<PARTICLEINTERACTION::InteractionWriter> particleinteractionwriter_;

    //! equation of state bundle
    std::shared_ptr<PARTICLEINTERACTION::SPHEquationOfStateBundle> equationofstatebundle_;

    //! neighbor pair handler
    std::shared_ptr<PARTICLEINTERACTION::SPHNeighborPairs> neighborpairs_;

    //! virtual wall particle handler
    std::shared_ptr<PARTICLEINTERACTION::SPHVirtualWallParticle> virtualwallparticle_;

    //! momentum formulation handler
    std::unique_ptr<PARTICLEINTERACTION::SPHMomentumFormulationBase> momentumformulation_;

    //! artificial viscosity handler
    std::unique_ptr<PARTICLEINTERACTION::SPHArtificialViscosity> artificialviscosity_;

    //! type of boundary particle interaction
    INPAR::PARTICLE::BoundaryParticleInteraction boundaryparticleinteraction_;

    //! type of transport velocity formulation
    INPAR::PARTICLE::TransportVelocityFormulation transportvelocityformulation_;

    //! pointer to fluid material of particle types
    std::vector<const MAT::PAR::ParticleMaterialSPHFluid*> fluidmaterial_;

    //! write particle-wall interaction output
    const bool writeparticlewallinteraction_;

    //! set of all fluid particle types
    std::set<PARTICLEENGINE::TypeEnum> allfluidtypes_;

    //! set of integrated fluid particle types
    std::set<PARTICLEENGINE::TypeEnum> intfluidtypes_;

    //! set of pure fluid particle types
    std::set<PARTICLEENGINE::TypeEnum> purefluidtypes_;

    //! set of boundary particle types
    std::set<PARTICLEENGINE::TypeEnum> boundarytypes_;
  };

}  // namespace PARTICLEINTERACTION

/*---------------------------------------------------------------------------*/
BACI_NAMESPACE_CLOSE

#endif