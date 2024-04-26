/*----------------------------------------------------------------------*/
/*! \file
 \brief porous fluid multiphase material. Contains and defines the single phases.

        The input line should read (for example: 4 fluid phases, 2 volume fractions)

        MAT 0 MAT_FluidPoroMultiPhase LOCAL No PERMEABILITY 1.0 NUMMAT 8 MATIDS 1 2 3 4 5 6 7 8
        NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE 4 END with: 4 fluid phases in multiphase pores pace:
        materials have to be MAT_FluidPoroSinglePhase | 2 volume fractions: materials have to be
        MAT_FluidPoroSingleVolFrac | 2 volume fraction pressures: materials have to be
        MAT_FluidPoroVolFracPressure

\level 3

 *----------------------------------------------------------------------*/

#ifndef FOUR_C_MAT_FLUIDPORO_MULTIPHASE_HPP
#define FOUR_C_MAT_FLUIDPORO_MULTIPHASE_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_material.hpp"
#include "4C_mat_par_parameter.hpp"

FOUR_C_NAMESPACE_OPEN

namespace MAT
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for list of single fluid phases
    class FluidPoroMultiPhase : public MatList
    {
     public:
      /// standard constructor
      FluidPoroMultiPhase(Teuchos::RCP<MAT::PAR::Material> matdata);

      /// create material instance of matching type with my parameters
      Teuchos::RCP<MAT::Material> CreateMaterial() override;

      /// initialize the material
      virtual void Initialize();

      /// @name material parameters
      //@{
      /// permeability
      const double permeability_;

      /// number of fluid phases of the nummat
      const int numfluidphases_;

      /// number of volume fractions of the nummat
      int numvolfrac_;

      //@}

      //! transformation of degrees of freedom to true pressures
      Teuchos::RCP<CORE::LINALG::SerialDenseMatrix> dof2pres_;

      //! number of constraint saturation phase
      int constraintphaseID_;

      //! initialize flag
      bool isinit_;

    };  // class FluidPoroMultiPhase

  }  // namespace PAR

  class FluidPoroMultiPhaseType : public CORE::COMM::ParObjectType
  {
   public:
    std::string Name() const override { return "FluidPoroMultiPhaseType"; }

    static FluidPoroMultiPhaseType& Instance() { return instance_; };

    CORE::COMM::ParObject* Create(const std::vector<char>& data) override;

   private:
    static FluidPoroMultiPhaseType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for a list of porous flow phases
  class FluidPoroMultiPhase : public MatList
  {
   public:
    /// construct empty material object
    FluidPoroMultiPhase();

    /// construct the material object given material parameters
    explicit FluidPoroMultiPhase(MAT::PAR::FluidPoroMultiPhase* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int UniqueParObjectId() const override
    {
      return FluidPoroMultiPhaseType::Instance().UniqueParObjectId();
    }

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique parobject id delivered by UniqueParObjectId() which will then
      identify the exact class on the receiving processor.

      \param data (in/out): char vector to store class information
    */
    void Pack(CORE::COMM::PackBuffer& data) const override;

    /*!
      \brief Unpack data from a char vector into this class

      The vector data contains all information to rebuild the
      exact copy of an instance of a class on a different processor.
      The first entry in data has to be an integer which is the unique
      parobject id defined at the top of this file and delivered by
      UniqueParObjectId().

      \param data (in) : vector storing all data to be unpacked into this
      instance.
    */
    void Unpack(const std::vector<char>& data) override;

    //@}

    /// material type
    INPAR::MAT::MaterialType MaterialType() const override
    {
      return INPAR::MAT::m_fluidporo_multiphase;
    }

    /// return copy of this material object
    Teuchos::RCP<Material> Clone() const override
    {
      return Teuchos::rcp(new FluidPoroMultiPhase(*this));
    }

    /// return permeability
    double Permeability() const { return paramsporo_->permeability_; }

    /// return number of fluid phases
    int NumFluidPhases() const { return paramsporo_->numfluidphases_; }

    /// return number of volume fractions
    int NumVolFrac() const { return paramsporo_->numvolfrac_; }

    /// Return quick accessible material parameter data
    MAT::PAR::FluidPoroMultiPhase* Parameter() const override { return paramsporo_; }

    /// initialize the material
    virtual void Initialize();

    /// return whether reaction terms need to be evaluated
    virtual bool IsReactive() const { return false; };

    /// evaluate the generalized(!) pressure and saturation of all phases
    void EvaluateGenPressureAndSaturation(
        std::vector<double>& genpressure, const std::vector<double>& phinp) const;

    /// evaluate the generalized(!) pressure of all phases
    void EvaluateGenPressure(
        std::vector<double>& genpressure, const std::vector<double>& phinp) const;

    /// evaluate saturation of all phases
    void EvaluateSaturation(std::vector<double>& saturation, const std::vector<double>& phinp,
        const std::vector<double>& pressure) const;

    //! transform generalized pressures to true pressures
    void TransformGenPresToTruePres(
        const std::vector<double>& phi, std::vector<double>& phi_transformed) const;

    //! evaluate derivative of degree of freedom with respect to pressure
    void EvaluateDerivOfDofWrtPressure(
        CORE::LINALG::SerialDenseMatrix& derivs, const std::vector<double>& state) const;

    //! evaluate derivative of saturation with respect to pressure
    void EvaluateDerivOfSaturationWrtPressure(
        CORE::LINALG::SerialDenseMatrix& derivs, const std::vector<double>& pressure) const;

    //! evaluate second derivative of saturation with respect to pressure
    void EvaluateSecondDerivOfSaturationWrtPressure(
        std::vector<CORE::LINALG::SerialDenseMatrix>& derivs,
        const std::vector<double>& pressure) const;

   private:
    /// clear everything
    void Clear();

    /// my material parameters
    MAT::PAR::FluidPoroMultiPhase* paramsporo_;
  };

}  // namespace MAT


FOUR_C_NAMESPACE_CLOSE

#endif