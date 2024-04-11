/*----------------------------------------------------------------------*/
/*! \file
\brief Definitions of classes for the exponential shear behavior for fibers

\level 3
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_MATELAST_COUPANISOEXPOSHEAR_HPP
#define FOUR_C_MATELAST_COUPANISOEXPOSHEAR_HPP

#include "baci_config.hpp"

#include "baci_mat_anisotropy_extension_default.hpp"
#include "baci_mat_anisotropy_extension_provider.hpp"
#include "baci_matelast_coupanisoexpobase.hpp"

BACI_NAMESPACE_OPEN


namespace MAT
{
  namespace ELASTIC
  {
    /*!
     * \brief Container class for communication with the base material
     */
    class CoupAnisoExpoShearAnisotropyExtension : public BaseAnisotropyExtension,
                                                  public CoupAnisoExpoBaseInterface
    {
     public:
      /*!
       * \brief Constructor
       *
       * \param init_mode initialization mode of the fibers
       * \param gamma Angle of the fiber if they are given in a local coordinate system
       * \param adapt_angle boolean, whether the fiber is subject to growth and remodeling
       * \param structuralTensorStrategy Strategy to compute the structural tensor
       * \param fiber_ids Ids of the fiber to be used for shear behavior
       */
      CoupAnisoExpoShearAnisotropyExtension(int init_mode, std::array<int, 2> fiber_ids);

      ///@name Packing and Unpacking
      /// @{
      void PackAnisotropy(CORE::COMM::PackBuffer& data) const override;

      void UnpackAnisotropy(
          const std::vector<char>& data, std::vector<char>::size_type& position) override;
      /// @}

      double GetScalarProduct(int gp) const override;
      const CORE::LINALG::Matrix<3, 3>& GetStructuralTensor(int gp) const override;
      const CORE::LINALG::Matrix<6, 1>& GetStructuralTensor_stress(int gp) const override;

      /*!
       * /copydoc
       *
       * The coupling structural tensor and the scalar product will be computed here
       */
      void OnGlobalDataInitialized() override;

     protected:
      void OnGlobalElementDataInitialized() override;
      void OnGlobalGPDataInitialized() override;

     private:
      /**
       * Scalar products of the fibers at the Gauss points
       */
      std::vector<double> scalarProducts_;

      /**
       * Coupling structural tensor of the fibers in stress like Voigt notation at the Gauss points
       */
      std::vector<CORE::LINALG::Matrix<6, 1>> structuralTensors_stress_;

      /**
       * Coupling structural tensor of the fibers at the Gauss points
       */
      std::vector<CORE::LINALG::Matrix<3, 3>> structuralTensors_;

      /// Flag whether fibers are initialized
      bool isInitialized_{};

      /// Initialization mode
      const int init_mode_;

      /// Fiber ids to be used to build the shear behavior
      const std::array<int, 2> fiber_ids_;
    };

    namespace PAR
    {
      /*!
       * @brief material parameters for coupled contribution of a anisotropic exponential fiber
       * material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_CoupAnisoExpoShear K1 10.0 K2 1.0 K1COMP 0.0 K2COMP 1.0 INIT 0 FIBER_IDS 1 2
       */
      class CoupAnisoExpoShear : public MAT::PAR::Parameter,
                                 public MAT::ELASTIC::PAR::CoupAnisoExpoBase
      {
       public:
        /// standard constructor
        explicit CoupAnisoExpoShear(const Teuchos::RCP<MAT::PAR::Material>& matdata);

        /// Override this method and throw error, as the material should be created in within the
        /// Factory method of the elastic summand
        Teuchos::RCP<MAT::Material> CreateMaterial() override
        {
          dserror(
              "Cannot create a material from this method, as it should be created in "
              "MAT::ELASTIC::Summand::Factory.");
          return Teuchos::null;
        };

        /// @name material parameters
        //@{
        /// Ids of the fiber for the shear behavior
        std::array<int, 2> fiber_id_{};
        //@}

      };  // class CoupAnisoExpoShear

    }  // namespace PAR

    /*!
     * \brief Exponential shear behavior between two fibers
     *
     * The strain energy function of this summand is
     * \[\psi = \frac{a_{fs}}{2b_{fs}} \left[ \exp( b_{fs} (I_{8fs} - \boldsymbol{f} \cdot
     * \boldsymbol{s})^2 ) - 1 \right]\]
     */
    class CoupAnisoExpoShear : public MAT::ELASTIC::CoupAnisoExpoBase
    {
     public:
      /// constructor with given material parameters
      explicit CoupAnisoExpoShear(MAT::ELASTIC::PAR::CoupAnisoExpoShear* params);

      /// @name Access material constants
      //@{

      /// material type
      INPAR::MAT::MaterialType MaterialType() const override
      {
        return INPAR::MAT::mes_coupanisoexposhear;
      }

      //@}

      /// @name Methods for Packing and Unpacking
      ///@{
      void PackSummand(CORE::COMM::PackBuffer& data) const override;

      void UnpackSummand(
          const std::vector<char>& data, std::vector<char>::size_type& position) override;
      ///@}

      /*!
       * \brief Register the anisotropy extension to the global anisotropy manager
       *
       * \param anisotropy anisotropy manager
       */
      void RegisterAnisotropyExtensions(MAT::Anisotropy& anisotropy) override;

      /// Set fiber directions
      void SetFiberVecs(double newgamma,             ///< new angle
          const CORE::LINALG::Matrix<3, 3>& locsys,  ///< local coordinate system
          const CORE::LINALG::Matrix<3, 3>& defgrd   ///< deformation gradient
          ) override;

      /// Get fiber directions
      void GetFiberVecs(
          std::vector<CORE::LINALG::Matrix<3, 1>>& fibervecs  ///< vector of all fiber vectors
          ) override;

     protected:
      const CoupAnisoExpoBaseInterface& GetCoupAnisoExpoBaseInterface() const override
      {
        return anisotropyExtension_;
      }

     private:
      /// my material parameters
      MAT::ELASTIC::PAR::CoupAnisoExpoShear* params_;

      /// Internal ansotropy information
      MAT::ELASTIC::CoupAnisoExpoShearAnisotropyExtension anisotropyExtension_;
    };

  }  // namespace ELASTIC
}  // namespace MAT

BACI_NAMESPACE_CLOSE

#endif