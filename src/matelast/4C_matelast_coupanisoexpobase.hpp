/*----------------------------------------------------------------------*/
/*! \file
\brief Definition of the base functionality of an exponential anisotropic summand

\level 1
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_MATELAST_COUPANISOEXPOBASE_HPP
#define FOUR_C_MATELAST_COUPANISOEXPOBASE_HPP

#include "4C_config.hpp"

#include "4C_mat_anisotropy_extension_base.hpp"
#include "4C_mat_anisotropy_extension_provider.hpp"
#include "4C_mat_par_parameter.hpp"
#include "4C_matelast_summand.hpp"

FOUR_C_NAMESPACE_OPEN


namespace MAT
{
  namespace ELASTIC
  {
    /*!
     * \brief Pure abstract class to define the interface to the implementation of specific
     * materials
     */
    class CoupAnisoExpoBaseInterface
    {
     public:
      virtual ~CoupAnisoExpoBaseInterface() = default;

      /*!
       * \brief Returns the scalar product of the fibers
       *
       * \param gp Gauss point
       * \return double
       */
      virtual double GetScalarProduct(int gp) const = 0;

      /*!
       * \brief Returns the structural tensor that should be used
       *
       * \param gp Gauss point
       * \return const CORE::LINALG::Matrix<3, 3>&
       */
      virtual const CORE::LINALG::Matrix<3, 3>& GetStructuralTensor(int gp) const = 0;

      /*!
       * \brief Returns the structural tensor in stress like Voigt notation that should be used
       *
       * \param gp Gauss point
       * \return const CORE::LINALG::Matrix<6, 1>&
       */
      virtual const CORE::LINALG::Matrix<6, 1>& GetStructuralTensor_stress(int gp) const = 0;
    };

    namespace PAR
    {
      /*!
       * @brief material parameters for coupled contribution of a anisotropic exponential fiber
       * material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_CoupAnisoExpo K1 10.0 K2 1.0 GAMMA 35.0 K1COMP 0.0 K2COMP 1.0 INIT 0
       * ADAPT_ANGLE 0
       */
      class CoupAnisoExpoBase
      {
       public:
        /// standard constructor
        explicit CoupAnisoExpoBase(const Teuchos::RCP<MAT::PAR::Material>& matdata);

        /// Constructor only used for unit testing
        CoupAnisoExpoBase();

        /// @name material parameters
        //@{

        /// fiber params
        double k1_;
        double k2_;
        /// angle between circumferential and fiber direction (used for cir, axi, rad nomenclature)
        double gamma_;
        /// fiber params for the compressible case
        double k1comp_;
        double k2comp_;
        /// fiber initalization status
        int init_;

        //@}
      };  // class CoupAnisoExpoBase

    }  // namespace PAR

    /*!
     * @brief Coupled anisotropic exponential fiber function, implemented for one possible fiber
     * family [1] This is a hyperelastic, anisotropic material of the most simple kind.
     *
     * Strain energy function is given by
     * \f[
     *   \Psi = \frac {k_1}{2 k_2} \left(e^{k_2 (IV_{\boldsymbol C}-1)^2 }-1 \right).
     * \f]
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] G.A. Holzapfel, T.C. Gasser, R.W. Ogden: A new constitutive framework for arterial
     * wall mechanics and a comparative study of material models, J. of Elasticity 61 (2000) 1-48.
     * </ul>
     */
    class CoupAnisoExpoBase : public Summand
    {
     public:
      /// constructor with given material parameters
      explicit CoupAnisoExpoBase(MAT::ELASTIC::PAR::CoupAnisoExpoBase* params);

      /*!
       * \brief Evaluate first derivative of the strain energy function with respect to the
       * anisotropic invariants.
       *
       * \param dPI_aniso (out) : First derivatives of the strain energy function with respect to
       * the anisotropic invariants
       * \param C (in) : Cauchy Green deformation tensor
       * \param gp (in) : Gauss point
       * \param eleGID (in) : global element id
       */
      void EvaluateFirstDerivativesAniso(CORE::LINALG::Matrix<2, 1>& dPI_aniso,
          const CORE::LINALG::Matrix<3, 3>& rcg, int gp, int eleGID) override;

      /*!
       * \brief Evaluate second derivative of the strain energy function with respect to the
       * anisotropic invariants.
       *
       * \param ddPI_aniso (out) : Second derivatives of the strain energy function with respect to
       * the anisotropic invariants
       * \param C (in) : Cauchy Green deformation tensor
       * \param gp (in) : Gauss point
       * \param eleGID (in) : global element id
       */
      void EvaluateSecondDerivativesAniso(CORE::LINALG::Matrix<3, 1>& ddPII_aniso,
          const CORE::LINALG::Matrix<3, 3>& rcg, int gp, int eleGID) override;


      /*!
       * @brief retrieve coefficients of first, second and third derivative of summand with respect
       * to anisotropic invariants
       *
       * The derivatives of the summand
       * \f$\Psi(IV_{\boldsymbol{C},\boldsymbol{a}},V_{\boldsymbol{C},\boldsymbol{a}})\f$ in which
       * the principal anisotropic invariants are the arguments are defined as following:
       *
       * First derivatives:
       *
       * \f[
       * dPI_{0,aniso} = \frac{\partial \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * \f[
       * dPI_{1,aniso} = \frac{\partial \Psi}{\partial V_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * Second derivatives:
       * \f[
       * ddPII_{0,aniso} = \frac{\partial^2 \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}^2} ;
       * \f]
       * \f[
       * ddPII_{1,aniso} = \frac{\partial^2 \Psi}{\partial V_{\boldsymbol{C},\boldsymbol{a}}^2} ;
       * \f]
       * \f[
       * ddPII_{2,aniso} = \frac{\partial^2 \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}
       * \partial V_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * Third derivatives:
       * \f[
       * dddPIII_{0,aniso} = \frac{\partial^3 \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}
       * \partial IV_{\boldsymbol{C},\boldsymbol{a}} \partial IV_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * \f[
       * dddPIII_{1,aniso} = \frac{\partial^3 \Psi}{\partial V_{\boldsymbol{C},\boldsymbol{a}}
       * \partial V_{\boldsymbol{C},\boldsymbol{a}} \partial V_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * \f[
       * dddPIII_{2,aniso} = \frac{\partial^3 \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}
       * \partial IV_{\boldsymbol{C},\boldsymbol{a}} \partial V_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * \f[
       * dddPIII_{3,aniso} = \frac{\partial^3 \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}
       * \partial V_{\boldsymbol{C},\boldsymbol{a}} \partial V_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       */
      template <typename T>
      void GetDerivativesAniso(CORE::LINALG::Matrix<2, 1, T>&
                                   dPI_aniso,  ///< first derivative with respect to invariants
          CORE::LINALG::Matrix<3, 1, T>&
              ddPII_aniso,  ///< second derivative with respect to invariants
          CORE::LINALG::Matrix<4, 1, T>&
              dddPIII_aniso,  ///< third derivative with respect to invariants
          CORE::LINALG::Matrix<3, 3, T> const& rcg,  ///< right Cauchy-Green tensor
          int gp,                                    ///< Gauss point
          int eleGID) const;                         ///< element GID

      /// Add anisotropic principal stresses
      void AddStressAnisoPrincipal(
          const CORE::LINALG::Matrix<6, 1>& rcg,  ///< right Cauchy Green Tensor
          CORE::LINALG::Matrix<6, 6>& cmat,       ///< material stiffness matrix
          CORE::LINALG::Matrix<6, 1>& stress,     ///< 2nd PK-stress
          Teuchos::ParameterList&
              params,  ///< additional parameters for computation of material properties
          int gp,      ///< Gauss point
          int eleGID   ///< element GID
          ) override;

      /// add strain energy
      void AddStrainEnergy(double& psi,  ///< strain energy functions
          const CORE::LINALG::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          const CORE::LINALG::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          const CORE::LINALG::Matrix<6, 1>&
              glstrain,  ///< Green-Lagrange strain in strain like Voigt notation
          int gp,        //< Gauss point
          int eleGID     ///< element GID
          ) override;

      /// Evaluates strain energy for automatic differentiation with FAD
      template <typename T>
      void EvaluateFunc(T& psi,                    ///< strain energy functions
          CORE::LINALG::Matrix<3, 3, T> const& C,  ///< Right Cauchy-Green tensor
          int gp,                                  ///< Gauss point
          int eleGID) const;                       ///< element GID

      /// Set fiber directions
      void SetFiberVecs(double newgamma,             ///< new angle
          const CORE::LINALG::Matrix<3, 3>& locsys,  ///< local coordinate system
          const CORE::LINALG::Matrix<3, 3>& defgrd   ///< deformation gradient
          ) override;

      /// Set fiber directions
      void SetFiberVecs(const CORE::LINALG::Matrix<3, 1>& fibervec  ///< new fiber vector
          ) override;

      /// Get fiber directions
      void GetFiberVecs(
          std::vector<CORE::LINALG::Matrix<3, 1>>& fibervecs  ///< vector of all fiber vectors
          ) override;

      /// Indicator for formulation
      void SpecifyFormulation(
          bool& isoprinc,     ///< global indicator for isotropic principal formulation
          bool& isomod,       ///< global indicator for isotropic splitted formulation
          bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
          bool& anisomod,     ///< global indicator for anisotropic splitted formulation
          bool& viscogeneral  ///< global indicator, if one viscoelastic formulation is used
          ) override
      {
        anisoprinc = true;
      };

     protected:
      /*!
       * \brief Get the structural tensor interface from the derived materials
       *
       * \return const CoupAnisoExpoBaseInterface& Interface that computes structural tensors and
       * scalar products
       */
      virtual const CoupAnisoExpoBaseInterface& GetCoupAnisoExpoBaseInterface() const = 0;

     private:
      /// my material parameters
      MAT::ELASTIC::PAR::CoupAnisoExpoBase* params_;
    };  // namespace ELASTIC

  }  // namespace ELASTIC
}  // namespace MAT

FOUR_C_NAMESPACE_CLOSE

#endif