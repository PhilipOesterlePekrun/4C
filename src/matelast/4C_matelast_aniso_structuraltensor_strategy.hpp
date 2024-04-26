/*----------------------------------------------------------------------*/
/*! \file
\brief Implementation of a strategy evaluating the structural tensor in anisotropic materials

\level 2
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_MATELAST_ANISO_STRUCTURALTENSOR_STRATEGY_HPP
#define FOUR_C_MATELAST_ANISO_STRUCTURALTENSOR_STRATEGY_HPP

#define numbgp 50
#define twice 100

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_par_parameter.hpp"

FOUR_C_NAMESPACE_OPEN

namespace MAT
{
  namespace ELASTIC
  {
    namespace PAR
    {
      typedef enum DistrType
      {
        distr_type_undefined = 0,
        distr_type_vonmisesfisher = 1,
        distr_type_bingham = 2
      } DISTR_TYPE_;

      typedef enum StrategyType
      {
        strategy_type_undefined = 0,
        strategy_type_standard = 1,
        strategy_type_bydistributionfunction = 2,
        strategy_type_dispersedtransverselyisotropic = 3
      } STRATEGY_TYPE_;

      /*!
       * @brief material parameters for generalized structural tensor with distribution
       * around mean fiber direction M
       *
       * We assume only one symmetry for the distribution function \rho(M), i.e.:
       * \rho(M) = \rho(-M) .
       *
       * Example input line in .dat file:
       * MAT 2 ELAST_IsoAnisoExpo  K1 1.0E6 K2 100.0 GAMMA 0.0 K1COMP 0.0 K2COMP 0.0 STR_TENS_ID
       * 100 MAT 100 ELAST_StructuralTensor STRATEGY ByDistributionFunction DISTR vonMisesFisher C1
       * 500.0
       */
      class StructuralTensorParameter : public MAT::PAR::Parameter
      {
       public:
        /// standard constructor
        StructuralTensorParameter(const Teuchos::RCP<MAT::PAR::Material>& matdata);

        /// @name material parameters
        //@{
        double c1_;  //!< constant 1 for distribution function
        double c2_;  //!< constant 2 for distribution function
        double c3_;  //!< constant 3 for distribution function
        double c4_;  //!< constant 4 for distribution function
        //@}

        /// type of distribution function around mean fiber direction
        int distribution_type_;

        /// type of structural tensor strategy
        int strategy_type_;

        /// Override this method and throw error, as the material should be created in within the
        /// Factory method of the elastic summand
        Teuchos::RCP<MAT::Material> CreateMaterial() override
        {
          FOUR_C_THROW(
              "Cannot create a material from this method, as it should be created in "
              "MAT::ELASTIC::Summand::Factory.");
          return Teuchos::null;
        };

      };  // class StructuralTensorParameter
    }     // namespace PAR

    /*!
     * @brief Base class for evaluation strategy of structural tensor for anisotropic materials.
     */
    class StructuralTensorStrategyBase
    {
     public:
      /// constructor
      StructuralTensorStrategyBase(MAT::ELASTIC::PAR::StructuralTensorParameter* params);

      /// destructor
      virtual ~StructuralTensorStrategyBase() { ; };

      /*!
       * @brief Method for computing the structural tensor in stress like Voigt notation for
       * anisotropic materials
       *
       * This is the core functionality of this object.
       * Each derived class has to implement this method (pure virtual).
       */
      virtual void SetupStructuralTensor(const CORE::LINALG::Matrix<3, 1>& fiber_vector,
          CORE::LINALG::Matrix<6, 1>& structural_tensor_stress) = 0;

      /*!
       * @brief Method for computing the structural tensor in matrix notation for anisotropic
       * materials
       *
       * This is the core functionality of this object.
       * Each derived class has to implement this method (pure virtual).
       */
      virtual void SetupStructuralTensor(const CORE::LINALG::Matrix<3, 1>& fiber_vector,
          CORE::LINALG::Matrix<3, 3>& structural_tensor) = 0;

      /*!
       * @brief calculate MxM
       *
       * @param[in] M vector (e.g. fiber direction)
       * @param[out] result result of dyadic product
       */
      void DyadicProduct(const CORE::LINALG::Matrix<3, 1>& M, CORE::LINALG::Matrix<6, 1>& result);

      /*!
       * @brief calculate MxM
       *
       * @param[in] M vector (e.g. fiber direction)
       * @param[out] result result of dyadic product
       */
      void DyadicProduct(const CORE::LINALG::Matrix<3, 1>& M, CORE::LINALG::Matrix<3, 3>& result);

     protected:
      /// return residual tolerance of structural problem
      double GetResidualTol();

      /// my material parameters
      MAT::ELASTIC::PAR::StructuralTensorParameter* params_;

    };  // class StructuralTensorStrategyBase


    /*!
     *
     * @brief Class for computing the standard structural tensor for anisotropic materials via
     * adyadic product of the current fiber direction.
     *
     * <h3>Definiton of Structural Tensor H</h3>
     * H = M x M
     *
     * By H we denote the structural tensor. M is the direction of the fiber.
     * Here, x denotes the dyadic product.
     */
    class StructuralTensorStrategyStandard : public StructuralTensorStrategyBase
    {
     public:
      /// constructor with given material parameters
      StructuralTensorStrategyStandard(MAT::ELASTIC::PAR::StructuralTensorParameter* params);

      /*!
       * @brief method for computing the structural tensor for anisotropic materials in stress like
       * Voigt notation
       *
       * Simplest variant assuming perfect alignment of fiber with a given fiber direction M.
       *
       * <h3>Definiton of Structural Tensor H</h3>
       * H = M x M (x denotes the dyadic product)
       *
       * @param fiber_vector (in) : direction of fiber 'M'
       * @param structural_tensor_stress (out) : structural tensor is 'H' filled inside
       */
      void SetupStructuralTensor(const CORE::LINALG::Matrix<3, 1>& fiber_vector,
          CORE::LINALG::Matrix<6, 1>& structural_tensor_stress) override;

      /*!
       * @brief method for computing the structural tensor for anisotropic materials in tensor
       * notation
       *
       * Simplest variant assuming perfect alignment of fiber with a given fiber direction M.
       *
       * <h3>Definiton of Structural Tensor H</h3>
       * H = M x M (x denotes the dyadic product)
       *
       * @param fiber_vector (in) : direction of fiber 'M'
       * @param structural_tensor_stress (out) : structural tensor is 'H' filled inside
       */
      void SetupStructuralTensor(const CORE::LINALG::Matrix<3, 1>& fiber_vector,
          CORE::LINALG::Matrix<3, 3>& structural_tensor) override;
    };  // namespace ELASTIC


    /*!
     * @brief Class for computing the generalized structural tensor
     *
     * Class for computing the generalized forms of the structural tensor for
     * anisotropic materials, by integrating a given distribution function for
     * fiber orientations.
     *
     * See: T.C. Gasser, R.W. Ogden, and G.A. Holzapfel. Hyperelastic modelling of arterial layers
     * with distributed collagen fibre orientations. J. R. Soc. Interface, 3:15-35, 2006. doi:
     * 10.1098/rsif.2005.0073.
     *
     *
     * <h3>Definiton of Structural Tensor H</h3>
     *
     * H = \frac{1}{4\pi}\int \rho(Theta,Phi) [m x m] sin(Theta) dTheta dPhi
     * with
     * m = \sin(\Theta)\cos(\Phi)e_1 + \sin(\Theta)\sin(\Phi)e_2 + \cos(\Theta)e_3 .
     *
     * Our convention for the rotations (cf. [Gasser 2006]) :
     *
     * 1) Vector m'' is initially aligned with e3
     * 2) We rotate m'' with angle Theta away from e3 in e1-e3-plane (e2 is normal to this plane).
     *    We arrive at the vector m'.
     * 3) We rotate the vector m' with angle Phi around e3 to arrive at m.
     *
     *
     * <h3>Distribution functions rho</h3>
     *
     * By \rho we denote the distribution function. For now we can choose between:
     *
     * 1) vonMisesFisher distribution:
     *
     *    This distribution function has only one parameter C1 determining the dispersion
     *    of fibers around the given mean direction M.
     *
     *    \rho = \frac{C1}{4\pi\sinh(C1)} \exp(C1 M \cdot m)
     *
     *    with 0 \leq C1 , |M| = 1 , and |m| = 1
     *
     *    See also: https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
     *
     *  2) Bingham distribution:
     *
     *     This distribution function requires four parameters C1, C2, C3, and C4.
     *
     *     !!! IMPORTANT REMARK :
     *     The definition of angles Theta and Phi in [Gasser et al. 2012] is different from our
     *     definition. This explains the difference in the formulation compared to the equation
     *     reported by Gasser et al. Further, you have to choose C3 = 0.0 to reproduce the reported
     *     distribution.
     *
     *     \rho(\Theta,\Phi) = \frac{1}{C4} \exp(C1 X_1 + C2 X_2 + C3 X_3)
     *
     *     with X_1 = ( sin(theta) * cos(phi) )^2
     *          X_2 = ( sin(theta) * sin(phi) )^2 * (K^2 / (1.0+K^2))
     *          X_3 = cos(theta)^2
     *
     *          K = sin(theta)*cos(phi) / cos(theta)
     *
     *     See:
     *     T.C. Gasser, S. Gallinetti, X. Xing, C. Forsell, J. Swedenborg, and J. Roy. Spatial
     *     orientation of collagen fibers in the abdominal aortic aneurysm's wall and its relation
     *     to wall mechanics Acta Biomater., 8:3091-3103, 2012. doi:
     *     http://dx.doi.org/10.1016/j.actbio.2012.04.044.
     *
     *
     * <h3>Numerical Integration</h3>
     *
     * The integration is performed by an Gauss integration rule in accordance with:
     * Atkinson K.; Numerical Integration on the Sphere.; J. Austral. Math. Soc.B; 23; 1982
     */
    class StructuralTensorStrategyByDistributionFunction : public StructuralTensorStrategyBase
    {
     public:
      /// constructor with given material parameters
      StructuralTensorStrategyByDistributionFunction(
          MAT::ELASTIC::PAR::StructuralTensorParameter* params);

      /*!
       * @brief Evaluate generalized structural tensor with given distribution function
       *
       * Here we integrate the general integral for the structural tensor:
       * H = \frac{1}{4\pi}\int \rho(Theta,Phi) [m x m] sin(Theta) dTheta dPhi
       *
       * wherein H denotes the structural tensor tensor, \rho is the distribution
       * function for the dispersion of fibers around the mean fiber direction.
       *
       * Here, M denotes the direction vector in each direction (Theta,Phi).
       * In case of a perfect alignment of all fibres the integral yields the standard
       *
       * @param[in] fiber_vector mean direction of fiber
       * @param[out] structural_tensor_stress generalized structural tensor in stress-like Voigt
       * notation
       */
      void SetupStructuralTensor(const CORE::LINALG::Matrix<3, 1>& fiber_vector,
          CORE::LINALG::Matrix<6, 1>& structural_tensor_stress) override;

      /*!
       * @brief Evaluate generalized structural tensor with given distribution function in matrix
       * notation
       *
       * @note This method is not yet implemented
       *
       * @param[in] fiber_vector
       * @param[out] structural_tensor
       */
      void SetupStructuralTensor(const CORE::LINALG::Matrix<3, 1>& fiber_vector,
          CORE::LINALG::Matrix<3, 3>& structural_tensor) override;
    };  // class StructuralTensorStrategyByDistributionFunction


    /*!
     * @brief Class for computing dispersed transversely isotropic structural tensor
     *
     * See: T.C. Gasser, R.W. Ogden, and G.A. Holzapfel. Hyperelastic modelling of arterial layers
     * with distributed collagen fibre orientations. J. R. Soc. Interface, 3:15-35, 2006.
     * doi:10.1098/rsif.2005.0073.
     *
     * <h3>Definiton of Structural Tensor H</h3>
     * H = c1*I + (1-3*c1)MxM
     * I = Identity matrix
     * MxM = dyadic product with fiber orientation M
     *
     * c1 is the dispersion parameter:
     * if c1 = 1/3 -> isotropic distribution
     * if c1 = 0.0 -> same as StructuralTensorStrategyStandard
     * if 0<c1<1/3 -> varying transverse isotropy
     */
    class StructuralTensorStrategyDispersedTransverselyIsotropic
        : public StructuralTensorStrategyBase
    {
     public:
      /// constructor with given material parameters
      StructuralTensorStrategyDispersedTransverselyIsotropic(
          MAT::ELASTIC::PAR::StructuralTensorParameter* params);

      /*!
       * @brief evaluate transversely isotropic structural tensor with dispersion
       *
       * H = c1*I + (1-3*c1)MxM
       * I = Identity matrix
       * MxM = dyadic product with fiber orientation M
       *
       * c1 is the dispersion parameter:
       * if c1 = 1/3 -> isotropic distribution
       * if c1 = 0.0 -> same as StructuralTensorStrategyStandard
       * if 0<c1<1/3 -> varying transverse isotropy
       */
      void SetupStructuralTensor(const CORE::LINALG::Matrix<3, 1>& fiber_vector,
          CORE::LINALG::Matrix<6, 1>& structural_tensor_stress) override;

      /*!
       * @brief Evaluates the transversely isotropic structural tensor with dispersin in matrix
       * notation
       *
       * @note: This is not implemented yet.
       *
       * @param fiber_vector
       * @param structural_tensor
       */
      void SetupStructuralTensor(const CORE::LINALG::Matrix<3, 1>& fiber_vector,
          CORE::LINALG::Matrix<3, 3>& structural_tensor) override;
    };  // class StructuralTensorStrategyDispersedTransverselyIsotropic

  }  // namespace ELASTIC
}  // namespace MAT

FOUR_C_NAMESPACE_CLOSE

#endif