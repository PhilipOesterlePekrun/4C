/*----------------------------------------------------------------------*/
/*! \file

\brief implementation of the 3D solid-poro element


\level 2

*----------------------------------------------------------------------*/
#ifndef FOUR_C_SO3_PORO_HPP
#define FOUR_C_SO3_PORO_HPP

#include "4C_config.hpp"

#include "4C_discretization_fem_general_utils_gausspoints.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_lib_element.hpp"

FOUR_C_NAMESPACE_OPEN

namespace MAT
{
  class StructPoro;
  class FluidPoro;
  class FluidPoroMultiPhase;
}  // namespace MAT

namespace DRT
{
  class Discretization;

  namespace ELEMENTS
  {
    /*!
    \brief A C++ version of a 3 dimensional solid element with modifications for porous media

    A structural 3 dimensional solid displacement element for large deformations
    and (near)-incompressibility.

    */
    template <class so3_ele, CORE::FE::CellType distype>
    class So3Poro : public so3_ele
    {
      //! @name Friends
      friend class SoTet4PoroType;
      friend class SoTet10PoroType;
      friend class SoHex8PoroType;
      friend class SoHex27PoroType;

     public:
      //!@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owner
      */
      So3Poro(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      So3Poro(const So3Poro& old);

      //!@}

      //! number of element nodes (
      static constexpr int numnod_ = CORE::FE::num_nodes<distype>;

      //! number of space dimensions
      static constexpr int numdim_ = CORE::FE::dim<distype>;

      //! number of dofs per node
      static constexpr int noddof_ = numdim_;

      //! total dofs per element
      static constexpr int numdof_ = noddof_ * numnod_;

      //! number of strains per node
      static constexpr int numstr_ = (numdim_ * (numdim_ + 1)) / 2;

      //! number of components necessary to store second derivatives
      /*!
       1 component  for nsd=1:  (N,xx)

       3 components for nsd=2:  (N,xx ; N,yy ; N,xy)

       6 components for nsd=3:  (N,xx ; N,yy ; N,zz ; N,xy ; N,xz ; N,yz)
      */
      static constexpr int numderiv2_ = CORE::FE::DisTypeToNumDeriv2<distype>::numderiv2;

      //! total gauss points per element
      int numgpt_;

      //! @name Acess methods

      /*!
      \brief Deep copy this instance of Solid3 and return pointer to the copy

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      DRT::Element* Clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int UniqueParObjectId() const override;

      /*!
      \brief Pack this class so it can be communicated

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Pack(CORE::COMM::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Unpack(const std::vector<char>& data) override;

      /*!
      \brief Get vector of Teuchos::RCPs to the lines of this element

      */
      std::vector<Teuchos::RCP<DRT::Element>> Lines() override;

      /*!
      \brief Get vector of Teuchos::RCPs to the surfaces of this element

      */
      std::vector<Teuchos::RCP<DRT::Element>> Surfaces() override;

      //! @name Access methods

      /*!
      \brief Print this element
      */
      void Print(std::ostream& os) const override;

      DRT::ElementType& ElementType() const override;

      //!@}

      /*!
      \brief Return value how expensive it is to evaluate this element

      \param double (out): cost to evaluate this element
      */
      double EvaluationCost() override { return 10000.0; }

      //! @name Evaluation

      /*!
      \brief Evaluate an element

      Evaluate element stiffness, mass, internal forces, etc.

      If nullptr on input, the controlling method does not expect the element
      to fill these matrices or vectors.

      \return 0 if successful, negative otherwise
      */
      int Evaluate(
          Teuchos::ParameterList&
              params,  //!< ParameterList for communication between control routine and elements
          DRT::Discretization& discretization,  //!< pointer to discretization for de-assembly
          DRT::Element::LocationArray& la,      //!< location array for de-assembly
          CORE::LINALG::SerialDenseMatrix&
              elemat1,  //!< (stiffness-)matrix to be filled by element.
          CORE::LINALG::SerialDenseMatrix& elemat2,  //!< (mass-)matrix to be filled by element.
          CORE::LINALG::SerialDenseVector&
              elevec1,  //!< (internal force-)vector to be filled by element
          CORE::LINALG::SerialDenseVector& elevec2,  //!< vector to be filled by element
          CORE::LINALG::SerialDenseVector& elevec3   //!< vector to be filled by element
          ) override;

      virtual void PreEvaluate(
          Teuchos::ParameterList&
              params,  //!< ParameterList for communication between control routine and elements
          DRT::Discretization& discretization,  //!< pointer to discretization for de-assembly
          DRT::Element::LocationArray& la       //!< location array for de-assembly
      );


      //! initialize the inverse of the jacobian and its determinant in the material configuration
      virtual void InitElement();

      //!@}

      //! @name Input and Creation

      /*!
      \brief Read input for this element
      */
      /*!
      \brief Query names of element data to be visualized using BINIO

      The element fills the provided map with key names of
      visualization data the element wants to visualize AT THE CENTER
      of the element geometry. The values is supposed to be dimension of the
      data to be visualized. It can either be 1 (scalar), 3 (vector), 6 (sym. tensor)
      or 9 (nonsym. tensor)

      Example:
      \code
        // Name of data is 'Owner', dimension is 1 (scalar value)
        names.insert(std::pair<std::string,int>("Owner",1));
        // Name of data is 'StressesXYZ', dimension is 6 (sym. tensor value)
        names.insert(std::pair<std::string,int>("StressesXYZ",6));
      \endcode

      \param names (out): On return, the derived class has filled names with
                          key names of data it wants to visualize and with int dimensions
                          of that data.
      */
      void VisNames(std::map<std::string, int>& names) override;

      /*!
      \brief Query data to be visualized using BINIO of a given name

      The method is supposed to call this base method to visualize the owner of
      the element.
      If the derived method recognizes a supported data name, it shall fill it
      with corresponding data.
      If it does NOT recognizes the name, it shall do nothing.

      \warning The method must not change size of data

      \param name (in):   Name of data that is currently processed for visualization
      \param data (out):  data to be filled by element if element recognizes the name
      */
      bool VisData(const std::string& name, std::vector<double>& data) override;

      /*!
      \brief Read input for this element
      */
      bool ReadElement(const std::string& eletype, const std::string& eledistype,
          INPUT::LineDefinition* linedef) override;

      //!@}

      //! compute porosity at gausspoint
      virtual void ComputePorosity(Teuchos::ParameterList& params, double press, double J, int gp,
          double& porosity, double* dphi_dp, double* dphi_dJ, double* dphi_dJdp, double* dphi_dJJ,
          double* dphi_dpp, bool save);

      //! compute porosity at gauss point at one face of the element
      virtual void ComputeSurfPorosity(Teuchos::ParameterList& params, double press, double J,
          int surfnum, int gp, double& porosity, double* dphi_dp, double* dphi_dJ,
          double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, bool save);

      //! return time derivative of reference porosity
      virtual double RefPorosityTimeDeriv();

      //! evaluate Cauchy stress at given point in parameter space
      virtual void GetCauchyNDirAndDerivativesAtXi(const CORE::LINALG::Matrix<3, 1>& xi,
          const std::vector<double>& disp, const std::vector<double>& pres,
          const CORE::LINALG::Matrix<3, 1>& n, const CORE::LINALG::Matrix<3, 1>& dir,
          double& cauchy_n_dir, CORE::LINALG::SerialDenseMatrix* d_cauchyndir_dd,
          CORE::LINALG::SerialDenseMatrix* d_cauchyndir_dp,
          CORE::LINALG::Matrix<3, 1>* d_cauchyndir_dn,
          CORE::LINALG::Matrix<3, 1>* d_cauchyndir_ddir,
          CORE::LINALG::Matrix<3, 1>* d_cauchyndir_dxi);

      //! return anisotropic permeability directions (used for cloning)
      const std::vector<std::vector<double>>& GetAnisotropicPermeabilityDirections() const
      {
        return anisotropic_permeability_directions_;
      }

      //! return scaling coefficients for anisotropic permeability (used for cloning)
      const std::vector<std::vector<double>>& GetAnisotropicPermeabilityNodalCoeffs() const
      {
        return anisotropic_permeability_nodal_coeffs_;
      }

      //! don't want = operator
      So3Poro& operator=(const So3Poro& old) = delete;

     protected:
      /*!
      \brief Evaluate an element

      Evaluate So3_poro element stiffness, mass, internal forces, etc.
      Templated evaluate routine of element matrixes

      If nullptr on input, the controlling method does not expect the element
      to fill these matrices or vectors.

      \return 0 if successful, negative otherwise
      */
      virtual int MyEvaluate(
          Teuchos::ParameterList&
              params,  //!< ParameterList for communication between control routine and elements
          DRT::Discretization& discretization,  //!< pointer to discretization for de-assembly
          DRT::Element::LocationArray& la,      //!< location array for de-assembly
          CORE::LINALG::SerialDenseMatrix&
              elemat1,  //!< (stiffness-)matrix to be filled by element.
          CORE::LINALG::SerialDenseMatrix& elemat2,  //!< (mass-)matrix to be filled by element.
          CORE::LINALG::SerialDenseVector&
              elevec1,  //!< (internal force-)vector to be filled by element
          CORE::LINALG::SerialDenseVector& elevec2,  //!< vector to be filled by element
          CORE::LINALG::SerialDenseVector& elevec3   //!< vector to be filled by element
      );

      //! compute porosity at gausspoint and linearization of porosity w.r.t. structural
      //! displacements
      virtual void ComputePorosityAndLinearization(Teuchos::ParameterList& params,
          const double& press, const double& J, const int& gp,
          const CORE::LINALG::Matrix<numnod_, 1>& shapfct,
          const CORE::LINALG::Matrix<numnod_, 1>* myporosity,
          const CORE::LINALG::Matrix<1, numdof_>& dJ_dus, double& porosity,
          CORE::LINALG::Matrix<1, numdof_>& dphi_dus);

      //! compute porosity at gausspoint and linearization of porosity w.r.t. fluid pressure
      virtual void ComputePorosityAndLinearizationOD(Teuchos::ParameterList& params,
          const double& press, const double& J, const int& gp,
          const CORE::LINALG::Matrix<numnod_, 1>& shapfct,
          const CORE::LINALG::Matrix<numnod_, 1>* myporosity, double& porosity, double& dphi_dp);

      //! action parameters recognized by so_hex8
      enum ActionType
      {
        none,
        calc_struct_nlnstiff,
        calc_struct_internalforce,
        calc_struct_nlnstiffmass,
        calc_struct_stress,
        calc_struct_multidofsetcoupling,  //!< structure-fluid coupling: internal force, stiffness
                                          //!< for poroelasticity (structural part)
        interpolate_porosity_to_given_point
      };

      //! vector of inverses of the jacobian in material frame
      std::vector<CORE::LINALG::Matrix<numdim_, numdim_>> invJ_;
      //! determinant of Jacobian in material frame
      std::vector<double> detJ_;
      //! vector of coordinates of current integration point in reference coordinates
      std::vector<CORE::LINALG::Matrix<numdim_, 1>> xsi_;

      //! Calculate nonlinear stiffness and internal force for poroelasticity problems
      void NonlinearStiffnessPoroelast(std::vector<int>& lm,    //!< location matrix
          CORE::LINALG::Matrix<numdim_, numnod_>& disp,         //! current displacements
          CORE::LINALG::Matrix<numdim_, numnod_>& vel,          //! current velocities
          CORE::LINALG::Matrix<numdim_, numnod_>& evelnp,       //! fluid velocity of element
          CORE::LINALG::Matrix<numnod_, 1>& epreaf,             //! fluid pressure of element
          CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix,  //! element stiffness matrix
          CORE::LINALG::Matrix<numdof_, numdof_>* reamatrix,    //! element reactive matrix
          CORE::LINALG::Matrix<numdof_, 1>* force,              //! element internal force vector
          Teuchos::ParameterList& params                        //! algorithmic parameters e.g. time
      );

      //! Calculate nonlinear stiffness and internal force for poroelasticity problems (pressure
      //! based formulation)
      virtual void NonlinearStiffnessPoroelastPressureBased(
          std::vector<int>& lm,                          //!< location matrix
          CORE::LINALG::Matrix<numdim_, numnod_>& disp,  //! current displacements
          const std::vector<double>& ephi,  //! current primary variable for poro-multiphase flow
          CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix,  //! element stiffness matrix
          CORE::LINALG::Matrix<numdof_, 1>* force,              //! element internal force vector
          Teuchos::ParameterList& params                        //! algorithmic parameters e.g. time
      );

      //! Calculate coupling terms in nonlinear stiffness and internal force for poroelasticity
      //! problems
      void CouplingPoroelast(std::vector<int>& lm,         //!< location matrix
          CORE::LINALG::Matrix<numdim_, numnod_>& disp,    //! current displacements
          CORE::LINALG::Matrix<numdim_, numnod_>& vel,     //! current velocities
          CORE::LINALG::Matrix<numdim_, numnod_>& evelnp,  //! fluid velocity of element
          CORE::LINALG::Matrix<numnod_, 1>& epreaf,        //! fluid pressure of element
          CORE::LINALG::Matrix<numdof_, (numdim_ + 1) * numnod_>*
              stiffmatrix,  //! element stiffness matrix
          CORE::LINALG::Matrix<numdof_, (numdim_ + 1) * numnod_>*
              reamatrix,                            //! element reactive matrix
          CORE::LINALG::Matrix<numdof_, 1>* force,  //! element internal force vector
          Teuchos::ParameterList& params            //! algorithmic parameters e.g. time
      );

      //! Calculate coupling terms in nonlinear stiffness and internal force for poroelasticity
      //! problems (pressure based formulation)
      virtual void CouplingPoroelastPressureBased(std::vector<int>& lm,  //!< location matrix
          CORE::LINALG::Matrix<numdim_, numnod_>& disp,                  //! current displacements
          const std::vector<double>& ephi,  //! current primary variable for poro-multiphase flow
          CORE::LINALG::SerialDenseMatrix& couplmat,  //!< element stiffness matrix
          Teuchos::ParameterList& params              //!< algorithmic parameters e.g. time
      );

      //! Calculate coupling stress for poroelasticity problems
      virtual void CouplingStressPoroelast(
          CORE::LINALG::Matrix<numdim_, numnod_>& disp,    //! current displacements
          CORE::LINALG::Matrix<numdim_, numnod_>& evelnp,  //! fluid velocity of element
          CORE::LINALG::Matrix<numnod_, 1>& epreaf,        //! fluid pressure of element
          CORE::LINALG::SerialDenseMatrix* elestress,      //! stresses at GP
          CORE::LINALG::SerialDenseMatrix* elestrain,      //! strains at GP
          Teuchos::ParameterList& params,                  //! algorithmic parameters e.g. time
          const INPAR::STR::StressType iostress            //! stress output option
      );

      //! Gauss Point Loop evaluating stiffness and force
      void GaussPointLoop(Teuchos::ParameterList& params,
          const CORE::LINALG::Matrix<numdim_, numnod_>& xrefe,
          const CORE::LINALG::Matrix<numdim_, numnod_>& xcurr,
          const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp,
          const CORE::LINALG::Matrix<numdim_, numnod_>& nodalvel,
          const CORE::LINALG::Matrix<numdim_, numnod_>& evelnp,
          const CORE::LINALG::Matrix<numnod_, 1>& epreaf,
          const CORE::LINALG::Matrix<numnod_, 1>* porosity_dof,
          CORE::LINALG::Matrix<numdof_, numdof_>& erea_v,
          CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix,
          CORE::LINALG::Matrix<numdof_, 1>* force);

      //! Gauss Point Loop evaluating stiffness and force
      void GaussPointLoopPressureBased(Teuchos::ParameterList& params,
          const CORE::LINALG::Matrix<numdim_, numnod_>& xrefe,
          const CORE::LINALG::Matrix<numdim_, numnod_>& xcurr,
          const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp, const std::vector<double>& ephi,
          CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix,
          CORE::LINALG::Matrix<numdof_, 1>* force);

      //! Gauss Point Loop evaluating stiffness (off diagonal)
      void GaussPointLoopODPressureBased(Teuchos::ParameterList& params,
          const CORE::LINALG::Matrix<numdim_, numnod_>& xrefe,
          const CORE::LINALG::Matrix<numdim_, numnod_>& xcurr,
          const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp, const std::vector<double>& ephi,
          CORE::LINALG::SerialDenseMatrix& couplmat);

      //! fill stiffness matrix and rhs vector for darcy flow
      void FillMatrixAndVectors(const int& gp, const CORE::LINALG::Matrix<numnod_, 1>& shapefct,
          const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& press,
          const double& porosity, const CORE::LINALG::Matrix<numdim_, 1>& velint,
          const CORE::LINALG::Matrix<numdim_, 1>& fvelint,
          const CORE::LINALG::Matrix<numdim_, numdim_>& fvelder,
          const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv,
          const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
          const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
          const CORE::LINALG::Matrix<numdim_, 1>& Finvgradp,
          const CORE::LINALG::Matrix<1, numdof_>& dphi_dus,
          const CORE::LINALG::Matrix<1, numdof_>& dJ_dus,
          const CORE::LINALG::Matrix<numstr_, numdof_>& dCinv_dus,
          const CORE::LINALG::Matrix<numdim_, numdof_>& dFinvdus_gradp,
          const CORE::LINALG::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
          CORE::LINALG::Matrix<numdof_, numdof_>& erea_v,
          CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix,
          CORE::LINALG::Matrix<numdof_, 1>* force, CORE::LINALG::Matrix<numstr_, 1>& fstress);

      //! fill stiffness matrix and rhs vector for brinkman flow
      void FillMatrixAndVectorsBrinkman(const int& gp, const double& J, const double& porosity,
          const CORE::LINALG::Matrix<numdim_, numdim_>& fvelder,
          const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv,
          const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
          const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
          const CORE::LINALG::Matrix<1, numdof_>& dphi_dus,
          const CORE::LINALG::Matrix<1, numdof_>& dJ_dus,
          const CORE::LINALG::Matrix<numstr_, numdof_>& dCinv_dus,
          const CORE::LINALG::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
          CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix,
          CORE::LINALG::Matrix<numdof_, 1>* force, CORE::LINALG::Matrix<numstr_, 1>& fstress);

      //! fill stiffness matrix and rhs vector for pressure-based formulation
      void FillMatrixAndVectorsPressureBased(const int& gp,
          const CORE::LINALG::Matrix<numnod_, 1>& shapefct,
          const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& press,
          const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
          const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
          const CORE::LINALG::Matrix<1, numdof_>& dJ_dus,
          const CORE::LINALG::Matrix<numstr_, numdof_>& dCinv_dus,
          const CORE::LINALG::Matrix<1, numdof_>& dps_dus,
          CORE::LINALG::Matrix<numdof_, numdof_>* stiffmatrix,
          CORE::LINALG::Matrix<numdof_, 1>* force);

      //! fill stiffness matrix and rhs vector for darcy flow (off diagonal terms)
      void FillMatrixAndVectorsOD(const int& gp, const CORE::LINALG::Matrix<numnod_, 1>& shapefct,
          const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ, const double& J,
          const double& porosity, const double& dphi_dp,
          const CORE::LINALG::Matrix<numdim_, 1>& velint,
          const CORE::LINALG::Matrix<numdim_, 1>& fvelint,
          const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv,
          const CORE::LINALG::Matrix<numdim_, 1>& Gradp,
          const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
          const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
          CORE::LINALG::Matrix<numdof_, (numdim_ + 1) * numnod_>* stiffmatrix);

      //! fill stiffness matrix (off diagonal terms) -- pressure-based formulation
      void FillMatrixAndVectorsODPressureBased(const int& gp,
          const CORE::LINALG::Matrix<numnod_, 1>& shapefct,
          const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ, const double& J,
          const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
          const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
          const std::vector<double>& solpressderiv, CORE::LINALG::SerialDenseMatrix& couplmat);

      //! fill stiffness matrix and rhs vector for darcy brinkman flow (off diagonal terms)
      void FillMatrixAndVectorsBrinkmanOD(const int& gp,
          const CORE::LINALG::Matrix<numnod_, 1>& shapefct,
          const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ, const double& J,
          const double& porosity, const double& dphi_dp,
          const CORE::LINALG::Matrix<numdim_, numdim_>& fvelder,
          const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv,
          const CORE::LINALG::Matrix<numstr_, numdof_>& bop,
          const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
          CORE::LINALG::Matrix<numdof_, (numdim_ + 1) * numnod_>* stiffmatrix);

      //! Gauss Point Loop evaluating stiffness (off diagonal)
      void GaussPointLoopOD(Teuchos::ParameterList& params,
          const CORE::LINALG::Matrix<numdim_, numnod_>& xrefe,
          const CORE::LINALG::Matrix<numdim_, numnod_>& xcurr,
          const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp,
          const CORE::LINALG::Matrix<numdim_, numnod_>& nodalvel,
          const CORE::LINALG::Matrix<numdim_, numnod_>& evelnp,
          const CORE::LINALG::Matrix<numnod_, 1>& epreaf,
          CORE::LINALG::Matrix<numdof_, (numdim_ + 1) * numnod_>* stiffmatrix);

      //! helper functions to get element vectors from global vector
      void ExtractValuesFromGlobalVector(
          const DRT::Discretization& discretization,             //!< discretization
          const int& dofset,                                     //!< number of dofset
          const std::vector<int>& lm,                            //!< location vector
          CORE::LINALG::Matrix<numdim_, numnod_>* matrixtofill,  //!< vector field
          CORE::LINALG::Matrix<numnod_, 1>* vectortofill,        //!< scalar field
          const std::string& state                               //!< state of the global vector
      );

      //! push forward of material stresses to the current, spatial configuration
      void PK2toCauchy(CORE::LINALG::Matrix<numstr_, 1>& stress,
          CORE::LINALG::Matrix<numdim_, numdim_>& defgrd,
          CORE::LINALG::Matrix<numdim_, numdim_>& cauchystress);

      //! get materials (solid and fluid)
      void GetMaterials();

      //! get materials (solid and fluid) for pressure based formulation
      void GetMaterialsPressureBased();

      //! Compute Solid-pressure derivative w.r.t. primary variable at GP
      void ComputeSolPressureDeriv(const std::vector<double>& phiAtGP,  //!<< primary variable
          const int numfluidphases,                                     //!<< number of fluid phases
          std::vector<double>& solidpressderiv  //!<< solid pressure derivative at GP
      );

      //! Compute solid pressure at GP
      double ComputeSolPressureAtGP(
          const int totalnumdofpernode,       //!<< total number of multiphase dofs
          const int numfluidphases,           //!<< number of fluid phases
          const std::vector<double>& phiAtGP  //!<< primary variable
      );

      //! recalculate solid pressure at GP in case of volfracs
      double RecalculateSolPressureAtGP(double press, const double porosity,
          const int totalnumdofpernode, const int numfluidphases, const int numvolfrac,
          const std::vector<double>& phiAtGP);

      //! recalculate solid pressure derivative in case of volfracs
      void RecalculateSolPressureDeriv(const std::vector<double>& phiAtGP,
          const int totalnumdofpernode, const int numfluidphases, const int numvolfrac,
          const double press, const double porosity, std::vector<double>& solidpressderiv);

      //! Compute primary variable for multiphase flow at GP
      void ComputePrimaryVariableAtGP(
          const std::vector<double>& ephi,                   //!<< primary variable at node
          const int totalnumdofpernode,                      //!<< total number of multiphase dofs
          const CORE::LINALG::Matrix<numnod_, 1>& shapefct,  //!<< shapefct
          std::vector<double>& phiAtGP                       //!<< primary variable at GP
      );

      //! Compute linearizaton of solid press w.r.t. displacements
      //! only needed if additional volume fractions are present and porosity depends on
      //! Jacobian of deformation gradient
      void ComputeLinearizationOfSolPressWrtDisp(const double fluidpress, const double porosity,
          const int totalnumdofpernode, const int numfluidphases, const int numvolfrac,
          const std::vector<double>& phiAtGP, const CORE::LINALG::Matrix<1, numdof_>& dphi_dus,
          CORE::LINALG::Matrix<1, numdof_>& dps_dus);

      //! evaluate shape functions and their derivatives at gauss point
      void ComputeShapeFunctionsAndDerivatives(const int& gp,
          CORE::LINALG::Matrix<numnod_, 1>& shapefct, CORE::LINALG::Matrix<numdim_, numnod_>& deriv,
          CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ);

      //! Compute Jacobian Determinant and the volume change
      void ComputeJacobianDeterminantVolumeChange(double& J, double& volchange,
          const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd,
          const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ,
          const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp);

      //! Compute Jacobian Determinant and the volume change and its linearizations
      virtual void ComputeJacobianDeterminantVolumeChangeAndLinearizations(
          double& J,          //!< (o) Jacobian Determinant
          double& volchange,  //!< (o) Change of Volume
          CORE::LINALG::Matrix<1, numdof_>&
              dJ_dus,  //!< (o) Linearization of Jacobian Determinant  w.r.t displacements
          CORE::LINALG::Matrix<1, numdof_>&
              dvolchange_dus,  //!<  (o) Linearization of Change of Volume  w.r.t displacements
          const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd,  //!< (i) deformation gradient
          const CORE::LINALG::Matrix<numdim_, numdim_>&
              defgrd_inv,  //!< (i) inverse of deformation gradient
          const CORE::LINALG::Matrix<numdim_, numnod_>&
              N_XYZ,  //!< (i) spatial gradient of material shape functions
          const CORE::LINALG::Matrix<numdim_, numnod_>& nodaldisp  //!<  (i) nodal displacements
      );

      //! Compute Linearization Of Jacobian
      void ComputeLinearizationOfJacobian(CORE::LINALG::Matrix<1, numdof_>& dJ_dus, const double& J,
          const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ,
          const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv);

      //! Compute Linearization some Auxiliary Values in gauss point loop
      void ComputeAuxiliaryValues(const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ,
          const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd_inv,
          const CORE::LINALG::Matrix<numdim_, numdim_>& C_inv,
          const CORE::LINALG::Matrix<numdim_, 1>& Gradp,
          CORE::LINALG::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
          CORE::LINALG::Matrix<numdim_, 1>& Finvgradp,
          CORE::LINALG::Matrix<numdim_, numdof_>& dFinvdus_gradp,
          CORE::LINALG::Matrix<numstr_, numdof_>& dCinv_dus);

      //! Compute  nonlinear b-operator
      void ComputeBOperator(CORE::LINALG::Matrix<numstr_, numdof_>& bop,
          const CORE::LINALG::Matrix<numdim_, numdim_>& defgrd,
          const CORE::LINALG::Matrix<numdim_, numnod_>& N_XYZ);

      //! Compute deformation gradient
      void ComputeDefGradient(CORE::LINALG::Matrix<numdim_, numdim_>&
                                  defgrd,  //!<<    (i) deformation gradient at gausspoint
          const CORE::LINALG::Matrix<numdim_, numnod_>&
              N_XYZ,  //!<<    (i) derivatives of shape functions w.r.t. reference coordinates
          const CORE::LINALG::Matrix<numdim_, numnod_>&
              xcurr  //!<<    (i) current position of gausspoint
      );

      //! read anisotropic permeability directions in the element definition
      void ReadAnisotropicPermeabilityDirectionsFromElementLineDefinition(
          INPUT::LineDefinition* linedef);

      //! read nodal anisotropic permeability scaling coefficients in the element definition
      void ReadAnisotropicPermeabilityNodalCoeffsFromElementLineDefinition(
          INPUT::LineDefinition* linedef);

      //! interpolate the anisotropic permeability coefficients at GP from nodal values
      std::vector<double> ComputeAnisotropicPermeabilityCoeffsAtGP(
          const CORE::LINALG::Matrix<numnod_, 1>& shapefct) const;

      //! Gauss integration rule
      CORE::FE::GaussIntegration intpoints_;

      //! flag indicating if element has been initialized
      bool init_;

      //! flag for scatra coupling
      bool scatra_coupling_;

      //! flag for nurbs
      bool isNurbs_;

      //! weights for nurbs elements
      CORE::LINALG::Matrix<numnod_, 1> weights_;

      //! knot vector for nurbs elements
      std::vector<CORE::LINALG::SerialDenseVector> myknots_;

      //! corresponding fluid material
      Teuchos::RCP<MAT::FluidPoro> fluid_mat_;

      //! corresponding multiphase fluid material
      Teuchos::RCP<MAT::FluidPoroMultiPhase> fluidmulti_mat_;

      //! own poro structure material
      Teuchos::RCP<MAT::StructPoro> struct_mat_;

      //! directions for anisotropic permeability
      std::vector<std::vector<double>> anisotropic_permeability_directions_;

      //! scaling coefficients for nodal anisotropic permeability
      std::vector<std::vector<double>> anisotropic_permeability_nodal_coeffs_;

      //! get nodes of element
      DRT::Node** Nodes() override;

      //! get material of element
      Teuchos::RCP<MAT::Material> Material() const;

      //! get global id of element
      int Id() const;
    };
  }  // namespace ELEMENTS
}  // namespace DRT


FOUR_C_NAMESPACE_CLOSE

#endif