/*----------------------------------------------------------------------*/
/*! \file
\brief Solid Tet4 Element
\level 3
*----------------------------------------------------------------------*/
#ifndef FOUR_C_SO3_TET4AV_HPP
#define FOUR_C_SO3_TET4AV_HPP


#include "4C_config.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_lib_element.hpp"
#include "4C_lib_elementtype.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_mat_material.hpp"
#include "4C_so3_base.hpp"


// #define SYMMETRIC
// #define SYMMETRIC_C

#define NUMNOD_SOTET4av 4       ///< number of nodes
#define NODDOF_SOTET4av 4       ///< number of dofs per node
#define NUMDOF_SOTET4av 16      ///< total dofs per element
#define NUMGPT_SOTET4av 1       ///< total gauss points per element  /****/
#define NUMDIM_SOTET4av 3       ///< number of dimensions/****/
#define NUMCOORD_SOTET4av 3     ///< number of shape function cooordinates (ksi1-ksi4)
#define NUMNOD_SOTET4av_FACE 3  ///< number of nodes on a TET4 face (which is a TRI3)
#define NUMGPT_SOTET4av_FACE 1  ///< number of GP on a TET4 face (which is a TRI3)

FOUR_C_NAMESPACE_OPEN

namespace DRT
{
  // forward declarations
  class Discretization;

  namespace ELEMENTS
  {
    class SoTet4avType : public DRT::ElementType
    {
     public:
      std::string Name() const override { return "So_tet4avType"; }

      static SoTet4avType& Instance();

      CORE::COMM::ParObject* Create(const std::vector<char>& data) override;

      Teuchos::RCP<DRT::Element> Create(const std::string eletype, const std::string eledistype,
          const int id, const int owner) override;

      Teuchos::RCP<DRT::Element> Create(const int id, const int owner) override;

      int Initialize(DRT::Discretization& dis) override;

      void NodalBlockInformation(
          DRT::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      CORE::LINALG::SerialDenseMatrix ComputeNullSpace(
          DRT::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      void SetupElementDefinition(
          std::map<std::string, std::map<std::string, INPUT::LineDefinition>>& definitions)
          override;

     private:
      static SoTet4avType instance_;

      std::string GetElementTypeString() const { return "SOLIDT4AV"; }
    };

    /*!
    \brief A C++ version of the 4-node tet solid element

    */

    class SoTet4av : public SoBase
    {
     public:
      //! @name Friends
      friend class SoTet4avType;


      //@}
      //! @name Constructors and destructors and related methods

      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner : elements owning processor
      */
      SoTet4av(int id, int owner);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      SoTet4av(const SoTet4av& old);

      /*!
      \brief Deep copy this instance of Solid3 and return pointer to the copy

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      DRT::Element* Clone() const override;

      /*!
      \brief Get shape type of element
      */
      CORE::FE::CellType Shape() const override;

      /*!
      \brief Return number of volumes of this element
      */
      int NumVolume() const override { return 1; }

      /*!
      \brief Return number of surfaces of this element
      */
      int NumSurface() const override { return 4; }

      /*!
      \brief Return number of lines of this element
      */
      int NumLine() const override { return 6; }

      /*!
      \brief Get vector of Teuchos::RCPs to the lines of this element

      */
      std::vector<Teuchos::RCP<DRT::Element>> Lines() override;

      /*!
      \brief Get vector of Teuchos::RCPs to the surfaces of this element

      */
      std::vector<Teuchos::RCP<DRT::Element>> Surfaces() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int UniqueParObjectId() const override
      {
        return SoTet4avType::Instance().UniqueParObjectId();
      }

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

      //@}

      //! @name Acess methods


      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual DRT::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can redecide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int NumDofPerNode(const DRT::Node& node) const override { return 4; }

      /*!
      \brief Get number of degrees of freedom per element
             (implements pure virtual DRT::Element)

      The element decides how many element degrees of freedom it has.
      It can redecide along the way of a simulation.

      \note Element degrees of freedom mentioned here are dofs that are visible
            at the level of the total system of equations. Purely internal
            element dofs that are condensed internally should NOT be considered.
      */
      int NumDofPerElement() const override { return 0; }

      /*!
      \brief Print this element
      */
      void Print(std::ostream& os) const override;

      DRT::ElementType& ElementType() const override { return SoTet4avType::Instance(); }

      //@}

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

      //@}

      //! @name Input and Creation

      /*!
      \brief Read input for this element
      */
      bool ReadElement(const std::string& eletype, const std::string& distype,
          INPUT::LineDefinition* linedef) override;


      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element

      Evaluate so_tet4 element stiffness, mass, internal forces, etc.

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param discretization : pointer to discretization for de-assembly
      \param lm (in)        : location matrix for de-assembly
      \param elemat1 (out)  : (stiffness-)matrix to be filled by element. If nullptr on input,
                              the controling method does not expect the element to fill
                              this matrix.
      \param elemat2 (out)  : (mass-)matrix to be filled by element. If nullptr on input,
                              the controling method does not expect the element to fill
                              this matrix.
      \param elevec1 (out)  : (internal force-)vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int Evaluate(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          std::vector<int>& lm, CORE::LINALG::SerialDenseMatrix& elemat1,
          CORE::LINALG::SerialDenseMatrix& elemat2, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseVector& elevec2,
          CORE::LINALG::SerialDenseVector& elevec3) override;


      /*!
      \brief Evaluate a Neumann boundary condition

      this method evaluates a surface Neumann condition on the solid3 element

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): A reference to the underlying discretization
      \param condition (in)     : The condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

      \return 0 if successful, negative otherwise
      */
      int EvaluateNeumann(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          DRT::Condition& condition, std::vector<int>& lm, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseMatrix* elemat1 = nullptr) override;

      /*!
      \brief Return value how expensive it is to evaluate this element

      \param double (out): cost to evaluate this element
      */
      double EvaluationCost() override
      {
        if (Material()->MaterialType() == INPAR::MAT::m_struct_multiscale)
          return 25000.0;
        else
          return 10.0;
      }

      //@}

     protected:
      //! action parameters recognized by so_tet4av
      enum ActionType
      {
        none,
        calc_struct_nlnstiff,
        calc_struct_internalforce,
        calc_struct_nlnstiffmass,
        calc_struct_stress,
        calc_struct_update_istep,
        calc_struct_reset_istep,  //!< reset elementwise internal variables
                                  //!< during iteration to last converged state
        calc_struct_reset_all,    //!< reset elementwise internal variables
                                  //!< to state in the beginning of the computation
      };

      //! vector of inverses of the jacobian in material frame
      std::vector<CORE::LINALG::Matrix<3, 3>> invJ_;
      //! determinant of Jacobian in material frame
      std::vector<double> detJ_;
      //! vector of coordinates of current integration point in reference coordinates
      std::vector<CORE::LINALG::Matrix<3, 1>> xsi_;
      //! Gauss point weights
      std::vector<double> wgt_;
      int numgpt_{};

      // internal calculation methods

      // don't want = operator
      SoTet4av& operator=(const SoTet4av& old);

      //! init the inverse of the jacobian and its determinant in the material configuration
      virtual void InitJacobianMapping();

      //! Calculate nonlinear stiffness and mass matrix
      virtual void nlnstiffmass(std::vector<int>& lm, std::vector<double>& disp,
          CORE::LINALG::Matrix<NUMDOF_SOTET4av, NUMDOF_SOTET4av>* stiffmatrix,
          CORE::LINALG::Matrix<NUMDOF_SOTET4av, NUMDOF_SOTET4av>* massmatrix,
          CORE::LINALG::Matrix<NUMDOF_SOTET4av, 1>* force,
          CORE::LINALG::Matrix<NUMGPT_SOTET4av, MAT::NUM_STRESS_3D>* elestress,
          CORE::LINALG::Matrix<NUMGPT_SOTET4av, MAT::NUM_STRESS_3D>* elestrain,
          Teuchos::ParameterList& params, const INPAR::STR::StressType iostress,
          const INPAR::STR::StrainType iostrain);

      //@}

     private:
      std::string GetElementTypeString() const { return "SOLIDT4AV"; }
    };  // class So_tet4


  }  // namespace ELEMENTS
}  // namespace DRT

FOUR_C_NAMESPACE_CLOSE

#endif